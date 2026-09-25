"""FastAPI service exposing the pipeline over HTTP.

    uvicorn umlsmatch.service:app --port 8000

Design notes worth knowing before changing anything here:

**Never log note text.** Every request body is PHI. Handlers log document
counts, character totals and durations -- never content, never a snippet, never
an annotation's matched text. An exception handler that echoed the offending
input would quietly write clinical narrative into a log aggregator, which is
exactly the kind of leak nobody notices until an audit, so failures are reported
by shape ("document 3 of 12") rather than by value.

The one place that rule is not absolute is an exception raised *inside* spaCy
or sqlite3, which may quote the input that broke it. No response ever carries
such a message; whether the **log** does is
:data:`LOG_EXCEPTION_DETAIL`, default on. Leave it on and application logs are
PHI and must be handled as such; turn it off and a failure is logged as its
type and call path with every message dropped.

**No outbound network calls.** The pipeline reads a local SQLite dictionary and
a locally-installed spaCy model. The service is intended to run egress-blocked
by design, so nothing here fetches a model, phones a licence
server, or emits telemetry.

**Liveness and readiness are different questions.** ``/health`` answers "is this
process running" and stays cheap and always-200 once the app is up. ``/ready``
answers "can it serve a request", which is false for the seconds -- or minutes,
on a cold page cache -- while the pool builds. Wiring an orchestrator's
readiness probe to ``/health`` would route traffic into a service that is up but
cannot yet analyze anything, and its liveness probe to ``/ready`` would restart
a healthy service that is merely still warming up.

**Analysis runs in the threadpool, not the event loop.** The handlers are
``def``, not ``async def``, so Starlette runs them in its worker threadpool.
Analysis is CPU-bound and takes ~100 ms per document; doing it in an ``async``
handler would block the event loop and stall every other connection, including
the health probes. Concurrency is then bounded by the pipeline pool rather than
by the threadpool -- see :mod:`umlsmatch.service.pool`.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
import traceback
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from umlsmatch import __version__
from umlsmatch.analyze import DEFAULT_PROFILE, Annotation, find_dictionary
from umlsmatch.service.pool import PipelinePool, PoolExhausted

__all__ = ["app", "create_app"]

log = logging.getLogger("umlsmatch.service")


# --- configuration -----------------------------------------------------------
# Environment rather than a config file: this ships as a container, and every
# knob below is something an operator sets per deployment.


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


#: Accepted spellings of a boolean environment variable. Rejecting anything
#: else matters more here than usual: this is the switch on whether exception
#: text reaches the logs, and a typo silently reverting to the default is the
#: wrong failure for a privacy control.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if not raw:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE | _FALSE)}, got {raw!r}"
    )


#: Pipelines to build. Concurrency cap and memory multiplier both.
POOL_SIZE = _env_int("UMLSMATCH_POOL_SIZE", 2)
#: Named configuration for every pipeline in the pool; see
#: :data:`umlsmatch.analyze.PROFILES`. ``strict`` (the default) is what the
#: published figures are measured on; ``clinical_recall`` trades a literal
#: reading for recall against adjudicated verdicts. An unknown name fails at
#: pool construction, which is startup -- the readiness probe never passes, so
#: a typo here cannot reach traffic.
PROFILE = os.environ.get("UMLSMATCH_PROFILE") or None
#: Seconds a request waits for a free pipeline before 503.
LEASE_TIMEOUT = _env_float("UMLSMATCH_LEASE_TIMEOUT", 30.0)
#: Documents per /analyze/batch call. Bounded so one request cannot pin a
#: pipeline for minutes and starve everything else.
MAX_BATCH = _env_int("UMLSMATCH_MAX_BATCH", 64)
#: Characters per document. A runaway upload should be rejected before it is
#: tokenized, not after.
MAX_CHARS = _env_int("UMLSMATCH_MAX_CHARS", 1_000_000)
#: Whether a failing batch document's exception *message* reaches the log.
#:
#: The HTTP response has only ever carried the exception type name, because a
#: message from spaCy or sqlite3 is entitled to quote the input that broke it.
#: The log has always received the full thing, which is a defensible trade for
#: an isolated deployment -- the logs are one place to audit -- and the wrong
#: one the moment they ship to an aggregator with a wider audience than the API.
#:
#: The default preserves the existing behaviour rather than silently tightening
#: a deployment on upgrade. Turning it off keeps the exception type and the call
#: path (file, line, function) and drops every message, including those of
#: chained causes; see :func:`_redacted_traceback`. **With it on, treat
#: application logs as PHI.**
LOG_EXCEPTION_DETAIL = _env_bool("UMLSMATCH_LOG_EXCEPTION_DETAIL", True)
#: Shared secret required on the PHI endpoints, or ``None`` for no check.
#:
#: **Not a replacement for the reverse proxy**, which still terminates TLS,
#: rate-limits, and is where authentication belongs; docs/SERVICE.md is
#: unchanged on that. This is a second line for the failure that proxy guidance
#: cannot cover -- a misconfigured ingress, or a container that bound
#: ``0.0.0.0`` on a flatter network than intended -- on a service where the
#: thing exposed is clinical narrative. One check that costs nothing beats
#: relying on a single configuration being right.
#:
#: Default off, because on would break every existing deployment on upgrade and
#: this is a backstop rather than the control. Over plain HTTP the key travels
#: in cleartext and is only as good as the network; that is the proxy's job.
API_KEY = os.environ.get("UMLSMATCH_API_KEY") or None
#: Header the key is read from. Fixed rather than configurable: a per-deployment
#: header name is one more thing to get wrong for no security gain.
API_KEY_HEADER = "x-api-key"
#: Bytes per request body, checked before the body is read.
#:
#: ``MAX_CHARS`` and ``MAX_BATCH`` bound the *analysis*, but they are enforced
#: inside the handler -- by which point Starlette has read the whole body and
#: Pydantic has materialized it. Their product is ~64 MB of JSON parsed in a
#: worker thread before a 413 comes back. This bounds the parsing too. The
#: default is generous next to real traffic (the reference corpus averages
#: 5,558 characters a note, so a full 64-document batch is ~350 KB) and still
#: two orders of magnitude below that product.
MAX_BODY_BYTES = _env_int("UMLSMATCH_MAX_BODY_BYTES", 16 * 1024 * 1024)


# --- metrics -----------------------------------------------------------------
# A private registry rather than prometheus_client's global default: the default
# is shared by every library in the process, so scraping it mixes in whatever
# else happens to be installed and makes a test assertion about our own counters
# depend on unrelated code. Nothing here is labelled by anything derived from
# note content -- a label taken from, say, a document id would put PHI into the
# metrics endpoint, which is typically scraped by systems with wider access than
# the API itself.

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "umlsmatch_requests_total", "Requests handled.", ["endpoint", "status"], registry=REGISTRY
)
LATENCY = Histogram(
    "umlsmatch_request_duration_seconds",
    "Wall time per request.",
    ["endpoint"],
    registry=REGISTRY,
)
DOCUMENTS = Counter(
    "umlsmatch_documents_total", "Documents analyzed.", registry=REGISTRY
)
ANNOTATIONS = Counter(
    "umlsmatch_annotations_total", "Annotations returned.", registry=REGISTRY
)
CHARACTERS = Counter(
    "umlsmatch_characters_total", "Characters analyzed.", registry=REGISTRY
)
POOL_AVAILABLE = Gauge(
    "umlsmatch_pool_available", "Pipelines not currently leased.", registry=REGISTRY
)
POOL_SIZE_GAUGE = Gauge(
    "umlsmatch_pool_size", "Pipelines in the pool.", registry=REGISTRY
)
LEASE_WAIT = Histogram(
    "umlsmatch_lease_wait_seconds",
    "Time spent waiting for a free pipeline. Sustained non-zero means the pool "
    "is undersized for the offered load.",
    registry=REGISTRY,
)
MEMO_ENTRIES = Gauge(
    "umlsmatch_matcher_memo_entries",
    "Entries held in the matcher's bounded memo caches, summed over the pool. "
    "Plateaus at pool_size * memo_size; a flat line well below that means the "
    "bound is not binding and memo_size could be lowered.",
    ["cache"],
    registry=REGISTRY,
)


# --- schemas -----------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """One document to analyze. The body is PHI; see the service description."""

    text: str = Field(..., description="Clinical note text.")
    groups: list[str] | None = Field(
        None,
        description="Keep only these semantic groups, e.g. ['DISORDER','DRUG']. "
        "Null means no filtering.",
    )
    negated_only: bool = False
    affirmed_only: bool = False


class BatchRequest(BaseModel):
    """Several documents analyzed under one pipeline lease."""

    documents: list[str] = Field(..., description="Note texts.")
    ids: list[str] | None = Field(
        None,
        description="Optional caller-supplied identifiers, echoed back so "
        "results can be reattached without relying on ordering.",
    )
    groups: list[str] | None = None
    negated_only: bool = False
    affirmed_only: bool = False


class AnnotationOut(BaseModel):
    """One concept found in a document. Mirrors :class:`umlsmatch.Annotation`.

    ``start``/``end`` are character offsets into the document that was sent, so
    a caller can re-slice the original text rather than trust ``text``.

    **A null assertion attribute means "not assessed", not "absent".** The five
    non-polarity attributes are nullable for the reason given on
    :class:`umlsmatch.Annotation`: this pipeline implements some of them, does
    not implement ``generic``, and leaves ``conditional`` off unless the service
    was configured with it, and a ``false`` a client cannot distinguish from an
    assessment is the expensive kind of wrong in a clinical record. Clients must
    branch on null rather than coerce it -- and must not infer from one response
    which attributes this deployment assesses, since a null is the same shape
    either way. ``negated`` is always assessed and stays non-nullable.
    """

    cui: str
    text: str
    start: int
    end: int
    group: str
    negated: bool
    preferred_text: str = ""
    term: str = ""
    subject: str | None = None
    history_of: bool | None = None
    uncertain: bool | None = None
    conditional: bool | None = None
    generic: bool | None = None


class AnalyzeResponse(BaseModel):
    """Annotations for one document, in document order."""

    annotations: list[AnnotationOut]
    n_annotations: int
    took_ms: float


class DocumentResult(BaseModel):
    """One document's result inside a batch.

    ``error`` holds the exception *type name* when that document failed and the
    rest of the batch continued; it is never the exception message, which may
    quote the input. It is ``None`` on success.
    """

    id: str | None = None
    annotations: list[AnnotationOut]
    n_annotations: int
    error: str | None = None


class BatchResponse(BaseModel):
    """Batch results, in request order. ``ids`` are echoed back per document."""

    results: list[DocumentResult]
    n_documents: int
    n_annotations: int
    took_ms: float


# --- helpers -----------------------------------------------------------------


def _redacted_traceback(exc: BaseException) -> str:
    """`exc`'s type and call path, with every exception message dropped.

    The call path is what makes a failure diagnosable -- which line in which
    module raised, and through what -- and none of it derives from the request.
    The message is the part that can quote the input, so it is the part that
    goes. Chained causes are walked for the same reason: a ``sqlite3`` error
    re-raised as something of ours still carries the original text on
    ``__cause__``, and reporting only the outermost type would leak it via the
    chain that ``log.exception`` would have printed anyway.

    File names are reduced to their basename: the full path is host layout,
    the same reasoning ``/info`` applies to the dictionary.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    # Bounded because an exception chain can be cyclic (`raise X from Y` where
    # Y's context is X), and this runs inside an error path that must not hang.
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = " -> ".join(
            f"{PurePath(f.filename).name}:{f.lineno} in {f.name}"
            for f in traceback.extract_tb(current.__traceback__)
        )
        parts.append(f"{type(current).__qualname__} [{frames or 'no traceback'}]")
        current = current.__cause__ or current.__context__
    return " <- caused by ".join(parts)


def _log_document_failure(index: int, exc: BaseException) -> None:
    """Report a failed batch document at the configured level of detail.

    See :data:`LOG_EXCEPTION_DETAIL`. The document *index* is safe to log and
    is the whole point of logging at all -- it is what lets an operator find
    the input that failed in their own system, without this service having
    repeated it.
    """
    if LOG_EXCEPTION_DETAIL:
        log.exception("batch document %d failed", index)
    else:
        log.error("batch document %d failed: %s", index, _redacted_traceback(exc))


def _filter(
    annotations: Sequence[Annotation],
    groups: Sequence[str] | None,
    negated_only: bool,
    affirmed_only: bool,
) -> list[Annotation]:
    """Apply the request's group and polarity filters.

    `groups` is normalized here rather than in the schema so a caller may send
    ``["disorder", " DRUG "]`` and mean what the pipeline means. ``None`` and an
    all-blank list both mean "no filtering"; the two ``_only`` flags are
    rejected as a pair upstream, so at most one is ever set.
    """
    wanted = {g.strip().upper() for g in groups if g.strip()} if groups else None
    out: list[Annotation] = []
    for a in annotations:
        if wanted is not None and a.group not in wanted:
            continue
        if negated_only and not a.negated:
            continue
        if affirmed_only and a.negated:
            continue
        out.append(a)
    return out


class BodySizeLimit:
    """Bound the request body, by ``Content-Length`` and then as it streams.

    Raw ASGI rather than ``BaseHTTPMiddleware`` because this has to run *before*
    anything consumes the request stream, which is the entire point -- a check
    that runs after the body is buffered bounds nothing.

    Two checks, because there are two ways to send a body:

    1. **Declared.** A ``Content-Length`` over the cap is refused before a byte
       is read. Cheapest possible rejection and the common case.
    2. **Streamed.** A ``Transfer-Encoding: chunked`` request carries no
       ``Content-Length``, so nothing is declared to check. The wrapped
       ``receive`` below counts what actually arrives and refuses on the chunk
       that crosses the cap -- while the handler is still reading, and therefore
       before Starlette has buffered the rest or pydantic has materialized
       anything.

    (2) could be left entirely to the reverse proxy, on the argument that a
    proxy sees the request first and can reject it without occupying a worker.
    That is true and the proxy is still the right primary control, but leaving
    it there makes the only bound on a direct client an operator's proxy
    configuration -- and the failure is
    silent memory growth, on a service whose every request body is PHI. Counting
    costs one integer add per chunk. See docs/SERVICE.md.

    Both checks are applied to every request. A declared length is not taken on
    trust: if a caller under-declares, the counter still stops the read.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _counted(self, receive: Receive, path: str) -> Receive:
        """`receive`, with a running total that raises once it passes the cap.

        ``HTTPException`` specifically, and not an exception of our own: the
        raise lands inside FastAPI's body read, which converts *any* other
        exception into a generic ``400 There was an error parsing the body`` --
        turning a deliberate refusal into what looks like a client's malformed
        JSON. FastAPI re-raises ``HTTPException`` untouched for exactly this
        case ("if a middleware raises an HTTPException, it should be raised
        again"), so the status survives and ``_http_error`` counts it with every
        other rejection. ``tests/test_service.py`` pins the status, so a change
        in that behaviour fails loudly rather than degrading to 400.
        """
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    log.warning(
                        "aborted body on %s after %d bytes (limit %d)",
                        path, seen, self.max_bytes,
                    )
                    # Length only -- never the body.
                    raise HTTPException(
                        status_code=413,
                        detail=f"request body exceeded {self.max_bytes:,} bytes; "
                               f"read was aborted at {seen:,}",
                    )
            return message

        return counted

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        for key, value in scope.get("headers", ()):
            if key != b"content-length":
                continue
            try:
                declared = int(value)
            except ValueError:
                break
            if declared > self.max_bytes:
                # Length only -- never the body.
                REQUESTS.labels(endpoint_label(path), "413").inc()
                log.warning(
                    "rejected %d-byte body on %s (limit %d)",
                    declared, path, self.max_bytes,
                )
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"request body is {declared:,} bytes; "
                                  f"limit is {self.max_bytes:,}"
                    },
                )
                await response(scope, receive, send)
                return
            break

        await self.app(scope, self._counted(receive, path), send)


#: Label used for any request path that is not one of this app's routes.
UNMATCHED_ENDPOINT = "<other>"

#: Paths allowed to appear as an ``endpoint`` label on :data:`REQUESTS`.
#:
#: The label must come from a fixed set. ``request.url.path`` is caller
#: controlled, so counting 404s under it would mint a Prometheus time series per
#: URL a scanner tries -- unbounded cardinality in the registry, and the metric
#: least able to survive it is the error counter, which is exactly where the
#: unmatched paths land. ``test_the_known_endpoint_set_matches_the_app`` keeps
#: this in step with the routes.
KNOWN_ENDPOINTS = frozenset(
    {"/analyze", "/analyze/batch", "/health", "/ready", "/metrics", "/info"}
)


def endpoint_label(path: str) -> str:
    """Bound a request path to the fixed set of labels. See :data:`KNOWN_ENDPOINTS`."""
    return path if path in KNOWN_ENDPOINTS else UNMATCHED_ENDPOINT


def require_api_key(request: Request) -> None:
    """Refuse the request unless it carries :data:`API_KEY`, if one is set.

    A no-op when ``API_KEY`` is ``None``, which is the default and the existing
    behaviour. Read from the module at call time rather than captured when the
    app is built, so a test can set it without rebuilding the app.

    Applied to the analysis endpoints only. ``/health`` and ``/ready`` are
    orchestrator probes and ``/metrics`` is scraped by a collector that
    generally sends no custom headers -- putting a key on any of them turns a
    misconfiguration into restart loops or a silently dead dashboard, and none
    of the three carries PHI. They stay the proxy's to restrict, as
    docs/SERVICE.md already says.

    ``hmac.compare_digest`` rather than ``==``: the comparison is on a secret,
    and a short-circuiting compare leaks its prefix through timing. Both sides
    are encoded first because ``compare_digest`` rejects non-ASCII ``str``, and
    a key with a non-ASCII character should be wrong, not a 500.
    """
    if API_KEY is None:
        return
    presented = request.headers.get(API_KEY_HEADER, "")
    if not hmac.compare_digest(presented.encode(), API_KEY.encode()):
        # No echo of what was presented -- it is a credential, and this path is
        # reached by scanners.
        log.warning("rejected %s: missing or invalid API key", request.url.path)
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _validate(text: str, *, where: str) -> None:
    if len(text) > MAX_CHARS:
        # Length only -- never the text.
        raise HTTPException(
            status_code=413,
            detail=f"{where} is {len(text):,} characters; limit is {MAX_CHARS:,}",
        )


# --- app ---------------------------------------------------------------------


def create_app(pool: PipelinePool | None = None) -> FastAPI:
    """Build the application.

    `pool` is injectable so tests can supply a size-1 pool, or a fake, without
    building several real pipelines (seconds each) per test.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pool = pool or PipelinePool(
            size=POOL_SIZE,
            lease_timeout=LEASE_TIMEOUT,
            profile=PROFILE,
            # The handler's own MAX_CHARS check runs first and answers 413,
            # which is the right status and the right place. Passing the same
            # number down keeps the two from drifting: raising
            # UMLSMATCH_MAX_CHARS past the pipeline's default would otherwise
            # have the service accept a document the pipeline then refuses,
            # turning a deliberate 413 into an accidental 500.
            max_chars=MAX_CHARS,
        )
        POOL_SIZE_GAUGE.set(app.state.pool.size)
        # Build eagerly so /ready is honest; see PipelinePool.start.
        app.state.pool.start()
        POOL_AVAILABLE.set(app.state.pool.available)
        try:
            yield
        finally:
            app.state.pool.close()

    app = FastAPI(
        title="umlsmatch",
        version=__version__,
        description=(
            "Clinical concept extraction with negation. Request and response "
            "bodies contain PHI; run egress-blocked and treat access logs "
            "accordingly."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    @app.exception_handler(PoolExhausted)
    async def _pool_exhausted(request: Request, exc: PoolExhausted) -> JSONResponse:
        # 503 + Retry-After rather than queueing without bound: a caller can
        # back off, where a growing queue just converts overload into latency.
        REQUESTS.labels(endpoint_label(request.url.path), "503").inc()
        log.warning("pool exhausted serving %s", request.url.path)
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """Count every rejection, then answer exactly as the default would.

        ``REQUESTS`` is counted here rather than at each ``raise``, so the next
        ``raise`` is counted without anyone remembering to. Incrementing it only
        on the success path leaves the handler rejections (413s, every 422) out
        of the one metric whose audience is an alert on error rate -- it reads
        clean while clients are being turned away.

        This handler also sees 404s and 405s, whose paths are whatever the
        caller asked for -- hence :func:`endpoint_label`.
        """
        REQUESTS.labels(endpoint_label(request.url.path), str(exc.status_code)).inc()
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """422 without echoing the body back.

        FastAPI's default handler puts the offending value in an ``input`` key.
        For ``{"text": 12345}`` that is harmless; for ``{"text": ["<a note>"]}``
        it is note content in a response body, and from there in whatever logs
        the response -- which is the one thing every other error path here is
        careful not to do (``# Length only -- never the body``, twice). The
        service's own description tells operators to treat bodies as PHI.

        ``type``, ``loc`` and ``msg`` are what makes the error actionable and
        none of them carry caller data, so the client still learns which field
        was wrong and why. ``url`` goes too: pydantic's docs link is the only
        outbound URL this egress-blocked service would ever hand out.
        """
        REQUESTS.labels(endpoint_label(request.url.path), "422").inc()
        redacted = [
            {k: v for k, v in error.items() if k in ("type", "loc", "msg")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": redacted})

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        """Liveness: the process is up. Cheap, and never touches the pool."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready(response: Response) -> dict[str, Any]:
        """Readiness: the pool is built and can serve a request."""
        pool: PipelinePool = app.state.pool
        if not pool.ready():
            response.status_code = 503
            return {"status": "starting", "ready": False}
        return {
            "status": "ok",
            "ready": True,
            "pool_size": pool.size,
            "available": pool.available,
        }

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        """Prometheus exposition for this process's private registry."""
        pool: PipelinePool | None = getattr(app.state, "pool", None)
        if pool is not None:
            POOL_AVAILABLE.set(pool.available)
            # Sampled at scrape time rather than per request: reading it is a
            # len() per pipeline, and nothing here should add work to the hot
            # path to populate a gauge.
            candidates, concepts = pool.memo_entries()
            MEMO_ENTRIES.labels("candidate").set(candidates)
            MEMO_ENTRIES.labels("concept").set(concepts)
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        tags=["analysis"],
        dependencies=[Depends(require_api_key)],
    )
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        """Analyze one document."""
        if req.negated_only and req.affirmed_only:
            raise HTTPException(
                status_code=422,
                detail="negated_only and affirmed_only are mutually exclusive",
            )
        _validate(req.text, where="text")

        started = time.monotonic()
        with LATENCY.labels("/analyze").time():
            waited = time.monotonic()
            with app.state.pool.lease() as pipeline:
                LEASE_WAIT.observe(time.monotonic() - waited)
                annotations = pipeline.analyze(req.text)

        annotations = _filter(
            annotations, req.groups, req.negated_only, req.affirmed_only
        )
        took = (time.monotonic() - started) * 1000

        DOCUMENTS.inc()
        CHARACTERS.inc(len(req.text))
        ANNOTATIONS.inc(len(annotations))
        REQUESTS.labels("/analyze", "200").inc()
        # Sizes and timing only -- no text.
        log.info("analyze: %d chars -> %d annotations in %.0fms",
                 len(req.text), len(annotations), took)

        return AnalyzeResponse(
            annotations=[AnnotationOut(**a.to_dict()) for a in annotations],
            n_annotations=len(annotations),
            took_ms=round(took, 1),
        )

    @app.post(
        "/analyze/batch",
        response_model=BatchResponse,
        tags=["analysis"],
        dependencies=[Depends(require_api_key)],
    )
    def analyze_batch(req: BatchRequest) -> BatchResponse:
        """Analyze many documents, holding one pipeline for the whole call.

        One lease for the batch, not one per document: re-leasing per document
        would let two batches interleave and thrash the matcher's per-pipeline
        caches against each other. `MAX_BATCH` is what stops that from pinning
        a pipeline indefinitely.

        A document that fails is reported in its own `error` field and the rest
        still run -- one malformed note should not void a batch, the same
        reasoning as ``examples/parse_to_jsonl.py``.
        """
        if req.negated_only and req.affirmed_only:
            raise HTTPException(
                status_code=422,
                detail="negated_only and affirmed_only are mutually exclusive",
            )
        if not req.documents:
            raise HTTPException(status_code=422, detail="documents must not be empty")
        if len(req.documents) > MAX_BATCH:
            raise HTTPException(
                status_code=413,
                detail=f"{len(req.documents)} documents; limit is {MAX_BATCH}",
            )
        if req.ids is not None and len(req.ids) != len(req.documents):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ids has {len(req.ids)} entries but documents has "
                    f"{len(req.documents)}"
                ),
            )
        for i, text in enumerate(req.documents):
            _validate(text, where=f"document {i}")

        started = time.monotonic()
        results: list[DocumentResult] = []
        total_annotations = 0

        with LATENCY.labels("/analyze/batch").time():
            waited = time.monotonic()
            with app.state.pool.lease() as pipeline:
                LEASE_WAIT.observe(time.monotonic() - waited)
                for i, text in enumerate(req.documents):
                    doc_id = req.ids[i] if req.ids else None
                    try:
                        annotations = _filter(
                            pipeline.analyze(text),
                            req.groups,
                            req.negated_only,
                            req.affirmed_only,
                        )
                    # Broad by intent -- see the docstring: one malformed
                    # document must not void the whole batch.
                    except Exception as exc:
                        # Exception *type* only, never the message. That it
                        # comes from our own code paths holds for the raises in
                        # this module, but not for anything thrown inside spaCy
                        # or sqlite3, which are entitled to quote the input that
                        # broke them. The response crosses the network, so it
                        # never gets the message regardless; how much reaches
                        # the *log* is the operator's call -- see
                        # LOG_EXCEPTION_DETAIL.
                        _log_document_failure(i, exc)
                        results.append(
                            DocumentResult(
                                id=doc_id,
                                annotations=[],
                                n_annotations=0,
                                error=type(exc).__name__,
                            )
                        )
                        continue
                    total_annotations += len(annotations)
                    CHARACTERS.inc(len(text))
                    results.append(
                        DocumentResult(
                            id=doc_id,
                            annotations=[AnnotationOut(**a.to_dict()) for a in annotations],
                            n_annotations=len(annotations),
                        )
                    )

        took = (time.monotonic() - started) * 1000
        DOCUMENTS.inc(len(req.documents))
        ANNOTATIONS.inc(total_annotations)
        REQUESTS.labels("/analyze/batch", "200").inc()
        log.info(
            "batch: %d documents -> %d annotations in %.0fms",
            len(req.documents), total_annotations, took,
        )

        return BatchResponse(
            results=results,
            n_documents=len(req.documents),
            n_annotations=total_annotations,
            took_ms=round(took, 1),
        )

    @app.get("/info", tags=["ops"])
    def info() -> dict[str, Any]:
        """Configuration and dictionary provenance, for debugging a deployment.

        The dictionary is reported by **file name, not full path**. Which
        artifact is loaded is the operational question ("is this the 7-source
        build or the all-sources one?"); the directory it sits in is host
        layout, and this endpoint has no authentication in front of it.

        ``log_exception_detail`` and ``api_key_required`` are reported for the
        opposite reason to the others: they are not capacity knobs but privacy
        ones, and a setting that governs whether PHI can reach the logs -- or
        whether the analysis endpoints are open -- is worth being able to
        confirm from outside the container rather than inferring from a
        deployment manifest. Both disclose a policy, not any data:
        ``api_key_required`` is a boolean, never the key.
        """
        pool: PipelinePool = app.state.pool
        return {
            "version": __version__,
            "dictionary": find_dictionary(pool.db_path).name,
            # Which trade-off this deployment is running. Two instances can
            # return different annotations for the same note and both be
            # correct; this is how a caller tells which one answered.
            "profile": PROFILE or DEFAULT_PROFILE,
            "pool_size": pool.size,
            "lease_timeout_s": pool.lease_timeout,
            "max_batch": MAX_BATCH,
            "max_chars": MAX_CHARS,
            "max_body_bytes": MAX_BODY_BYTES,
            "log_exception_detail": LOG_EXCEPTION_DETAIL,
            "api_key_required": API_KEY is not None,
        }

    return app


app = create_app()

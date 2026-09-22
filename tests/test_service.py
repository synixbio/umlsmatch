"""Tests for the HTTP service.

Uses a fake pipeline injected into a real :class:`PipelinePool`, so these run
without a 589 MB dictionary or a spaCy model -- the same reason the rest of the
suite stays runnable on a fresh clone. What is under test here is the service
layer (validation, filtering, error mapping, pool behaviour, metrics), not the
analysis, which has its own tests.

The concurrency test is the one that matters most: ``ClinicalPipeline`` is not
thread-safe, so the pool's job is to guarantee no two threads ever hold the same
instance. That is an invariant a fake can check precisely and a real pipeline
cannot.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from umlsmatch.service.pool import PipelinePool, PoolExhausted

fastapi = pytest.importorskip("fastapi", reason="needs the 'service' extra")
from fastapi.testclient import TestClient  # noqa: E402

from umlsmatch.service.api import create_app  # noqa: E402


class FakeAnnotation:
    """Stands in for umlsmatch.analyze.Annotation."""

    def __init__(self, cui, text, start, end, group, negated):
        self._d = {
            "cui": cui, "text": text, "start": start, "end": end,
            "group": group, "negated": negated,
            "preferred_text": f"pref-{cui}", "term": text.lower(),
        }
        self.group = group
        self.negated = negated

    def to_dict(self):
        return dict(self._d)


class FakePipeline:
    """Deterministic stand-in that also records concurrent use.

    ``in_use`` is incremented on entry and decremented on exit; the pool is
    correct only if it never exceeds 1 for any single instance.
    """

    def __init__(self, delay: float = 0.0):
        self.db_path = "fake.sqlite"
        self.delay = delay
        self.calls = 0
        self.in_use = 0
        self.max_concurrent = 0
        self.closed = False

    def analyze(self, text: str):
        self.in_use += 1
        self.max_concurrent = max(self.max_concurrent, self.in_use)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls += 1
            if "BOOM" in text:
                raise ValueError("synthetic failure")
            out = []
            if "pain" in text.lower():
                i = text.lower().index("pain")
                out.append(FakeAnnotation("C0030193", text[i:i + 4], i, i + 4, "FINDING", True))
            if "aspirin" in text.lower():
                i = text.lower().index("aspirin")
                out.append(FakeAnnotation("C0004057", text[i:i + 7], i, i + 7, "DRUG", False))
            return out
        finally:
            self.in_use -= 1

    def close(self):
        self.closed = True


def make_pool(size=2, delay=0.0, **kw) -> PipelinePool:
    """A real pool holding fake pipelines -- pool logic is what we want tested."""
    pool = PipelinePool(size=size, **kw)
    pool._all = [FakePipeline(delay) for _ in range(size)]
    for p in pool._all:
        pool._free.put(p)
    pool._started = True
    return pool


@pytest.fixture
def client():
    pool = make_pool(size=2)
    app = create_app(pool)
    with TestClient(app) as c:
        c.pool = pool
        yield c


# --- probes ------------------------------------------------------------------


def test_health_is_up(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_pool_state(client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["pool_size"] == 2


def test_ready_is_503_before_the_pool_is_built():
    """Readiness must be false while warming, or traffic arrives too early."""
    pool = PipelinePool(size=1)  # never started
    app = create_app(pool)
    # No `with`: TestClient runs the app's lifespan on __enter__ only, and that
    # lifespan calls pool.start(), which builds real pipelines and needs the
    # 589 MB dictionary. Entering it here contradicted the thing under test --
    # the pool has to stay unbuilt for /ready to have anything to report -- and
    # tore the test in two directions: on a machine with a dictionary it built
    # a real pool and then faked it back down afterwards, and on one without
    # (CI, a fresh clone) it raised FileNotFoundError out of startup. Driving
    # the client without the context manager skips the lifespan entirely, which
    # is what "never started" meant all along.
    # Skipping the lifespan means skipping the line that publishes the pool, so
    # `ready()` would read a missing app.state.pool and 500 instead of 503.
    app.state.pool = pool
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_info_reports_configuration(client):
    body = client.get("/info").json()
    assert body["pool_size"] == 2
    assert "max_batch" in body
    assert "max_body_bytes" in body


def test_info_does_not_disclose_the_dictionary_path(client):
    """/info is unauthenticated; which artifact is loaded is operational, where
    it lives on disk is host layout."""
    dictionary = client.get("/info").json()["dictionary"]
    assert "/" not in dictionary and "\\" not in dictionary, dictionary


# --- optional API key ---------------------------------------------------------


@pytest.fixture
def keyed_client(monkeypatch):
    """A client whose analysis endpoints require ``s3cret``.

    ``require_api_key`` reads the module attribute per request, so patching it
    needs no rebuilt app -- but the fixture builds one anyway so the unkeyed
    ``client`` fixture stays untouched for every other test.
    """
    from umlsmatch.service import api

    monkeypatch.setattr(api, "API_KEY", "s3cret")
    pool = make_pool(size=2)
    with TestClient(api.create_app(pool)) as c:
        c.pool = pool
        yield c


def test_no_key_configured_leaves_the_service_open(client):
    """The default is off, and has to stay off: on by default would break every
    existing deployment on upgrade."""
    from umlsmatch.service import api

    assert api.API_KEY is None
    assert client.post("/analyze", json={"text": "pain"}).status_code == 200
    assert client.get("/info").json()["api_key_required"] is False


def test_analysis_requires_the_key_when_one_is_set(keyed_client):
    for path, payload in (
        ("/analyze", {"text": "pain"}),
        ("/analyze/batch", {"documents": [{"text": "pain"}]}),
    ):
        r = keyed_client.post(path, json=payload)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "ApiKey"
    # Refused at the door: nothing was analyzed to find out.
    assert all(p.calls == 0 for p in keyed_client.pool._all)


def test_the_right_key_is_accepted(keyed_client):
    r = keyed_client.post(
        "/analyze", json={"text": "Patient has pain."}, headers={"x-api-key": "s3cret"}
    )
    assert r.status_code == 200


def test_a_wrong_key_is_refused(keyed_client):
    r = keyed_client.post(
        "/analyze", json={"text": "pain"}, headers={"x-api-key": "s3cre"}
    )
    assert r.status_code == 401


def test_probes_and_metrics_stay_open(keyed_client):
    """Deliberate. A key on /health or /ready turns a misconfigured secret into
    a restart loop, and one on /metrics into a silently dead dashboard, because
    neither an orchestrator probe nor a Prometheus scrape sends custom headers
    by default. None of the three carries PHI; restricting them is the proxy's
    job, which is what docs/SERVICE.md already says."""
    for path in ("/health", "/ready", "/metrics", "/info"):
        assert keyed_client.get(path).status_code == 200, path


def test_info_reports_that_a_key_is_required_but_never_the_key(keyed_client):
    body = keyed_client.get("/info").json()
    assert body["api_key_required"] is True
    assert "s3cret" not in keyed_client.get("/info").text


def test_a_non_ascii_configured_key_refuses_rather_than_500(monkeypatch):
    """``hmac.compare_digest`` raises ``TypeError`` on non-ASCII ``str``.

    Both sides are encoded to bytes first, so an operator who sets a key with an
    accent in it gets a clean 401 instead of an unhandled exception on every
    request. Such a key can never actually be *presented* -- HTTP header values
    are ASCII -- so the endpoint is unreachable either way; the point is that it
    fails as a refusal and not as a 500 that looks like the service is broken.
    """
    from umlsmatch.service import api

    monkeypatch.setattr(api, "API_KEY", "s3crét")
    pool = make_pool(size=2)
    with TestClient(api.create_app(pool)) as c:
        r = c.post("/analyze", json={"text": "pain"}, headers={"x-api-key": "s3cret"})
    assert r.status_code == 401


def test_rejected_key_is_counted_as_401(keyed_client):
    keyed_client.post("/analyze", json={"text": "pain"})
    metrics = keyed_client.get("/metrics").text
    assert 'umlsmatch_requests_total{endpoint="/analyze",status="401"}' in metrics


# --- body size limit ---------------------------------------------------------


def test_oversized_body_is_rejected_before_parsing(client):
    """MAX_CHARS/MAX_BATCH are enforced *inside* the handler.

    By then Starlette has read the body and Pydantic has built it, so their
    product is ~64 MB of JSON parsed in a worker thread before the 413. This
    ceiling has to bite earlier, off the Content-Length.
    """
    from umlsmatch.service.api import MAX_BODY_BYTES

    r = client.post(
        "/analyze",
        content=b'{"text":"x"}',
        headers={
            "content-type": "application/json",
            "content-length": str(MAX_BODY_BYTES + 1),
        },
    )
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]
    # Rejected at the door: no pipeline was leased to find that out.
    assert all(p.calls == 0 for p in client.pool._all)


def test_ordinary_body_passes_the_limit(client):
    r = client.post("/analyze", json={"text": "Patient has pain."})
    assert r.status_code == 200


def _chunks(total: int, chunk: int = 8 * 1024):
    """A body httpx will send chunked, because it is an iterator and not bytes.

    Chunked framing carries no ``Content-Length``, which is the whole point:
    the declared-length check above has nothing to look at.
    """
    sent = 0
    while sent < total:
        n = min(chunk, total - sent)
        yield b"x" * n
        sent += n


@pytest.fixture
def small_limit_client(monkeypatch):
    """A client whose body cap is 64 KiB, so the streaming tests stay cheap.

    Patched before ``create_app``, which reads ``MAX_BODY_BYTES`` once when it
    installs the middleware.
    """
    from umlsmatch.service import api

    monkeypatch.setattr(api, "MAX_BODY_BYTES", 64 * 1024)
    pool = make_pool(size=2)
    with TestClient(api.create_app(pool)) as c:
        c.pool = pool
        yield c


def test_streamed_body_is_aborted_at_the_cap(small_limit_client):
    """A chunked upload declares no length, so only counting can bound it.

    Without this the body streams past the cap into Starlette's buffer and
    pydantic's parser, and the only thing standing between a direct client and
    the worker's memory is the operator's reverse-proxy configuration.
    """
    r = small_limit_client.post(
        "/analyze",
        content=_chunks(1024 * 1024),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert "exceeded" in r.json()["detail"]
    # Aborted mid-read: the request never reached a pipeline.
    assert all(p.calls == 0 for p in small_limit_client.pool._all)


def test_streamed_body_under_the_cap_is_analyzed(small_limit_client):
    """The counter must not break the ordinary streamed request."""
    import json

    payload = json.dumps({"text": "Patient has pain."}).encode()
    r = small_limit_client.post(
        "/analyze",
        content=iter([payload]),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    assert r.json()["annotations"][0]["cui"] == "C0030193"


def test_streamed_rejection_is_counted_once(small_limit_client):
    """One 413 per rejected request, not two.

    The streamed refusal is an ``HTTPException``, so ``_http_error`` counts it
    like every other handler rejection. Counting again at the raise -- which is
    where the byte total is known and therefore the tempting place -- would
    silently double the error rate an alert is reading.
    """
    def _count() -> float:
        body = small_limit_client.get("/metrics").text
        for line in body.splitlines():
            if line.startswith('umlsmatch_requests_total{endpoint="/analyze",status="413"}'):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    before = _count()
    small_limit_client.post(
        "/analyze",
        content=_chunks(1024 * 1024),
        headers={"content-type": "application/json"},
    )
    assert _count() == before + 1


def test_under_declared_content_length_is_still_bounded(small_limit_client):
    """A declared length is not taken on trust.

    The declared-length check passes (it says 12 bytes), and the counter is what
    stops the read. This is why the wrapper is applied to every request rather
    than only to the ones with no ``Content-Length``.
    """
    r = small_limit_client.post(
        "/analyze",
        content=_chunks(1024 * 1024),
        headers={"content-type": "application/json", "content-length": "12"},
    )
    assert r.status_code == 413


# --- analyze -----------------------------------------------------------------


def test_analyze_returns_annotations(client):
    r = client.post("/analyze", json={"text": "Patient has pain and aspirin."})
    assert r.status_code == 200
    body = r.json()
    assert body["n_annotations"] == 2
    assert {a["cui"] for a in body["annotations"]} == {"C0030193", "C0004057"}
    assert body["took_ms"] >= 0


def test_analyze_offsets_point_into_the_submitted_text(client):
    text = "Patient has pain today."
    a = client.post("/analyze", json={"text": text}).json()["annotations"][0]
    assert text[a["start"]:a["end"]] == a["text"]


def test_unassessed_attributes_serialize_as_null_not_false(client):
    """A client must be able to tell "absent" from "not looked at".

    The fake pipeline omits the five non-polarity attributes entirely, which is
    exactly what a pipeline that does not assess them does. They must arrive as
    JSON null -- a `false` here is indistinguishable downstream from a genuine
    negative assessment. See umlsmatch.analyze.Annotation.
    """
    a = client.post("/analyze", json={"text": "Patient has pain."}).json()["annotations"][0]
    for name in ("subject", "history_of", "uncertain", "conditional", "generic"):
        assert name in a, f"{name} must be present in the response schema"
        assert a[name] is None, f"{name} arrived as {a[name]!r}, not null"
    assert isinstance(a["negated"], bool), "polarity is always assessed"


def test_openapi_marks_the_five_attributes_nullable(client):
    """The contract is published, not just honoured by accident."""
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["AnnotationOut"]["properties"]
    for name in ("subject", "history_of", "uncertain", "conditional", "generic"):
        # Pydantic v2 spells `X | None` as an anyOf including {"type": "null"}.
        assert "anyOf" in props[name], f"{name} is not nullable in the OpenAPI schema"
        assert {"type": "null"} in props[name]["anyOf"]
    assert props["negated"] == {"title": "Negated", "type": "boolean"}


def test_group_filter(client):
    body = client.post(
        "/analyze", json={"text": "pain and aspirin", "groups": ["DRUG"]}
    ).json()
    assert [a["group"] for a in body["annotations"]] == ["DRUG"]


def test_group_filter_is_case_insensitive(client):
    body = client.post(
        "/analyze", json={"text": "pain and aspirin", "groups": ["drug"]}
    ).json()
    assert [a["group"] for a in body["annotations"]] == ["DRUG"]


def test_negated_only(client):
    body = client.post(
        "/analyze", json={"text": "pain and aspirin", "negated_only": True}
    ).json()
    assert all(a["negated"] for a in body["annotations"])
    assert body["n_annotations"] == 1


def test_affirmed_only(client):
    body = client.post(
        "/analyze", json={"text": "pain and aspirin", "affirmed_only": True}
    ).json()
    assert not any(a["negated"] for a in body["annotations"])


def test_conflicting_polarity_flags_are_422(client):
    r = client.post(
        "/analyze",
        json={"text": "x", "negated_only": True, "affirmed_only": True},
    )
    assert r.status_code == 422


def test_empty_text_is_accepted_and_returns_nothing(client):
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 200
    assert r.json()["n_annotations"] == 0


def test_oversized_document_is_413_not_analyzed(client):
    from umlsmatch.service import api

    r = client.post("/analyze", json={"text": "x" * (api.MAX_CHARS + 1)})
    assert r.status_code == 413
    # Rejected before reaching a pipeline.
    assert all(p.calls == 0 for p in client.pool._all)


def test_missing_text_field_is_422(client):
    assert client.post("/analyze", json={}).status_code == 422


# --- batch -------------------------------------------------------------------


def test_batch_analyzes_every_document(client):
    r = client.post("/analyze/batch", json={"documents": ["pain", "aspirin", "nothing"]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_documents"] == 3
    assert [d["n_annotations"] for d in body["results"]] == [1, 1, 0]


def test_batch_echoes_ids_so_results_need_no_positional_trust(client):
    body = client.post(
        "/analyze/batch", json={"documents": ["pain", "aspirin"], "ids": ["a", "b"]}
    ).json()
    assert [d["id"] for d in body["results"]] == ["a", "b"]


def test_batch_id_length_mismatch_is_422(client):
    r = client.post(
        "/analyze/batch", json={"documents": ["a", "b"], "ids": ["only-one"]}
    )
    assert r.status_code == 422


def test_one_bad_document_does_not_void_the_batch(client):
    body = client.post(
        "/analyze/batch", json={"documents": ["pain", "BOOM", "aspirin"]}
    ).json()
    assert body["results"][1]["error"].startswith("ValueError")
    assert body["results"][0]["n_annotations"] == 1
    assert body["results"][2]["n_annotations"] == 1


def test_batch_error_does_not_leak_document_text(client):
    """A failure message must never echo note content."""
    secret = "PATIENT SSN 123-45-6789 BOOM"
    body = client.post("/analyze/batch", json={"documents": [secret]}).json()
    assert "123-45-6789" not in body["results"][0]["error"]


def test_empty_batch_is_422(client):
    assert client.post("/analyze/batch", json={"documents": []}).status_code == 422


# --- exception logging -------------------------------------------------------
#
# The response has never carried an exception message. The *log* has, which is
# a defensible trade on an isolated deployment and the wrong one when logs ship
# to a shared aggregator. These pin the switch and the redaction it selects.


class LeakyPipeline(FakePipeline):
    """Raises the way spaCy and sqlite3 are entitled to: quoting the input.

    The existing ``FakePipeline`` raises ``ValueError("synthetic failure")``,
    whose message is ours and carries nothing. That is the easy case; this is
    the one the redaction exists for.
    """

    def analyze(self, text: str):
        if "LEAKY" in text:
            try:
                raise KeyError(f"sqlite3 could not parse {text!r}")
            except KeyError as cause:
                raise RuntimeError(f"tokenizer failed on {text!r}") from cause
        return super().analyze(text)


def _leaky_client():
    pool = PipelinePool(size=1)
    pool._all = [LeakyPipeline()]
    pool._free.put(pool._all[0])
    pool._started = True
    return TestClient(create_app(pool))


NOTE = "LEAKY: patient MRN 998877 with chest pain"


def test_redacted_traceback_keeps_the_call_path_and_drops_messages():
    from umlsmatch.service.api import _redacted_traceback

    try:
        raise ValueError("note text 123-45-6789")
    except ValueError as exc:
        rendered = _redacted_traceback(exc)

    assert "ValueError" in rendered
    assert "test_service.py:" in rendered  # the call path survives
    assert "123-45-6789" not in rendered


def test_redacted_traceback_walks_chained_causes():
    """A re-raise carries the original message on ``__cause__``.

    Reporting only the outermost type would leak through the chain that
    ``log.exception`` prints anyway.
    """
    from umlsmatch.service.api import _redacted_traceback

    try:
        try:
            raise KeyError("inner 123-45-6789")
        except KeyError as cause:
            raise RuntimeError("outer 987-65-4321") from cause
    except RuntimeError as exc:
        rendered = _redacted_traceback(exc)

    assert "RuntimeError" in rendered and "KeyError" in rendered
    assert "123-45-6789" not in rendered
    assert "987-65-4321" not in rendered


def test_log_exception_detail_off_keeps_note_text_out_of_the_log(monkeypatch, caplog):
    from umlsmatch.service import api

    monkeypatch.setattr(api, "LOG_EXCEPTION_DETAIL", False)
    with caplog.at_level(logging.DEBUG, logger="umlsmatch.service"), _leaky_client() as c:
        body = c.post("/analyze/batch", json={"documents": [NOTE]}).json()

    assert body["results"][0]["error"] == "RuntimeError"
    logged = caplog.text
    assert "998877" not in logged
    assert "chest pain" not in logged
    # Still diagnosable: type, call path and which document failed.
    assert "RuntimeError" in logged and "batch document 0 failed" in logged


def test_log_exception_detail_defaults_to_on(monkeypatch, caplog):
    """The default is unchanged behaviour, and this test says so out loud.

    Tightening a privacy control silently on upgrade is its own kind of
    surprise; the deployment that wants the stricter mode asks for it, and
    ``docs/SERVICE.md`` states the consequence of leaving it alone.
    """
    from umlsmatch.service import api

    assert api._env_bool("UMLSMATCH_LOG_EXCEPTION_DETAIL", True) is True
    monkeypatch.setattr(api, "LOG_EXCEPTION_DETAIL", True)
    with caplog.at_level(logging.DEBUG, logger="umlsmatch.service"), _leaky_client() as c:
        c.post("/analyze/batch", json={"documents": [NOTE]})

    assert "998877" in caplog.text  # the trade-off, made visible


def test_info_reports_the_logging_policy(client):
    """A privacy setting nobody can confirm from outside is half a control."""
    assert client.get("/info").json()["log_exception_detail"] in (True, False)


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_env_bool_accepts_true_spellings(monkeypatch, raw):
    from umlsmatch.service import api

    monkeypatch.setenv("UMLSMATCH_TEST_FLAG", raw)
    assert api._env_bool("UMLSMATCH_TEST_FLAG", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "NO", "off"])
def test_env_bool_accepts_false_spellings(monkeypatch, raw):
    from umlsmatch.service import api

    monkeypatch.setenv("UMLSMATCH_TEST_FLAG", raw)
    assert api._env_bool("UMLSMATCH_TEST_FLAG", True) is False


def test_env_bool_rejects_a_typo_rather_than_defaulting(monkeypatch):
    """For a privacy switch, silently falling back to the default is the wrong
    failure -- ``UMLSMATCH_LOG_EXCEPTION_DETAIL=flase`` must not read as "on"."""
    from umlsmatch.service import api

    monkeypatch.setenv("UMLSMATCH_TEST_FLAG", "flase")
    with pytest.raises(ValueError, match="UMLSMATCH_TEST_FLAG"):
        api._env_bool("UMLSMATCH_TEST_FLAG", True)


def test_oversized_batch_is_413(client):
    from umlsmatch.service import api

    r = client.post(
        "/analyze/batch", json={"documents": ["x"] * (api.MAX_BATCH + 1)}
    )
    assert r.status_code == 413


def test_batch_holds_a_single_pipeline_for_the_whole_call(client):
    """One lease per batch, not per document."""
    client.post("/analyze/batch", json={"documents": ["pain"] * 5})
    used = [p for p in client.pool._all if p.calls]
    assert len(used) == 1
    assert used[0].calls == 5


# --- pool --------------------------------------------------------------------


def test_pool_never_hands_one_pipeline_to_two_threads():
    """The invariant the whole pool exists for."""
    pool = make_pool(size=3, delay=0.02)
    # Held separately: lifespan shutdown clears the pool on context exit, so
    # asserting against pool._all afterwards silently checks an empty list.
    pipelines = list(pool._all)
    app = create_app(pool)
    with TestClient(app) as c:
        errors = []

        def hit():
            try:
                assert c.post("/analyze", json={"text": "pain"}).status_code == 200
            except Exception as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors
    assert sum(p.calls for p in pipelines) == 12
    # No instance was ever used by two threads at once.
    assert all(p.max_concurrent == 1 for p in pipelines)
    # And the work actually spread across the pool rather than serializing.
    assert sum(1 for p in pipelines if p.calls) > 1


def test_pool_gives_each_pipeline_its_own_model(monkeypatch):
    """The pool's isolation has to reach the spaCy model too.

    ``ClinicalPipeline`` holds a model *name* and resolves it through a cached
    loader, so distinct pipelines used to share one ``Language`` and threads
    ran inference on it concurrently. Driven through a recorder rather than a
    real pipeline: what is under test is which kwargs the pool constructs, and
    a real one needs a 589 MB dictionary.
    """
    built: list[object] = []

    class Recorder:
        def __init__(self, db_path, **kwargs):
            self.db_path = db_path
            built.append(kwargs.get("model"))

        def close(self):
            pass

    sentinels = iter(["model-0", "model-1", "model-2"])
    monkeypatch.setattr("umlsmatch.analyze.ClinicalPipeline", Recorder)
    monkeypatch.setattr(
        "umlsmatch.pipeline.tokenizer.new_model", lambda name: next(sentinels)
    )

    pool = PipelinePool(size=3)
    pool.start()

    assert built == ["model-0", "model-1", "model-2"]
    assert len(set(map(id, built))) == 3


def test_pool_honours_an_explicitly_preloaded_model(monkeypatch):
    """Passing a ``Language`` says *which object*; the pool must not override it.

    Sharing one model across the pool stays available for a deployment that
    has measured it and wants the memory back -- it is only no longer the
    accidental default.
    """
    built: list[object] = []

    class Recorder:
        def __init__(self, db_path, **kwargs):
            self.db_path = db_path
            built.append(kwargs.get("model"))

        def close(self):
            pass

    class FakeLanguage:  # stands in for spacy.language.Language
        pass

    shared = FakeLanguage()
    monkeypatch.setattr("umlsmatch.analyze.ClinicalPipeline", Recorder)
    monkeypatch.setattr(
        "umlsmatch.pipeline.tokenizer.new_model",
        lambda name: pytest.fail("should not load a model when one was supplied"),
    )

    pool = PipelinePool(size=2, model=shared)
    pool.start()

    assert built == [shared, shared]


def test_pool_exhaustion_returns_503_with_retry_after():
    """Shed load rather than queue without bound."""
    pool = make_pool(size=1, delay=0.5, lease_timeout=0.05)
    app = create_app(pool)
    with TestClient(app) as c:
        results = []

        def hit():
            results.append(c.post("/analyze", json={"text": "pain"}))

        threads = [threading.Thread(target=hit) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    codes = sorted(r.status_code for r in results)
    assert 503 in codes, f"expected at least one shed request, got {codes}"
    shed = next(r for r in results if r.status_code == 503)
    assert shed.headers["Retry-After"] == "5"


def test_lease_returns_the_pipeline_even_when_analysis_raises():
    """A leak here permanently reduces capacity; `size` leaks would deadlock."""
    pool = make_pool(size=1)
    before = pool.available
    with pytest.raises(ValueError), pool.lease() as p:
        p.analyze("BOOM")
    assert pool.available == before


def test_lease_on_an_unstarted_pool_raises_rather_than_hanging():
    pool = PipelinePool(size=1)
    with pytest.raises(PoolExhausted), pool.lease():
        pass


def test_close_closes_every_pipeline_and_is_idempotent():
    pool = make_pool(size=2)
    pipelines = list(pool._all)
    pool.close()
    pool.close()
    assert all(p.closed for p in pipelines)
    assert not pool.ready()


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_pool_size_is_rejected(size):
    with pytest.raises(ValueError):
        PipelinePool(size=size)


def test_invalid_lease_timeout_is_rejected():
    with pytest.raises(ValueError):
        PipelinePool(size=1, lease_timeout=0)


# --- metrics -----------------------------------------------------------------


def test_metrics_exposes_prometheus_text(client):
    client.post("/analyze", json={"text": "pain"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "umlsmatch_documents_total" in body
    assert "umlsmatch_pool_size" in body


def test_metrics_do_not_contain_document_text(client):
    """Metrics are scraped by systems with wider access than the API."""
    client.post("/analyze", json={"text": "Patient Jane Doe has pain."})
    assert "Jane" not in client.get("/metrics").text


def _request_count(client, endpoint: str, status: str) -> float:
    """Read one ``umlsmatch_requests_total`` series out of the exposition."""
    wanted = f'umlsmatch_requests_total{{endpoint="{endpoint}",status="{status}"}}'
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(wanted):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def _oversized_batch() -> dict:
    from umlsmatch.service import api

    return {"documents": ["x"] * (api.MAX_BATCH + 1)}


@pytest.mark.parametrize(
    ("path", "payload", "status"),
    [
        ("/analyze", {}, "422"),                                # pydantic
        ("/analyze", {"text": "x", "negated_only": True,
                      "affirmed_only": True}, "422"),           # raised in-handler
        ("/analyze/batch", {"documents": []}, "422"),
        ("/analyze/batch", _oversized_batch(), "413"),
    ],
)
def test_every_rejection_is_counted(client, path, payload, status):
    """The error-rate metric must move when a client is turned away.

    ``REQUESTS`` was incremented only on the 200 path, in the pool-exhausted
    handler, and in the body-size middleware, so all four of these were
    invisible: an alert on error rate read clean while requests were being
    rejected. Both the pydantic 422 and the in-handler ``raise`` are covered
    because they reach the counter by different routes.
    """
    before = _request_count(client, path, status)
    assert client.post(path, json=payload).status_code == int(status)
    assert _request_count(client, path, status) == before + 1


def test_the_known_endpoint_set_matches_the_app(client):
    """A new route must be added to ``KNOWN_ENDPOINTS`` or its metrics vanish.

    Bucketing to ``<other>`` is the safe failure -- a series is lost, not an
    unbounded number created -- but it is still a silent one, so pin it.
    """
    from umlsmatch.service import api

    routed = {
        route.path
        for route in client.app.routes
        if getattr(route, "path", "").startswith("/")
        and route.path not in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc")
    }
    assert routed == set(api.KNOWN_ENDPOINTS)


def test_an_unknown_path_does_not_mint_a_new_metric_series(client):
    """Metric labels must come from a fixed set.

    ``request.url.path`` is caller-controlled, so counting 404s under it would
    create one Prometheus time series per URL a scanner tries. The error counter
    is both where unmatched paths land and the series least able to survive
    unbounded cardinality.
    """
    from umlsmatch.service import api

    for path in ("/wp-login.php", "/../etc/passwd", "/analyze/../../x"):
        client.get(path)
    body = client.get("/metrics").text
    assert "wp-login" not in body
    assert "passwd" not in body
    assert f'endpoint="{api.UNMATCHED_ENDPOINT}"' in body


def test_a_validation_error_does_not_echo_the_submitted_text(client):
    """422 bodies are a PHI path: FastAPI's default handler echoes the input.

    A client that sends `text` as the wrong JSON type would otherwise get note
    content back in the response, and into whatever logs it.
    """
    secret = "Jane Doe MRN 123-45-6789 denies chest pain"
    r = client.post("/analyze", json={"text": [secret]})
    assert r.status_code == 422
    assert "123-45-6789" not in r.text
    assert secret not in r.text
    # Still actionable: the client learns which field was wrong and why.
    detail = r.json()["detail"][0]
    assert detail["loc"] == ["body", "text"]
    assert detail["msg"]
    assert "input" not in detail

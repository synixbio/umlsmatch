"""A bounded pool of pipelines, because ``ClinicalPipeline`` is not thread-safe.

This is the whole reason the service needs more than a module-level pipeline.
FastAPI runs a synchronous endpoint in Starlette's threadpool, so several
requests execute the handler concurrently on different threads. A single shared
:class:`~umlsmatch.analyze.ClinicalPipeline` would have them interleave on
``RareWordMatcher``'s unsynchronized candidate and concept memo dicts -- a race
that corrupts cached rows rather than raising, so it would surface as wrong
annotations under load and nothing at all in testing.

Three designs were available:

  * **One pipeline behind a lock.** Correct, but serializes every request, so
    the service would run at single-document throughput no matter the hardware.
  * **A pipeline per thread** (``threading.local``). Correct and concurrent, but
    unbounded: Starlette's default threadpool is 40 threads, and each pipeline
    holds a SQLite connection with a ~200 MB page cache plus a loaded spaCy
    model. Forty of those is not a memory budget anyone intends.
  * **A fixed pool, leased per request** -- this. Concurrency is capped at a
    number you choose and memory is that number times one pipeline, which is
    the pair of properties a deployment actually needs to reason about.

Requests beyond the pool size wait for a lease, and wait longer than
``lease_timeout`` fails the request rather than queueing without bound: shedding
load with a 503 lets a caller retry or shift, while an unbounded queue turns a
throughput shortfall into rising latency and eventual memory exhaustion.

Sizing: concurrency is CPU-bound (tokenize + match), so the pool wants to track
cores, not connections. Past roughly one pipeline per core the threads only
contend. See ``docs/SERVICE.md``.

**Isolation has to cover the spaCy model too**, which it did not until
:meth:`PipelinePool._kwargs_for_one` -- ``tokenizer.load_model`` is cached per
process, so leasing distinct ``ClinicalPipeline`` objects still handed every
thread the same ``Language``. See that method and
:func:`~umlsmatch.pipeline.tokenizer.new_model`. The cost is one model per
pipeline; see the memory table in ``docs/SERVICE.md``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["PipelinePool", "PoolExhausted"]

log = logging.getLogger(__name__)


class PoolExhausted(RuntimeError):
    """No pipeline became available within the lease timeout."""


class PipelinePool:
    """Fixed-size pool of :class:`~umlsmatch.analyze.ClinicalPipeline` instances.

    Args:
        size: how many pipelines to build, and therefore the maximum number of
            documents analyzed concurrently.
        db_path: dictionary path; ``None`` uses the usual resolution order
            (see :func:`~umlsmatch.analyze.find_dictionary`).
        lease_timeout: seconds a request may wait for a free pipeline.
        pipeline_kwargs: forwarded to every ``ClinicalPipeline``.
    """

    def __init__(
        self,
        size: int = 2,
        *,
        db_path: str | Path | None = None,
        lease_timeout: float = 30.0,
        **pipeline_kwargs: Any,
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1, got {size}")
        if lease_timeout <= 0:
            raise ValueError(f"lease_timeout must be > 0, got {lease_timeout}")

        self.size = size
        self.db_path = db_path
        self.lease_timeout = lease_timeout
        self._pipeline_kwargs = pipeline_kwargs

        self._free: queue.Queue = queue.Queue(maxsize=size)
        self._all: list[Any] = []
        self._lock = threading.Lock()
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    def _kwargs_for_one(self) -> dict[str, Any]:
        """Constructor arguments for one pipeline, with its own spaCy model.

        The pool exists so that no two threads touch one pipeline's mutable
        state, and it got all of that state except the spaCy ``Language``:
        ``ClinicalPipeline`` keeps the model *name*, and
        ``tokenizer.load_model`` is ``lru_cache``d, so every pipeline here
        resolved to the same object and concurrent requests ran ``nlp(text)``
        on it together. :func:`~umlsmatch.pipeline.tokenizer.new_model`
        explains why that is a hazard; this is where it is closed.

        A caller who passes an already-loaded ``Language`` has said explicitly
        which object to use, so it is honoured as given -- sharing one model
        across the pool remains available, it is just no longer what you get
        by accident. A string (or nothing) means "this model", not "this
        instance", and each pipeline gets its own.
        """
        kwargs = dict(self._pipeline_kwargs)
        requested = kwargs.get("model")
        if requested is None or isinstance(requested, str):
            from umlsmatch.pipeline.tokenizer import DEFAULT_MODEL, new_model

            kwargs["model"] = new_model(requested or DEFAULT_MODEL)
        return kwargs

    def start(self) -> None:
        """Build every pipeline. Slow (seconds each) and deliberately eager.

        Constructing on first use instead would push model loading into the
        first few requests and make early latency wildly misleading. Building
        here is what lets :meth:`ready` gate traffic honestly -- a readiness
        probe that returns true before the pool can serve anything is worse
        than no probe.
        """
        with self._lock:
            if self._started:
                return
            from umlsmatch.analyze import ClinicalPipeline

            started = time.monotonic()
            for i in range(self.size):
                pipeline = ClinicalPipeline(self.db_path, **self._kwargs_for_one())
                self._all.append(pipeline)
                self._free.put(pipeline)
                log.info("pipeline %d/%d ready", i + 1, self.size)
            self._started = True
            log.info(
                "pool of %d ready in %.1fs (dictionary: %s)",
                self.size,
                time.monotonic() - started,
                self._all[0].db_path if self._all else "?",
            )

    def close(self) -> None:
        """Close every pipeline. Safe to call twice."""
        with self._lock:
            for pipeline in self._all:
                try:
                    pipeline.close()
                # Broad by intent: shutdown must close every remaining pipeline
                # even if one connection is already broken.
                except Exception:
                    log.warning("error closing a pipeline", exc_info=True)
            self._all.clear()
            self._started = False
            while not self._free.empty():
                try:
                    self._free.get_nowait()
                except queue.Empty:  # pragma: no cover - drained concurrently
                    break

    # -- state -------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def available(self) -> int:
        """Pipelines not currently leased. Approximate under concurrency."""
        return self._free.qsize()

    def ready(self) -> bool:
        """True once :meth:`start` has built at least one pipeline.

        Stricter than :attr:`started` on purpose: it is what ``/ready`` answers,
        and a pool flagged started but holding nothing would route traffic to a
        service that cannot analyze anything.
        """
        return self._started and bool(self._all)

    def memo_entries(self) -> tuple[int, int]:
        """``(candidate, concept)`` memo entries summed across every pipeline.

        Read without a lease on purpose: this is a metrics sample, two ``len()``
        calls per pipeline, and blocking a scrape behind a busy pool would make
        the gauge disappear exactly when the pool is under load. A torn read
        costs an off-by-a-few gauge and nothing else.

        Pipelines are duck-typed here (tests inject fakes to avoid building a
        real one), so anything without a matcher simply contributes nothing.
        A metrics helper must not be the reason an injected pipeline stops
        working.
        """
        totals = [
            matcher.memo_entries
            for p in self._all
            if (matcher := getattr(p, "matcher", None)) is not None
        ]
        return sum(t[0] for t in totals), sum(t[1] for t in totals)

    # -- use ---------------------------------------------------------------

    @contextmanager
    def lease(self, timeout: float | None = None) -> Iterator[Any]:
        """Borrow a pipeline for the duration of the block.

        The pipeline is returned in a ``finally``, so an exception raised while
        analyzing -- including a client disconnect propagating as a
        cancellation -- cannot leak one out of the pool. Leaking even a single
        pipeline permanently reduces capacity, and leaking `size` of them
        deadlocks the service with no error to point at.
        """
        if not self._started:
            raise PoolExhausted("pool is not started")
        wait = self.lease_timeout if timeout is None else timeout
        try:
            pipeline = self._free.get(timeout=wait)
        except queue.Empty:
            raise PoolExhausted(
                f"no pipeline available within {wait:g}s "
                f"(pool size {self.size}; all busy)"
            ) from None
        try:
            yield pipeline
        finally:
            self._free.put(pipeline)

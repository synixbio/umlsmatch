"""HTTP service for the pipeline.

    uvicorn umlsmatch.service:app --port 8000

``app`` is imported lazily so that ``import umlsmatch.service.pool`` -- and the
test suite generally -- does not require FastAPI to be installed when only the
pool is under test. Importing the name raises an :class:`ImportError` naming the
``service`` extra, matching how :mod:`umlsmatch` handles the ``nlp`` extra.

The implementation lives in ``api.py`` rather than ``app.py`` deliberately: a
submodule named ``app`` would shadow this module's ``app`` attribute once
imported, so ``uvicorn umlsmatch.service:app`` would resolve to the *module*
instead of the application and fail with a confusing error.
"""

from __future__ import annotations

__all__ = ["PipelinePool", "PoolExhausted", "app", "create_app"]

from umlsmatch.service.pool import PipelinePool, PoolExhausted


def __getattr__(name: str) -> object:
    if name in ("app", "create_app"):
        try:
            from umlsmatch.service import api as _api
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The HTTP service needs the optional 'service' extra "
                '(FastAPI, uvicorn, prometheus-client). Install it with:  '
                'pip install -e ".[service]"'
            ) from exc
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

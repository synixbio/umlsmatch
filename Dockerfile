# syntax=docker/dockerfile:1

# Two stages so the spaCy model download happens once, at build time. The
# runtime image must work with no outbound network at all, so anything fetched
# from the internet has to be baked in here.

FROM python:3.12-slim AS build

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Install into a venv we can copy wholesale into the runtime stage, which keeps
# build toolchains out of the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first, against a stub package: this layer is the expensive one
# (spaCy, FastAPI, uvicorn, plus the en_core_web_sm download -- a genuine
# network fetch) and should only rebuild when the dependency set changes.
#
# The stub is what makes that possible. `pip install .` needs a package to
# install, so copying src/ before this line -- as this file used to -- put every
# source edit upstream of the install and re-downloaded the whole dependency
# set on every build. Installing against an empty package resolves and installs
# the dependencies; the real source arrives afterwards and goes in with
# --no-deps, which is metadata-only.
#
# The stub must define __version__: [tool.hatch.version] reads it from this
# exact file, and hatchling fails the build without it.
COPY pyproject.toml README.md ./
RUN mkdir -p src/umlsmatch \
    && printf '__version__ = "0.0.0"\n' > src/umlsmatch/__init__.py \
    && pip install --upgrade pip \
    && pip install ".[nlp,service]" \
    && python -m spacy download en_core_web_sm

COPY src/ ./src/
RUN pip install --no-deps --force-reinstall .


FROM python:3.12-slim AS runtime

# curl is for the container healthcheck below. Nothing else is added: the
# runtime has no compiler and no package index configured.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The dictionary is mounted read-only and nothing here writes to disk,
# so the service has no reason to run with write access to its own image.
RUN useradd --create-home --uid 10001 umlsmatch

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Where the compose file mounts the dictionary. Resolved by find_dictionary().
ENV UMLSMATCH_DB=/data/umls_sno_rx.sqlite \
    UMLSMATCH_POOL_SIZE=2

USER umlsmatch
WORKDIR /home/umlsmatch
EXPOSE 8000

# Readiness, not liveness: the pool takes seconds to build, and a container
# marked healthy before it can serve would receive traffic it must reject.
# start-period covers that build without counting it as a failure.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

# One uvicorn worker per container. Analysis is pure-Python and GIL-bound, so
# in-process concurrency saturates quickly (measured: pool 1 -> 13.4 docs/sec,
# pool 4 -> 17.4). Scale with container replicas, which get their own
# interpreter and therefore their own GIL. See docs/SERVICE.md.
CMD ["uvicorn", "umlsmatch.service:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]

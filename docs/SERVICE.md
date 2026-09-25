# HTTP service

```bash
pip install -e ".[nlp,service]"
uvicorn umlsmatch.service:app --host 127.0.0.1 --port 8000
```

Interactive docs at `/docs`. Request and response bodies contain PHI — read
[Deployment](#deployment) before exposing this anywhere.

## Endpoints

| method | path | purpose |
|---|---|---|
| POST | `/analyze` | one document |
| POST | `/analyze/batch` | many documents, one pipeline lease |
| GET | `/health` | **liveness** — the process is up |
| GET | `/ready` | **readiness** — the pool is built and can serve |
| GET | `/metrics` | Prometheus exposition |
| GET | `/info` | resolved config and dictionary path |

```bash
curl -s 127.0.0.1:8000/analyze -H 'content-type: application/json' \
     -d '{"text":"Patient denies chest pain.","groups":["FINDING"]}'
```

```json
{"annotations":[
  {"cui":"C0008031","text":"chest pain","start":15,"end":25,"group":"FINDING",
   "negated":true,"preferred_text":"CHEST PAIN","term":"chest pain",
   "subject":"patient","history_of":false,"uncertain":false,
   "conditional":null,"generic":null},
  {"cui":"C0030193","text":"pain","start":21,"end":25,"group":"FINDING",
   "negated":true,"preferred_text":"Ache","term":"pain",
   "subject":"patient","history_of":false,"uncertain":false,
   "conditional":null,"generic":null}],
 "n_annotations":2,"took_ms":3.1}
```

Two annotations for one phrase is not a bug: `chest pain` and `pain` both
match, and the API does not suppress the shorter one. See the README on
overlapping matches. `generic` is `null` on every annotation because it is
never assessed, and `conditional` is `null` unless the deployment enabled it
(it is a prototype, off by default) — branch on `None`, do not coerce it. A
client cannot tell from one response which attributes a deployment assesses,
because an unassessed attribute and a disabled one are the same `null`.

Validation failures answer `422` with `type`, `loc` and `msg` per error and
**no `input` key**. FastAPI's default handler echoes the offending value, which
for `{"text": ["<a note>"]}` would put note content in a response body; the
service replaces that handler. The client still learns which field was wrong
and why.

`/analyze/batch` takes `documents` and optional `ids`, which are echoed back so
results can be reattached without trusting ordering. A document that fails is
reported in its own `error` field and the rest still run.

**Wire the probes to the right things.** `/health` is liveness and answers "is
this process alive"; `/ready` is readiness and is `503` for the seconds the pool
spends building. Pointing a readiness probe at `/health` routes traffic into a
service that cannot serve yet; pointing a liveness probe at `/ready` restarts a
healthy service that is merely warming up.

`umlsmatch_requests_total{endpoint,status}` counts **every** answered request,
rejections included — the 422s, both 413s, and the 503 the pool sheds under
load. It is the series to alert an error rate on. It did not always count the
rejections, so a dashboard built against an older build read clean while clients
were being turned away; if you have one, check it moves when you post
`{"documents": []}` to `/analyze/batch`.

## Configuration

All environment variables.

| variable | default | effect |
|---|---|---|
| `UMLSMATCH_DB` | auto | dictionary path |
| `UMLSMATCH_PROFILE` | `strict` | named configuration every pipeline in the pool is built with — `strict` or `clinical_recall`. Changes what the service *reports*, not just how fast: see [Two profiles](USER_GUIDE.md#two-profiles). A typo fails pool construction at startup, so `/ready` never passes. `/info` reports the live value |
| `UMLSMATCH_POOL_SIZE` | `2` | concurrent analyses, and the memory multiplier |
| `UMLSMATCH_LEASE_TIMEOUT` | `30` | seconds to wait for a pipeline before `503` |
| `UMLSMATCH_MAX_BATCH` | `64` | documents per batch request |
| `UMLSMATCH_MAX_CHARS` | `1000000` | characters per document |
| `UMLSMATCH_MAX_BODY_BYTES` | `16777216` | request body bytes. Refused from `Content-Length` before the body is read, and counted as it streams for a request that declares no length |
| `UMLSMATCH_LOG_EXCEPTION_DETAIL` | `true` | whether a failing document's exception *message* reaches the log. See [PHI and egress](#phi-and-egress) — with the default, treat logs as PHI |
| `UMLSMATCH_API_KEY` | unset | when set, `/analyze` and `/analyze/batch` require it in an `X-API-Key` header; anything else gets `401`. Probes and `/metrics` stay open — see [Deploying this safely](#deploying-this-safely--what-the-service-does-not-do). `/info` reports whether one is required, never its value |

`MAX_BATCH` and `MAX_CHARS` bound the analysis but are checked inside the
handler, after the body is parsed; `MAX_BODY_BYTES` is what bounds the parsing.
Their product (~64 MB) is the ceiling it replaces.

## Why there is a pipeline pool

`ClinicalPipeline` is **not thread-safe** — `RareWordMatcher`'s candidate and
concept caches are unsynchronized dicts. FastAPI runs sync endpoints in
Starlette's threadpool, so a single shared pipeline would have concurrent
requests interleave on those dicts. That corrupts cached rows rather than
raising, so it surfaces as *wrong annotations under load* and nothing at all in
testing.

A fixed pool leased per request bounds concurrency and memory to one number you
choose. The alternatives were a global lock (correct, but serializes everything)
and a pipeline per thread (correct, but Starlette's threadpool is 40 threads).
`test_pool_never_hands_one_pipeline_to_two_threads` pins the invariant.

Past the pool size, requests wait up to `UMLSMATCH_LEASE_TIMEOUT` and then get
`503` with `Retry-After`. Shedding load is deliberate: an unbounded queue turns
a throughput shortfall into unbounded latency and eventually memory exhaustion.

## Measured performance

244,538 characters of real clinical notes, one uvicorn worker, warm caches,
client over loopback. **That corpus is not distributed here**, so these exact
numbers cannot be reproduced from this repository — re-measure on your notes with
[tools/load_test.py](../tools/load_test.py). The shape of the result is the
transferable part, not the absolute rates.

| pool size | concurrency | docs/sec | mean | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.7 | 0.102 | 0.175 | 0.289 |
| 1 | 4 | 13.4 | 0.291 | 0.339 | 0.361 |
| 1 | 8 | 13.4 | 0.551 | 0.644 | 0.681 |
| 4 | 1 | 10.3 | 0.096 | 0.139 | 0.179 |
| 4 | 4 | 16.9 | 0.231 | 0.378 | 0.445 |
| 4 | 8 | 17.4 | 0.434 | 0.555 | 0.747 |

**Quadrupling the pool bought 1.3×, not 4×.** Analysis is pure Python and
GIL-bound, so in-process concurrency saturates quickly. The practical
consequences:

- **Treat `POOL_SIZE` 1–2 as the practical ceiling, not just the default.**
  This is a bound rather than advice: above it you are paying ~107 MB per
  pipeline (see Memory) to buy threads that mostly contend, and past ~4 the
  measured throughput gain is inside the noise.
- **Scale with processes, not pool size** — container replicas (or
  `uvicorn --workers`) each get their own interpreter and their own GIL. The
  same holds for batch work: one pipeline per worker process.
- Throughput is ~55,000–96,000 characters/sec, which is the more stable figure;
  per-document rates depend entirely on note length.

### Memory

Measured, warmed on real notes:

| | RSS |
|---|---:|
| interpreter + imports | ~26 MB |
| \+ first pipeline | ~207 MB |
| each additional pipeline | ~+107 MB |
| — of which, its own spaCy model | ~+40 MB |

The dictionary is **not** loaded into memory — it is read from SQLite on demand,
which is why a 589 MB dictionary costs ~180 MB resident rather than 589.

**Why an additional pipeline costs more than it used to.** Every pooled
pipeline now loads its own spaCy model
([`tokenizer.new_model`](../src/umlsmatch/pipeline/tokenizer.py)). Before that
they shared one, because `load_model` is `lru_cache`d and a pipeline holds a
model *name* — so the pool isolated the matcher, the SQLite connection and the
memo caches, and then handed every thread the same `Language` to run inference
on concurrently. A `Language` is safe one thread at a time: the `Vocab`
interns every unseen string and the tokenizer caches segmentations, both
written during ordinary inference and neither synchronized. Clinical text
produces unseen strings constantly.

The ~+40 MB is one more `en_core_web_sm` and was measured directly, as the
difference between a shared and an unshared model across two warm protocols
(39 MB and 42 MB). The rest of the per-pipeline cost is the SQLite page cache
filling as that connection touches more of the dictionary, so the figure you
see depends on how much of the vocabulary your traffic reaches — measure it on
your own notes rather than adopting the number here.

To get the memory back on a deployment that has decided the risk is acceptable,
pass a preloaded model explicitly — `PipelinePool(..., model=load_model())` —
which the pool honours as given. That is now a deliberate choice rather than
the default.

Both of the caches that grow with the distinct vocabulary seen are bounded:
SQLite's page cache at ~195 MB per connection (`PRAGMA cache_size`, tunable via
`RareWordMatcher(cache_size=...)`), and the matcher's candidate and concept
memos at `memo_size` entries each, evicted least-recently-used. Bounding the
memos matters because a pooled pipeline lives for the life of the process and
would otherwise retain every novel token norm it ever saw. That growth is
sublinear in documents (distinct token *types*, not occurrences), so it is
a slow climb rather than a crash, which is precisely why it would reach
production.

Budget ~450 MB per worker for a long-lived process at `POOL_SIZE=2` (raised
from ~400 MB when each pipeline gained its own model). Expect a rising curve that
plateaus rather than a flat line, and confirm the plateau instead of assuming
it: `umlsmatch_matcher_memo_entries{cache="candidate"|"concept"}` reports the
entries held across the pool, and tops out at `pool_size × memo_size`. A gauge
sitting well below that bound means `memo_size` could be lowered; one pinned at
it means evictions are happening and the cache may be too small to be earning
its keep.

> Measure before provisioning replica counts: a per-worker estimate that is out
> by even a small factor compounds across the fleet.

## Deployment

```bash
UMLSMATCH_DICT_DIR=/srv/umlsmatch UMLSMATCH_BIND=0.0.0.0 \
  docker compose up -d --scale api=4
```

The dictionary is mounted **read-only** — the matcher opens SQLite with
`mode=ro` and must never mutate a ~590 MB artifact that takes tens of minutes
to rebuild. Ports bind to loopback unless `UMLSMATCH_BIND` is set deliberately, so an
unconfigured `up` cannot expose PHI to the whole network.

> The `Dockerfile` and `docker-compose.yml` are **written but not executed** —
> there is no Docker daemon in the development environment. Treat the first
> `docker compose build` as the real test.

### Deploying this safely — what the service does *not* do

**This service has no authentication, no authorization, and no TLS.** Every
endpoint is open to anything that can reach the port, and request and response
bodies are PHI. That is a deliberate delegation to the deployment layer, not an
oversight, but it makes the following a *requirement* rather than a
recommendation:

- **Run it behind an authenticating reverse proxy on a private network.** The
  proxy terminates TLS and decides who may call `/analyze`. Nothing in this
  process terminates TLS.
- **Set `UMLSMATCH_API_KEY` as well.** It is a backstop, not the control: a
  shared secret in a header is weaker than the proxy's authentication and, over
  plain HTTP, travels in cleartext. What it buys is that the proxy stops being
  a single point of failure — a misconfigured ingress, or a container that bound
  `0.0.0.0` on a flatter network than intended, then meets a `401` rather than
  an open `/analyze`. Off by default so an upgrade does not break a working
  deployment; `/info` reports `api_key_required` so you can confirm it is on.

  ```bash
  curl -s 127.0.0.1:8000/analyze -H 'x-api-key: <key>' \
    -H 'content-type: application/json' -d '{"text":"Patient denies chest pain."}'
  ```

  Probes and `/metrics` are deliberately left open: an orchestrator probe and a
  Prometheus scrape do not send custom headers by default, so a key on `/health`
  would turn a misconfigured secret into a restart loop and one on `/metrics`
  into a silently dead dashboard. Neither carries PHI. Restricting them stays
  the proxy's job.
- **Do not expose `/metrics` or `/info` publicly.** Neither carries PHI —
  `/info` reports the dictionary by file name, not path, for this reason — but
  both describe the deployment and neither is authenticated.
- **Set a body-size ceiling at the proxy.** `UMLSMATCH_MAX_BODY_BYTES`
  (default 16 MiB) is enforced in-process two ways: a `Content-Length` over the
  cap is refused before a byte is read, and a request that declares no length —
  `Transfer-Encoding: chunked` — is counted as it arrives and aborted with a
  `413` on the chunk that crosses the cap, before the handler sees it. Neither
  makes the proxy optional: it sees the request first and can reject it without
  occupying a worker at all. The in-process checks are the backstop for a direct
  client.
- **Decide where `UMLSMATCH_LOG_EXCEPTION_DETAIL` should sit — the default is
  the permissive one.** It ships `true`, so a failure inside spaCy or sqlite3
  can put the text that broke it into your application log. That is defensible
  when logs stay on the host, and wrong the moment they ship to an aggregator
  with a wider audience than the API. It is the one setting here whose default
  you should confirm rather than inherit; see
  [PHI and egress](#phi-and-egress) for what turning it off keeps.

  The default is deliberately *not* the safe one, because changing it would
  silently tighten every existing deployment on upgrade and the failure mode of
  that — diagnostics quietly disappearing from a log an operator is mid-incident
  with — is worse than the one it prevents. That trade only holds if the setting
  is visible, which is why it is on this list and reported by `/info`.
- **Rate limiting is the proxy's job too.** There is none here.

The defaults are chosen so that *failing to do the above* is visible rather
than silent: ports bind to loopback unless `UMLSMATCH_BIND` is set deliberately,
so an unconfigured `docker compose up` is unreachable from the network instead
of being unreachable-looking while open.

**Before exposing this to anything that matters**, confirm the live
configuration from outside the container rather than from the manifest that was
supposed to produce it:

```bash
curl -s 127.0.0.1:8000/info | python -m json.tool
```

| field | want | why |
|---|---|---|
| `api_key_required` | `true` | the backstop is on, not just configured somewhere |
| `log_exception_detail` | `false` if logs leave the host | otherwise treat the log store as PHI |
| `profile` | whatever you measured against | two instances can answer the same note differently and both be right |
| `dictionary` | the build you expect | a stale volume mount is silent otherwise |

### PHI and egress

- **Nothing logs note text.** Handlers log document counts, character totals and
  durations. Metrics carry no label derived from content — `/metrics` is usually
  scraped by systems with broader access than the API.
- **Batch errors return the exception *type* only** — never the message, in any
  configuration. Our own raises never quote note content, but an exception
  thrown inside spaCy or sqlite3 is entitled to quote the input that broke it.
- **By default that message still reaches the log, so treat application logs as
  PHI.** This is the right trade when logs stay on the host and are one thing to
  audit, and the wrong one when they ship to an aggregator with a wider audience
  than the API itself. Set `UMLSMATCH_LOG_EXCEPTION_DETAIL=false` for the latter:
  a failure is then logged as its exception type and call path (file, line,
  function) with every message dropped, including those of chained causes, which
  keeps it diagnosable without keeping it quotable. `/info` reports
  `log_exception_detail` so the live setting can be confirmed from outside the
  container.
- **Validation errors do not echo the body.** FastAPI's default handler puts the
  offending value in an `input` key; the service replaces it with one that keeps
  `type`, `loc` and `msg` and drops `input`. Without that, a client sending
  `text` as the wrong JSON type gets note content back in a `422`.
- **Access logs still record paths and timings.** Cap retention; the compose
  file does.
- **The service makes no outbound calls.** The dictionary is a local file and
  the spaCy model is baked into the image at build time.

`internal: true` is deliberately *not* used on the compose network: it removes
the gateway and so breaks published ports, which would look like egress
protection while actually disabling ingress. Enforce egress at a layer that can
block it while still accepting traffic — a host firewall rule on the container
subnet, or a Kubernetes NetworkPolicy with an egress allow-list — and verify it:

```bash
docker compose exec api python -c \
  "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
```

should fail once the policy is in place.

### Sizing

Start from throughput, not cores. At ~14 docs/sec per replica with
`POOL_SIZE=2`, a target of 100 docs/sec needs ~8 replicas and ~3.6 GB. Confirm
against your own note lengths — the corpus above averages 5,558 characters, and
throughput tracks characters far more closely than documents.

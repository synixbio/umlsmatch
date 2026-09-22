#!/usr/bin/env python3
"""Load-test a running umlsmatch service and report throughput and latency.

Measures the throughput target README's Validation section leaves to you to set;
clinical document length varies enough that a number from another corpus does not
transfer.

Deliberately dependency-free (stdlib ``http.client`` + threads) so it runs
against a deployed service from anywhere, including a host inside an
egress-blocked network that has nothing installed.

**Each worker holds one keep-alive connection.** A fresh connection per request
measures connection setup, not the server: on Windows, ``localhost`` resolves to
``::1`` first and the IPv4 fallback costs ~2 s per connection, which reads as
0.5 docs/sec against a service whose own timing says 3 ms. Prefer ``127.0.0.1``
over ``localhost`` anyway; the script warns if you do not.

Reports the **p95 and p99**, not just the mean: a pipeline pool sheds load by
returning 503 past ``lease_timeout``, so the interesting failure is a tail that
grows while the mean stays flat. A mean-only report hides exactly the behaviour
you are load-testing for.

Usage::

    # send the corpus you already have, at 8 concurrent connections
    python tools/load_test.py --url http://localhost:8000 \\
        --input-dir free_texts/synthetic --concurrency 8

    # or synthetic text, no corpus needed
    python tools/load_test.py --url http://localhost:8000 --requests 200

PHI note: this reads note files and sends them to the service. It prints
timings and status codes only -- never document text, and never annotations.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.corpus import iter_text_files, read_text

_SYNTHETIC = (
    "Patient denies chest pain, shortness of breath, or palpitations. "
    "History of CHF and type 2 diabetes mellitus. Takes aspirin 81 mg daily "
    "and metformin 500 mg twice daily. No known drug allergies. "
    "Physical exam unremarkable. Chest x-ray showed no acute infiltrate."
)


def _post(
    conn: http.client.HTTPConnection, path: str, payload: dict
) -> tuple[int, float, int]:
    """Return (status, seconds, n_annotations). Never returns body text."""
    body = json.dumps(payload).encode()
    started = time.perf_counter()
    try:
        conn.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        raw = resp.read()  # must drain before the connection is reusable
        elapsed = time.perf_counter() - started
        if resp.status != 200:
            return resp.status, elapsed, 0
        return 200, elapsed, json.loads(raw).get("n_annotations", 0)
    except Exception:
        # A dead keep-alive connection cannot be reused; the caller rebuilds it.
        conn.close()
        return 0, time.perf_counter() - started, 0


def run(
    url: str, documents: list[str], concurrency: int, timeout: float
) -> tuple[list[float], Counter, int]:
    parsed = urllib.parse.urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    path = (parsed.path.rstrip("/") or "") + "/analyze"

    pending = list(documents)
    lock = threading.Lock()
    latencies: list[float] = []
    statuses: Counter = Counter()
    annotations = 0

    def worker() -> None:
        nonlocal annotations
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            while True:
                with lock:
                    if not pending:
                        return
                    text = pending.pop()
                status, elapsed, n = _post(conn, path, {"text": text})
                if status == 0:  # connection died; replace it for the next one
                    conn = http.client.HTTPConnection(host, port, timeout=timeout)
                with lock:
                    latencies.append(elapsed)
                    statuses[status] += 1
                    annotations += n
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return latencies, statuses, annotations


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://localhost:8000", help="service base URL")
    ap.add_argument("--input-dir", type=Path, default=None, help="folder of .txt notes")
    ap.add_argument(
        "--requests", type=int, default=100,
        help="synthetic requests when --input-dir is absent (default: 100)",
    )
    ap.add_argument("--concurrency", type=int, default=4, help="parallel connections")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    ap.add_argument(
        "--warmup", type=int, default=2,
        help="requests to discard before measuring (default: 2)",
    )
    args = ap.parse_args()

    if args.input_dir:
        files = list(iter_text_files(args.input_dir))
        if not files:
            sys.exit(f"error: no note files under {args.input_dir}")
        documents = [read_text(f) for f in files]
        source = f"{len(documents)} notes from {args.input_dir}"
    else:
        documents = [_SYNTHETIC] * args.requests
        source = f"{len(documents)} synthetic requests"

    chars = sum(len(d) for d in documents)
    print(f"{source}, {chars:,} characters, concurrency {args.concurrency}")

    if urllib.parse.urlparse(args.url).hostname == "localhost":
        # This exact mistake produced a 0.5 docs/sec reading against a service
        # whose own timing said 3 ms. Loud, because the resulting number looks
        # plausible rather than broken.
        print(
            "  warning: prefer 127.0.0.1 over 'localhost' -- on Windows it "
            "resolves to ::1 first\n"
            "           and the IPv4 fallback can add ~2s to each new "
            "connection, which\n"
            "           this script would report as service latency.",
            file=sys.stderr,
        )

    if args.warmup:
        # The first requests pay page-cache and lazy-import costs that no
        # steady-state number should include.
        run(args.url, documents[: args.warmup], 1, args.timeout)

    started = time.perf_counter()
    latencies, statuses, annotations = run(
        args.url, documents, args.concurrency, args.timeout
    )
    wall = time.perf_counter() - started

    ok = statuses.get(200, 0)
    print(f"\n{ok}/{len(documents)} succeeded in {wall:.1f}s")
    if statuses.keys() - {200}:
        print("  non-200:", ", ".join(f"{c}x{n}" for c, n in sorted(statuses.items()) if c != 200))
    if not ok:
        sys.exit("error: no successful requests")

    ordered = sorted(latencies)
    print(f"\n  throughput   {ok / wall:6.1f} docs/sec")
    print(f"  chars/sec    {chars / wall:,.0f}")
    print(f"  annotations  {annotations:,}")
    print("\n  latency (s)")
    print(f"    mean       {statistics.fmean(ordered):6.3f}")
    print(f"    median     {statistics.median(ordered):6.3f}")
    print(f"    p95        {ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]:6.3f}")
    print(f"    p99        {ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)]:6.3f}")
    print(f"    max        {ordered[-1]:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

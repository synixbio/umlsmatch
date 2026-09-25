#!/usr/bin/env python3
"""Scale across CPU cores with multiprocessing.

    python examples/parse_to_jsonl_batch.py free_texts/synthetic --workers 4

With no output path the run saves itself, the same way `python -m umlsmatch`
does: a fresh directory under out/runs/ holding annotations.jsonl and a
manifest. Pass a path to choose the destination instead::

    python examples/parse_to_jsonl_batch.py free_texts/synthetic out/parallel.jsonl

A single pipeline runs at roughly 12-13 notes/sec and is **not thread-safe**, so
threads don't help — but processes do, since the work is CPU-bound.

Two details that matter:

  * **Each worker builds its own pipeline once**, in an initializer, not per
    document. A pipeline costs seconds to construct; per-document construction
    would be slower than running single-process.
  * **The `if __name__ == "__main__"` guard is required**, not stylistic. On
    Windows (spawn start method) its absence makes every child re-import and
    re-execute this module, forking recursively.

Output is byte-comparable with `parse_to_jsonl.py`'s, and follows the same three
conventions as every exporter in this directory: `source` is the note's **file
name**, a note with no findings gets **no record**, a note that fails is
reported and skipped, and an attribute this pipeline never assesses gets no key
(`--all-attributes` keeps them). Only the ordering differs -- results arrive as
workers finish them, so records are not in corpus order.

Each worker holds its own model and SQLite connection. Measured on a real-note
corpus: ~210 MB resident per worker once warm, so four workers need ~0.9 GB.
That figure is a property of the model and dictionary, not of the corpus, so
the 20-note corpus now in `free_texts/synthetic/` does not change it -- but it
is too small to observe the growth described next.

Expect that to climb on a long run rather than sit flat. Two caches grow with
the distinct vocabulary a worker has seen: SQLite's page cache, capped at
~195 MB per connection by the matcher's `PRAGMA cache_size`, and the matcher's
candidate memo, which has no eviction. Budget nearer ~400 MB per worker for a
long-lived process; size `--workers` against RAM as well as cores.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from umlsmatch.corpus import (
    document_label,
    duplicate_label_warning,
    iter_text_files,
    read_text,
)
from umlsmatch.runs import DEFAULT_RUN_ROOT, RunWriter, manifest_stamp

# One pipeline per worker process, created in _init_worker.
_PIPELINE = None
#: Attribute keys this worker's pipeline never populates, dropped on the way
#: out. Decided in the worker because that is where the pipeline is: the parent
#: never builds one, and asking it to would load spaCy in a process that only
#: writes JSON. Every worker is configured identically, so they all agree.
_SKIP: frozenset[str] = frozenset()


def _init_worker(db_path: str | None, all_attributes: bool) -> None:
    global _PIPELINE, _SKIP
    # Imported inside the worker, not at module scope: under the spawn start
    # method each child re-imports this module, and deferring the heavy import
    # keeps the parent process from loading spaCy it will never use.
    from umlsmatch import ClinicalPipeline
    from umlsmatch.assertion.attributes import attribute_names

    _PIPELINE = ClinicalPipeline(db_path)
    _SKIP = (frozenset() if all_attributes
             else frozenset(attribute_names()) - _PIPELINE.assessed_attributes)


def _analyze_one(path_str: str) -> tuple[str, list[dict], str | None]:
    """Run in a worker. Returns (path, annotations, error)."""
    try:
        text = read_text(path_str)
        return path_str, [
            {k: v for k, v in a.to_dict().items() if k not in _SKIP}
            for a in _PIPELINE.analyze(text)
        ], None
    except Exception as exc:  # keep one bad document from killing the pool
        return path_str, [], f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path)
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=f"where to write. Omit to save a run under {DEFAULT_RUN_ROOT}.",
    )
    ap.add_argument("--db", default=None)
    ap.add_argument("--run-id", default=None, help="name this run's directory")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="processes (default: half the cores; budget ~400 MB each)",
    )
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument(
        "--all-attributes",
        action="store_true",
        help="keep every assertion attribute key, including ones this pipeline "
             "does not assess (they are null on every annotation)",
    )
    args = ap.parse_args()

    if args.run_id and args.output:
        ap.error("--run-id has nothing to name when an output path is given")

    files = list(iter_text_files(args.input_dir))
    if not files:
        print(f"no .txt files under {args.input_dir}", file=sys.stderr)
        return 1

    # Documents are keyed by file name, so two notes with the same name in
    # different subdirectories would be indistinguishable in the output.
    collision = duplicate_label_warning(files)
    if collision:
        print(collision, file=sys.stderr)

    # A run directory when no destination was named; an explicit path wins.
    run = None
    if args.output is None:
        try:
            run = RunWriter(DEFAULT_RUN_ROOT, run_id=args.run_id)
        except FileExistsError:
            print(f"run directory already exists: {args.run_id}", file=sys.stderr)
            return 2
        args.output = run.artifact("annotations.jsonl")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} documents across {args.workers} worker(s)", file=sys.stderr)

    started = time.time()
    total = empty = failed = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.db, args.all_attributes),
    ) as pool, args.output.open("w", encoding="utf-8") as out:
        results = pool.imap_unordered(
            _analyze_one, (str(f) for f in files), chunksize=args.chunksize
        )
        for i, (path_str, annotations, error) in enumerate(results, 1):
            # Progress ticks on every result, empty or not, so a stretch of
            # notes with no findings does not look like a stall.
            if i % 10 == 0 or i == len(files):
                print(
                    f"  [{i}/{len(files)}] {i/(time.time()-started):.1f} docs/sec",
                    file=sys.stderr,
                )
            if error:
                failed += 1
                print(f"  FAILED {Path(path_str).name}: {error}", file=sys.stderr)
                continue
            # A note with no findings gets no record, matching every other
            # exporter here.
            if not annotations:
                empty += 1
                continue
            out.write(
                json.dumps(
                    {
                        "source": document_label(path_str),
                        "n_annotations": len(annotations),
                        "annotations": annotations,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            total += len(annotations)

    elapsed = time.time() - started
    n_ok = len(files) - failed

    if run is not None:
        run.record_artifact(
            args.output, documents=n_ok, annotations=total, empty=empty, failed=failed
        )
        run.write_manifest(
            **manifest_stamp(
                script=Path(__file__).name,
                input_dir=str(args.input_dir),
                workers=args.workers,
            ),
            totals={"documents": n_ok, "annotations": total},
        )

    print(
        f"\n{n_ok}/{len(files)} documents -> {args.output}\n"
        # Rate over documents actually analyzed, not attempted -- see 02.
        f"{total:,} annotations in {elapsed:.1f}s ({n_ok/elapsed:.1f} docs/sec)"
        + (f"\n{empty} documents had no annotations and are not in the output"
           if empty else "")
        + (f"\n{failed} failed" if failed else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    # Required on Windows/macOS spawn. See module docstring.
    raise SystemExit(main())

#!/usr/bin/env python3
"""Process a directory of notes into JSONL — one record per document.

    python examples/parse_to_jsonl.py free_texts/synthetic

With no output path the run saves itself, the same way `python -m umlsmatch`
does: a fresh directory under out/runs/ holding annotations.jsonl and a
manifest of the settings that produced it. Nothing overwrites a previous run.
Pass a path to choose the destination instead::

    python examples/parse_to_jsonl.py free_texts/synthetic out/annotations.jsonl

JSONL suits this well: it streams, survives a crash mid-run (everything already
written stays valid), and loads straight into pandas or jq.

Shows the three things any real batch job needs: reuse one pipeline, don't let
one bad document kill the run, and write incrementally rather than buffering
everything in memory.

Three conventions are shared with `parse_to_csv.py`, `parse_to_parquet.py` and
`parse_to_sqlite.py`, so that exports of one corpus in different formats hold
the same rows and join on the same key:

  * `source` is the note's **file name** (`umlsmatch.corpus.document_label`),
    not its path. A path would record where the corpus happened to be mounted,
    which differs between machines and stops two exports of one corpus joining.
  * A note the pipeline finds nothing in gets **no record**; it is counted and
    reported, not written. This format *could* carry an empty record, and did
    before -- but a reader then has to decide whether an empty JSONL record and
    an absent CSV row mean the same thing, and the answer has to be yes.
  * A note that fails analysis is reported and **skipped**, not fatal.
  * An attribute this pipeline never assesses gets **no key**. A default
    pipeline leaves `conditional` and `generic` unassessed, and `"conditional":
    null` on every annotation states a fact about the pipeline, not about the
    mention. The keys come from `ClinicalPipeline.assessed_attributes`, so
    building with `conditional=True` brings that one back; `--all-attributes`
    keeps them all, for a consumer whose schema must not change between runs.

NOTE: output contains verbatim note text in the `text` field of each annotation.
Treat the output file with the same access controls as the source notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from umlsmatch import ClinicalPipeline
from umlsmatch.assertion.attributes import attribute_names
from umlsmatch.corpus import (
    document_label,
    duplicate_label_warning,
    iter_text_files,
    parse_groups,
    read_text,
)
from umlsmatch.runs import DEFAULT_RUN_ROOT, RunWriter, manifest_stamp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path)
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=f"where to write. Omit to save a run under {DEFAULT_RUN_ROOT}.",
    )
    ap.add_argument("--db", default=None, help="dictionary path (default: auto)")
    ap.add_argument("--groups", default=None, help="e.g. DISORDER,DRUG")
    ap.add_argument("--run-id", default=None, help="name this run's directory")
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

    groups = parse_groups(args.groups)

    # Documents are keyed by file name, so two notes with the same name in
    # different subdirectories would be indistinguishable in the output.
    collision = duplicate_label_warning(files)
    if collision:
        print(collision, file=sys.stderr)

    # A run directory when no destination was named, so a batch job keeps its
    # own output instead of overwriting the last one. An explicit path still
    # wins -- this is a default, not a policy.
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

    started = time.time()
    n_annotations = 0
    n_empty = 0
    n_failed = 0

    with ClinicalPipeline(args.db, groups=groups) as nlp, args.output.open(
        "w", encoding="utf-8"
    ) as out:
        # Which keys an annotation carries is a property of the pipeline, not
        # of the data -- see ClinicalPipeline.assessed_attributes.
        skip = (set() if args.all_attributes
                else set(attribute_names()) - nlp.assessed_attributes)
        if skip:
            print(f"not assessed by this pipeline, so not written: "
                  f"{', '.join(sorted(skip))}", file=sys.stderr)

        for i, path in enumerate(files, 1):
            try:
                text = read_text(path)
                annotations = nlp.analyze(text)
            except Exception as exc:
                # One malformed document shouldn't abort a long corpus run.
                n_failed += 1
                print(f"  [{i}/{len(files)}] FAILED {path.name}: {exc}", file=sys.stderr)
                continue

            # Progress before the empty check, so a stretch of notes with no
            # findings still ticks rather than looking like a stall.
            if i % 10 == 0 or i == len(files):
                rate = i / (time.time() - started)
                print(f"  [{i}/{len(files)}] {rate:.1f} docs/sec", file=sys.stderr)

            # A note with no findings gets no record, matching the tabular
            # exports, which have no way to represent one.
            if not annotations:
                n_empty += 1
                continue

            out.write(
                json.dumps(
                    {
                        "source": document_label(path),
                        "n_annotations": len(annotations),
                        "annotations": [
                            {k: v for k, v in a.to_dict().items() if k not in skip}
                            for a in annotations
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_annotations += len(annotations)

    elapsed = time.time() - started
    n_ok = len(files) - n_failed

    if run is not None:
        run.record_artifact(
            args.output,
            documents=n_ok,
            annotations=n_annotations,
            empty=n_empty,
            failed=n_failed,
        )
        run.write_manifest(
            **manifest_stamp(
                script=Path(__file__).name,
                input_dir=str(args.input_dir),
                filters={"groups": sorted(groups) if groups else None},
            ),
            totals={"documents": n_ok, "annotations": n_annotations},
        )

    print(
        f"\n{n_ok}/{len(files)} documents -> {args.output}\n"
        # Rate over documents actually analyzed: a run where half the corpus
        # failed fast would otherwise report a flatteringly high throughput.
        f"{n_annotations:,} annotations in {elapsed:.1f}s "
        f"({n_ok/elapsed:.1f} docs/sec)"
        # Both counts are reported, never silently absorbed: a corpus where a
        # third of the notes produced nothing is a finding about the corpus.
        + (f"\n{n_empty} documents had no annotations and are not in the output"
           if n_empty else "")
        + (f"\n{n_failed} failed" if n_failed else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

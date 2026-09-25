#!/usr/bin/env python3
"""Flatten annotations into a CSV — one row per annotation.

    python examples/parse_to_csv.py free_texts/synthetic

With no output path the run saves itself, the same way `python -m umlsmatch`
does: a fresh directory under out/runs/ holding annotations.csv and a manifest
of the settings that produced it. Nothing overwrites a previous run. Pass a
path to choose the destination instead::

    python examples/parse_to_csv.py free_texts/synthetic out/annotations.csv

Useful when the next step is pandas, R, Excel, or a database load. Uses only the
standard library's `csv` module, so no pandas dependency.

Three groups of columns deserve comment:

  * `negated` is written as 0/1 rather than True/False, because spreadsheet tools
    handle it more predictably.
  * The other assertion attributes are written as 0/1 too, except that an
    attribute the pipeline does not assess is written **empty**, not 0. CSV has
    no null, and an empty cell is the closest honest spelling of "not
    assessed" -- writing 0 would claim an assessment that never happened. See
    `umlsmatch.assertion.attributes` for which are which.
  * **An attribute the pipeline never assesses gets no column at all.** A
    default pipeline leaves `conditional` and `generic` unassessed, so a column
    for either would be empty in every row -- and an empty cell means "not
    assessed", which is a fact about the pipeline, not about the mention, so
    repeating it 16,000 times says nothing. The columns are chosen from
    `ClinicalPipeline.assessed_attributes`, so building the pipeline with
    `conditional=True` brings that column back without editing this list.
    `--all-attributes` keeps every column regardless, for a downstream schema
    that must not change shape between runs.
  * `text` is the verbatim note span, so **this file contains PHI**. If you only
    need counts, pass --no-text.

Three conventions are shared with `parse_to_jsonl.py`, `parse_to_parquet.py` and
`parse_to_sqlite.py`, so that exports of one corpus in different formats hold
the same rows and join on the same key:

  * `document` is the note's **file name** (`umlsmatch.corpus.document_label`).
  * A note the pipeline finds nothing in contributes **no rows**; it is counted
    and reported, not written.
  * A note that fails analysis is reported and **skipped**, not fatal.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

COLUMNS = [
    "document",
    "cui",
    "preferred_text",
    # The dictionary entry that matched, in tokenizer spelling -- kept because
    # it is what explains a surprising match, and dropping it here while the
    # JSONL and SQLite exports carry it made the same corpus look different
    # depending on which script wrote it.
    "term",
    "group",
    "negated",
    "subject",
    "history_of",
    "uncertain",
    "conditional",
    "generic",
    "start",
    "end",
    "text",
]

#: Attributes written as 0/1/empty. See the module docstring.
_NULLABLE_FLAGS = ("history_of", "uncertain", "conditional", "generic")


def _flag(value: bool | None) -> str:
    """0/1 for an assessed attribute, empty for an unassessed one."""
    return "" if value is None else str(int(value))


def _columns(nlp: ClinicalPipeline, *, no_text: bool, all_attributes: bool) -> list[str]:
    """The header for this run: :data:`COLUMNS` minus what would be empty.

    Asked of the pipeline rather than of the data. The alternative -- write
    every column, then drop the ones that came out empty -- cannot be done
    while streaming, and would also be wrong: a column is dropped here because
    nothing assessed it, not because no mention happened to have it.
    """
    skip = set() if all_attributes else set(attribute_names()) - nlp.assessed_attributes
    if no_text:
        skip.add("text")
    return [c for c in COLUMNS if c not in skip]


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
    ap.add_argument("--groups", default=None, help="e.g. DISORDER,DRUG")
    ap.add_argument("--run-id", default=None, help="name this run's directory")
    ap.add_argument(
        "--no-text",
        action="store_true",
        help="omit the verbatim matched text column (de-identifies the export)",
    )
    ap.add_argument(
        "--all-attributes",
        action="store_true",
        help="keep every assertion attribute column, including ones this "
             "pipeline does not assess (they are empty in every row)",
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
    # different subdirectories would merge into one `document` silently.
    collision = duplicate_label_warning(files)
    if collision:
        print(collision, file=sys.stderr)

    # A run directory when no destination was named, so an export keeps its own
    # output instead of overwriting the last one. An explicit path still wins.
    run = None
    if args.output is None:
        try:
            run = RunWriter(DEFAULT_RUN_ROOT, run_id=args.run_id)
        except FileExistsError:
            print(f"run directory already exists: {args.run_id}", file=sys.stderr)
            return 2
        args.output = run.artifact("annotations.csv")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    n_empty = 0
    n_failed = 0

    # The pipeline is built before the header is written, because it is what
    # decides the header: a column exists for an attribute this pipeline
    # assesses, and not for one it leaves None on every row.
    with ClinicalPipeline(args.db, groups=groups) as nlp:
        columns = _columns(nlp, no_text=args.no_text, all_attributes=args.all_attributes)
        dropped = [c for c in COLUMNS if c not in columns and c != "text"]
        if dropped:
            print(f"not assessed by this pipeline, so not written: {', '.join(dropped)}",
                  file=sys.stderr)

        with args.output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()

            for i, path in enumerate(files, 1):
                try:
                    text = read_text(path)
                    annotations = nlp.analyze(text)
                except Exception as exc:
                    # One malformed document shouldn't abort a long corpus run.
                    n_failed += 1
                    print(f"  [{i}/{len(files)}] FAILED {path.name}: {exc}",
                          file=sys.stderr)
                    continue

                # A note with no findings contributes no rows. Counted rather
                # than written, because a row-per-annotation table has no way to
                # say "this document, and nothing in it".
                if not annotations:
                    n_empty += 1
                    continue

                for a in annotations:
                    row = {
                        "document": document_label(path),
                        "cui": a.cui,
                        "preferred_text": a.preferred_text,
                        "term": a.term,
                        "group": a.group,
                        "negated": int(a.negated),
                        "subject": a.subject or "",
                        "start": a.start,
                        "end": a.end,
                        "text": a.text,
                        **{f: _flag(getattr(a, f)) for f in _NULLABLE_FLAGS},
                    }
                    writer.writerow({c: row[c] for c in columns})
                    rows += 1

    n_ok = len(files) - n_failed

    if run is not None:
        run.record_artifact(
            args.output, rows=rows, documents=n_ok, empty=n_empty, failed=n_failed
        )
        run.write_manifest(
            **manifest_stamp(
                script=Path(__file__).name,
                input_dir=str(args.input_dir),
                filters={"groups": sorted(groups) if groups else None},
                columns=columns,
            ),
            totals={"documents": n_ok, "rows": rows},
        )

    print(
        f"{rows:,} rows from {n_ok} documents -> {args.output}"
        # Both counts are reported, never silently absorbed: a corpus where a
        # third of the notes produced nothing is a finding about the corpus.
        + (f"\n{n_empty} documents had no annotations and are not in the output"
           if n_empty else "")
        + (f"\n{n_failed} failed" if n_failed else ""),
        file=sys.stderr,
    )
    if not args.no_text:
        print("NOTE: the `text` column contains verbatim note content (PHI).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

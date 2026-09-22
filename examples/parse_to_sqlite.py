#!/usr/bin/env python3
"""Analyze a corpus straight into a queryable SQLite database.

    python examples/parse_to_sqlite.py free_texts/synthetic

With no output path the run saves itself, the same way `python -m umlsmatch`
does: a fresh directory under out/runs/ holding annotations.db and a manifest of
the settings that produced it. Nothing overwrites a previous run. Pass a path to
choose the destination instead::

    python examples/parse_to_sqlite.py free_texts/synthetic out/annotations.db

This is the one-step version of **analyze -> export -> load**: where
`parse_to_csv.py` writes a file that `load_to_sqlite.py` then reads back, this
inserts each document's annotations as they are produced and never writes the
intermediate export at all. Either route ends at the same database -- and that
is not a coincidence, because this script imports its schema from
`load_to_sqlite.py` rather than restating it. Two copies of a CREATE TABLE are
two things to keep in step, and `sqlite_browser.py` queries whichever one it is
handed by name.

Take the two-step route instead when the export is itself a deliverable (a CSV
someone reviews in Excel, a JSONL another team consumes), when the same
annotations must be loaded more than once without re-running the pipeline, or
when the corpus is large enough that you want the expensive analysis pass and
the cheap load pass to be separately restartable. Take this one when the
database is the only thing you actually wanted.

Four conventions are shared with `parse_to_csv.py`, `parse_to_jsonl.py` and
`parse_to_parquet.py`, so that exports of one corpus in different formats hold
the same rows and join on the same key:

  * `documents.source` is the note's **file name**
    (`umlsmatch.corpus.document_label`), which is also what a CSV or Parquet
    export puts in its `document` column -- so a table loaded from one can be
    joined against a database built by the other.
  * A note the pipeline finds nothing in gets **no `documents` row**; it is
    counted and reported, not written. This schema *could* hold one, unlike the
    tabular formats -- and did before -- but then `COUNT(*) FROM documents`
    would mean something different here than everywhere else.
  * A note that fails analysis is reported and **skipped**, not fatal.
  * An attribute the pipeline never assesses gets **no column**. A default
    pipeline leaves `conditional` and `generic` unassessed, so this database
    has neither column -- the columns come from
    `ClinicalPipeline.assessed_attributes`, and `--all-attributes` keeps them
    all. `load_to_sqlite.py` builds the same table from the same definition and
    keeps every column, because its other input is a **Java cTAKES silver
    standard, which does assess all six**; its loader reads the column list off
    the table, so `--append` still works in both directions and says so when a
    value has nowhere to go.

NOTE: the `text` column holds verbatim note content, and `documents.source`
holds note file names, which in this corpus carry patient identifiers. **The
.db file is PHI.** `--no-text` drops the verbatim column if you only need
counts.
"""

from __future__ import annotations

import argparse
import sqlite3
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

# The schema, the indexes and the batching insert live in load_to_sqlite.py and
# are imported rather than copied: a database this script writes and one that
# script writes have to be the same database, or sqlite_browser.py's reports
# work against one and not the other. examples/ is not a package, so the
# directory goes on the path first -- the same sys.path-then-import shape the
# tools/ scripts use, and the reason the imports below are not at the top.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_to_sqlite import Loader, indexes, schema


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
        "--append",
        action="store_true",
        help="add to an existing database instead of replacing it",
    )
    ap.add_argument(
        "--no-text",
        action="store_true",
        help="leave the verbatim matched text column NULL",
    )
    ap.add_argument(
        "--all-attributes",
        action="store_true",
        help="keep every assertion attribute column, including ones this "
             "pipeline does not assess (they are NULL in every row)",
    )
    args = ap.parse_args()

    if args.run_id and args.output:
        ap.error("--run-id has nothing to name when an output path is given")
    # A run directory is created fresh by definition, so there is nothing there
    # to append to. Silently ignoring the flag would leave someone believing
    # they had accumulated two corpora in one database.
    if args.append and args.output is None:
        ap.error("--append needs an existing database path; a new run directory has none")

    files = list(iter_text_files(args.input_dir))
    if not files:
        print(f"no .txt files under {args.input_dir}", file=sys.stderr)
        return 1

    groups = parse_groups(args.groups)

    # Documents are keyed by file name, so two notes with the same name in
    # different subdirectories would collapse into one `documents` row.
    collision = duplicate_label_warning(files)
    if collision:
        print(collision, file=sys.stderr)

    # A run directory when no destination was named, so a run keeps its own
    # output instead of overwriting the last one. An explicit path still wins.
    run = None
    if args.output is None:
        try:
            run = RunWriter(DEFAULT_RUN_ROOT, run_id=args.run_id)
        except FileExistsError:
            print(f"run directory already exists: {args.run_id}", file=sys.stderr)
            return 2
        args.output = run.artifact("annotations.db")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.append:
            args.output.unlink()

    started = time.time()
    n_empty = 0
    n_failed = 0

    # The pipeline is built before the table, because it is what decides the
    # table: a column exists for an attribute this pipeline assesses, and not
    # for one it would leave NULL in every row.
    with ClinicalPipeline(args.db, groups=groups) as nlp:
        omit = (() if args.all_attributes
                else sorted(set(attribute_names()) - nlp.assessed_attributes))
        if omit:
            print(f"not assessed by this pipeline, so no column: {', '.join(omit)}",
                  file=sys.stderr)

        conn = sqlite3.connect(args.output)
        # Durability bought back by the run directory: this database is derived
        # output, reproducible by re-running, so a crash mid-run costs a re-run
        # rather than data. Both pragmas are worth roughly a factor on inserts.
        conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
        # A no-op under --append, where the existing table's shape wins. The
        # loader reads its columns off the table for exactly that reason, and
        # reports anything it could not store.
        conn.executescript(schema(omit))
        # basename=False because the label is already a file name: passing the
        # flag as well would re-derive it from something that no longer has a
        # directory, which works but hides where the decision is made.
        loader = Loader(conn)

        for i, path in enumerate(files, 1):
            source = document_label(path)
            try:
                text = read_text(path)
                annotations = nlp.analyze(text)
            except Exception as exc:
                # One malformed document shouldn't abort a long corpus run.
                n_failed += 1
                print(f"  [{i}/{len(files)}] FAILED {path.name}: {exc}",
                      file=sys.stderr)
                continue

            # Progress before the empty check, so a stretch of notes with no
            # findings still ticks rather than looking like a stall.
            if i % 10 == 0 or i == len(files):
                rate = i / (time.time() - started)
                print(f"  [{i}/{len(files)}] {rate:.1f} docs/sec", file=sys.stderr)

            # A note with no findings gets no `documents` row: `Loader.doc_id`
            # is reached only from `add`, so not calling it is the whole
            # mechanism. Counted, and reported at the end.
            if not annotations:
                n_empty += 1
                continue

            for a in annotations:
                record = a.to_dict()
                if args.no_text:
                    record["text"] = None
                loader.add(source, record)

    loader.flush()

    # Denormalized count, maintained here so the common "how big is this
    # document" question needs no join. Every row gets a non-zero count now
    # that notes with no findings are not written at all -- the UPDATE stays
    # because --append can add to a database written before that was true.
    conn.execute(
        "UPDATE documents SET n_annotations = "
        "(SELECT COUNT(*) FROM annotations a WHERE a.doc_id = documents.doc_id)"
    )
    # Indexes after the bulk insert, not before -- building them once at the end
    # is materially faster than maintaining them per row.
    conn.executescript(indexes(loader.missing))
    conn.commit()

    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    anns = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    cuis = conn.execute("SELECT COUNT(DISTINCT cui) FROM annotations").fetchone()[0]
    negated = conn.execute("SELECT COUNT(*) FROM annotations WHERE negated = 1").fetchone()[0]
    conn.close()

    elapsed = time.time() - started
    n_ok = len(files) - n_failed

    if run is not None:
        run.record_artifact(
            args.output,
            documents=n_ok,
            annotations=loader.rows,
            empty=n_empty,
            failed=n_failed,
        )
        run.write_manifest(
            **manifest_stamp(
                script=Path(__file__).name,
                input_dir=str(args.input_dir),
                filters={"groups": sorted(groups) if groups else None},
                text_column=not args.no_text,
            ),
            totals={"documents": n_ok, "annotations": loader.rows},
        )

    print(f"\n  documents        {docs:,}")
    print(f"  annotations      {anns:,}")
    print(f"  distinct CUIs    {cuis:,}")
    if anns:
        print(f"  negated          {negated:,} ({100*negated/anns:.1f}%)")
    # Both counts are reported, never silently absorbed: a corpus where a third
    # of the notes produced nothing is a finding about the corpus.
    if n_empty:
        print(f"  no findings      {n_empty:,} (not written)")
    if n_failed:
        print(f"  failed           {n_failed:,}")
    # Rate over documents actually analyzed: a run where half the corpus failed
    # fast would otherwise report a flatteringly high throughput.
    print(f"\nwrote {args.output} in {elapsed:.1f}s ({n_ok/elapsed:.1f} docs/sec)")

    # An empty database is worth saying out loud rather than leaving to be
    # discovered by a query that returns nothing. Not fatal here, unlike in
    # load_to_sqlite.py: the documents were read and the run directory records
    # what was tried, so the file is evidence rather than a mistake.
    if anns == 0:
        print(
            "WARNING: no annotations were produced -- check --groups and the dictionary.",
            file=sys.stderr,
        )
    if not args.no_text:
        print("NOTE: the `text` column contains verbatim note content (PHI).")

    # `sqlite_browser.py` with no argument opens the newest database under
    # out/runs/, which -- for a run that saved itself -- is the one just
    # written. Spelled out rather than assumed.
    query_arg = "" if run is not None else f" {args.output}"
    print(f"\nQuery it:  python examples/sqlite_browser.py{query_arg}")
    print(f"       or:  sqlite3 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

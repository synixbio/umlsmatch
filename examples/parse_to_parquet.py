#!/usr/bin/env python3
"""Flatten annotations into a Parquet file — one row per annotation.

    python examples/parse_to_parquet.py free_texts/synthetic

The same table `parse_to_csv.py` writes, stored as Parquet instead: typed,
columnar and compressed. With no output path the run saves itself, the same way
`python -m umlsmatch` does: a fresh directory under out/runs/ holding
annotations.parquet and a manifest of the settings that produced it. Nothing
overwrites a previous run. Pass a path to choose the destination instead::

    python examples/parse_to_parquet.py free_texts/synthetic out/annotations.parquet

Needs pyarrow, which umlsmatch does not depend on -- `pip install pyarrow`. That
is the cost of this format; what it buys over the CSV is three things:

  * **Types survive the round trip.** `negated` is a boolean, not 0/1, and the
    offsets come back as integers without a reader guessing. Where the CSV has
    to write an unassessed attribute as an empty cell -- CSV has no null -- this
    writes an actual null, so "not assessed" and "assessed as false" stay
    different values rather than different spellings of a blank. See
    `umlsmatch.assertion.attributes` for which attributes are which.

    An attribute this pipeline never assesses gets **no column at all**: a
    default pipeline leaves `conditional` and `generic` unassessed, and an
    all-null column states a fact about the pipeline once per row. The columns
    come from `ClinicalPipeline.assessed_attributes`, so `conditional=True`
    brings that one back; `--all-attributes` keeps every column regardless, for
    a downstream schema that must not change shape between runs.
  * **Size.** Every column here is either low-cardinality (`cui`, `group`,
    `document`, the flags) or sorted-ish, which is the case Parquet's per-column
    dictionary encoding is built for. No Arrow-side dictionary type is needed to
    get it; the file format applies it to repeated string values itself.
  * **Column pruning.** `pandas.read_parquet(path, columns=["cui", "negated"])`
    reads two columns off disk instead of parsing every row, which is most of
    what a cohort count actually touches.

Provenance rides inside the file as well as beside it: the same stamp written to
the run manifest is stored in the Parquet key-value metadata, so a file that
gets copied out of its run directory still says what produced it::

    pq.ParquetFile(path).schema_arrow.metadata[b"umlsmatch"]

As with the CSV, `text` is the verbatim note span, so **this file contains
PHI**. If you only need counts, pass --no-text.

Three conventions are shared with `parse_to_csv.py`, `parse_to_jsonl.py` and
`parse_to_sqlite.py`, so that exports of one corpus in different formats hold
the same rows and join on the same key:

  * `document` is the note's **file name** (`umlsmatch.corpus.document_label`).
  * A note the pipeline finds nothing in contributes **no rows**; it is counted
    and reported, not written.
  * A note that fails analysis is reported and **skipped**, not fatal.
"""

from __future__ import annotations

import argparse
import json
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
    # The dictionary entry that matched, in tokenizer spelling. Low-cardinality
    # and highly repetitive, so it costs almost nothing here -- see the note on
    # dictionary encoding above.
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

#: Attributes the pipeline may leave unassessed, written as null when it does.
#: The CSV's 0/1/empty convention exists only because CSV cannot say null.
_NULLABLE_FLAGS = ("history_of", "uncertain", "conditional", "generic")

#: Rows buffered before a row group is flushed to disk.
#:
#: Parquet cannot be appended a row at a time the way CSV can -- a row group is
#: written whole, with its own column statistics -- so something has to be held
#: in memory. This is the size of that something, and it is also the unit a
#: reader skips when a filter rules the group out, which is why it is tens of
#: thousands of rows rather than either extreme: one row group per corpus would
#: buffer the entire run, and one per document would leave a large corpus with
#: thousands of tiny groups and a footer bigger than the data.
ROW_GROUP_ROWS = 50_000


def _load_pyarrow():
    """Import pyarrow, or exit saying how to get it.

    An optional dependency of one example script, not of the package: the core
    pipeline is standard library only and the CSV export stays that way, so the
    failure has to be a sentence rather than a traceback from an import at the
    top of the file.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "parquet output needs pyarrow, which umlsmatch does not depend on:\n"
            "    pip install pyarrow\n"
            f"(import failed: {exc})"
        ) from exc
    return pa, pq


def _schema(pa, columns: list[str]):
    """The Arrow schema for `columns`, in the order they were asked for.

    Declared rather than inferred from the first batch. Inference would give a
    corpus that produced no annotations a file with no columns -- valid Parquet
    that no downstream query can read -- and would type an all-null attribute
    column as null instead of boolean.
    """
    types = {
        "document": pa.string(),
        "cui": pa.string(),
        "preferred_text": pa.string(),
        "term": pa.string(),
        "group": pa.string(),
        "negated": pa.bool_(),
        "subject": pa.string(),
        "start": pa.int32(),
        "end": pa.int32(),
        "text": pa.string(),
        # Nullable by default, which is the point: null means the pipeline did
        # not assess this attribute, false means it assessed it and said no.
        **{f: pa.bool_() for f in _NULLABLE_FLAGS},
    }
    return pa.schema([(c, types[c]) for c in columns])


def _columns(nlp: ClinicalPipeline, *, no_text: bool, all_attributes: bool) -> list[str]:
    """The schema for this run: :data:`COLUMNS` minus what would be all-null.

    Asked of the pipeline rather than of the data. Parquet is written in row
    groups as the run proceeds, so the schema is fixed before any annotation
    exists -- and it should be: a column is dropped here because nothing
    assessed the attribute, not because no mention happened to have it.
    """
    skip = set() if all_attributes else set(attribute_names()) - nlp.assessed_attributes
    if no_text:
        skip.add("text")
    return [c for c in COLUMNS if c not in skip]


def _rows_buffered(buffer: dict[str, list], columns: list[str]) -> int:
    """How many rows are waiting in `buffer`.

    Every column holds the same number of values, so the first one answers it;
    it is a function rather than a counter because the buffer is cleared in
    place and a separate count is one more thing to keep in step with it.
    """
    return len(buffer[columns[0]])


def _human_size(n: int) -> str:
    """Bytes as kB or MB. Reported because size is half the reason to use this."""
    return f"{n / 1_000:.0f} kB" if n < 1_000_000 else f"{n / 1_000_000:.1f} MB"


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
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
        help=(
            "zstd compresses this table appreciably better than snappy at "
            "similar speed; choose snappy for a reader older than Parquet's "
            "zstd support (parquet-mr 1.12, Spark 3.2)."
        ),
    )
    ap.add_argument(
        "--no-text",
        action="store_true",
        help="omit the verbatim matched text column (de-identifies the export)",
    )
    ap.add_argument(
        "--all-attributes",
        action="store_true",
        help="keep every assertion attribute column, including ones this "
             "pipeline does not assess (they are null in every row)",
    )
    args = ap.parse_args()

    if args.run_id and args.output:
        ap.error("--run-id has nothing to name when an output path is given")

    pa, pq = _load_pyarrow()

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
        args.output = run.artifact("annotations.parquet")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    n_empty = 0
    n_failed = 0

    # The pipeline is built before the schema, because it is what decides the
    # schema: a column exists for an attribute this pipeline assesses, and not
    # for one it leaves null on every row.
    with ClinicalPipeline(args.db, groups=groups) as nlp:
        columns = _columns(nlp, no_text=args.no_text, all_attributes=args.all_attributes)
        dropped = [c for c in COLUMNS if c not in columns and c != "text"]
        if dropped:
            print(f"not assessed by this pipeline, so not written: {', '.join(dropped)}",
                  file=sys.stderr)

        # One stamp, used twice: embedded in the file and written to the
        # manifest. Generated once so the two cannot disagree about when the
        # run happened -- and after `columns`, which it records.
        stamp = manifest_stamp(
            script=Path(__file__).name,
            input_dir=str(args.input_dir),
            filters={"groups": sorted(groups) if groups else None},
            columns=columns,
        )
        schema = _schema(pa, columns).with_metadata({"umlsmatch": json.dumps(stamp)})
        buffer: dict[str, list] = {c: [] for c in columns}

        def flush(writer) -> None:
            """Write what has accumulated as one row group, and start the next."""
            if not _rows_buffered(buffer, columns):
                return
            writer.write_table(pa.Table.from_pydict(buffer, schema=schema))
            for column in buffer.values():
                column.clear()

        with pq.ParquetWriter(args.output, schema, compression=args.compression) as writer:
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
                        "negated": a.negated,
                        # None, not "": a mention whose subject was not assessed
                        # is null here, and null is not the empty string.
                        "subject": a.subject,
                        "start": a.start,
                        "end": a.end,
                        "text": a.text,
                        **{f: getattr(a, f) for f in _NULLABLE_FLAGS},
                    }
                    for c in columns:
                        buffer[c].append(row[c])
                    rows += 1

                # Checked between documents rather than between annotations: a
                # row group is allowed to overshoot slightly, and a document's
                # rows landing in one group keeps per-document statistics tight
                # enough for a reader filtering on `document` to skip groups.
                if _rows_buffered(buffer, columns) >= ROW_GROUP_ROWS:
                    flush(writer)

            flush(writer)

    size = args.output.stat().st_size
    n_ok = len(files) - n_failed

    if run is not None:
        run.record_artifact(
            args.output,
            rows=rows,
            documents=n_ok,
            empty=n_empty,
            failed=n_failed,
            compression=args.compression,
        )
        run.write_manifest(**stamp, totals={"documents": n_ok, "rows": rows})

    print(
        f"{rows:,} rows from {n_ok} documents -> {args.output} "
        f"({_human_size(size)}, {args.compression})"
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

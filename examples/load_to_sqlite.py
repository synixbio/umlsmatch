#!/usr/bin/env python3
"""Load exported annotations (CSV or JSONL) into a queryable SQLite database.

    python examples/load_to_sqlite.py out/annotations.csv   out/annotations.db
    python examples/load_to_sqlite.py out/annotations.jsonl out/annotations.db --append

Completes the workflow: analyze -> export -> load -> query (``sqlite_browser.py``).
``parse_to_sqlite.py`` does all four in one pass, and imports this module's
schema to do it; use that when the export is not itself wanted, and this when it
is -- or when the same export must be loaded more than once.

Format is detected from the extension, and JSONL records are further detected by
shape:

  * ``.csv``   -- from ``parse_to_csv.py``
  * ``.jsonl`` with an ``annotations`` key -- from ``parse_to_jsonl.py`` or
    ``parse_to_jsonl_batch.py`` (also ``python -m umlsmatch --json``)
  * ``.jsonl`` with a ``mentions`` key -- a **Java cTAKES silver standard** from
    ``tools/run_java_ctakes.py``. Loading this lets you query cTAKES' own output
    with the same SQL, or diff it against the Python pipeline's.

All six assertion attributes have columns, and the five non-polarity ones are
**nullable on purpose**: NULL means "not assessed", which is a different fact
from 0 ("assessed, and absent"). umlsmatch never assesses ``generic``, and
leaves ``conditional`` unassessed unless the pipeline was built with
``conditional=True`` (see ``umlsmatch.assertion.attributes``), so a default
load carries NULL there while a silver load carries cTAKES' 0/1 — and a query
that lumps the two together is comparing an absence of evidence with evidence
of absence. Use ``IS NULL`` / ``IS NOT NULL`` rather than ``= 0``, and note that
``SUM(col)`` skips NULLs while ``COUNT(*)`` does not. One mention with several
concepts expands to one row per concept.

``term`` is **always NULL for silver loads**. It holds the dictionary entry that
matched, in tokenizer spelling -- a umlsmatch debugging aid. Java's XMI records
the span and the resolved concepts but not the dictionary string behind them, so
there is nothing to map it from. It could be approximated by normalizing
``text``, but cTAKES matched against 2016AB with its own tokenization, so the
guess would be subtly wrong exactly where this database is used to diff the two
pipelines. An honest NULL beats a column that is usually right.

Two column renames happen on the way in, and they are not cosmetic:

  * ``group`` -> ``semantic_group`` -- ``GROUP`` is a reserved SQL keyword
  * ``start`` / ``end`` -> ``start_offset`` / ``end_offset`` -- ``END`` is
    reserved in SQLite (``CASE ... END``)

Leaving those names alone forces every later query to quote them, which is a
trap worth designing out once rather than tripping over repeatedly.

The schema is normalized into ``documents`` + ``annotations`` so a document is
stored once and counting distinct documents per concept is a plain join rather
than a ``COUNT(DISTINCT text_blob)``.

Every exporter in this directory now records a document by **file name**
(``umlsmatch.corpus.document_label``), so appending a CSV and a JSONL export of
one corpus lands on the same ``documents`` rows. ``--basename`` remains for the
inputs that do not follow that convention: a silver standard, output from
``python -m umlsmatch --json`` (whose ``source`` is the path it was given), and
any export written before this was true. Without it, those create a second
``documents`` row per note and double-count everything.

NOTE: the ``text`` column holds verbatim note content (PHI). Protect the .db
file exactly like the source notes. Load a ``--no-text`` CSV to avoid it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePath

BATCH = 5_000

DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        INTEGER PRIMARY KEY,
    source        TEXT NOT NULL UNIQUE,
    n_annotations INTEGER NOT NULL DEFAULT 0
);
"""

#: The ``annotations`` table, column by column: name -> (declaration, comment).
#:
#: A table rather than one CREATE TABLE literal because two scripts build this
#: table and they do not always want every column. ``parse_to_sqlite.py`` omits
#: the assertion attributes its pipeline never assesses -- a column that is NULL
#: in every row says something about the pipeline, not about any mention -- while
#: this script keeps them all, because a **Java cTAKES silver standard assesses
#: all six** and a table without the columns could not hold one. Generating both
#: from one definition is what keeps those two shapes compatible: the narrow
#: table is the wide one minus columns, never a second schema that drifted.
ANNOTATION_COLUMNS: dict[str, tuple[str, str]] = {
    "cui": ("TEXT NOT NULL", ""),
    "preferred_text": ("TEXT", ""),
    "semantic_group": ("TEXT", ""),
    "negated": ("INTEGER NOT NULL", "0/1, not TRUE/FALSE: SQLite has no bool"),
    # The five non-polarity attributes. NOT NULL is wrong here: NULL is the
    # load-bearing value, meaning "the producer did not assess this, and did not
    # claim otherwise". See the module docstring.
    "subject": ("TEXT", "'patient' | 'family_member' | 'other'"),
    "history_of": ("INTEGER", ""),
    "uncertain": ("INTEGER", ""),
    "conditional": ("INTEGER", ""),
    "generic": ("INTEGER", ""),
    "start_offset": ("INTEGER", ""),
    "end_offset": ("INTEGER", ""),
    "text": ("TEXT", ""),
    "term": ("TEXT", ""),
}

#: Indexes, each naming the columns it needs. One that names a column the table
#: does not have is skipped rather than failing the load -- see :func:`indexes`.
_INDEXES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("ix_ann_cui", ("cui",), ""),
    ("ix_ann_group", ("semantic_group",), ""),
    ("ix_ann_doc", ("doc_id",), ""),
    ("ix_ann_negated", ("negated",), ""),
    ("ix_ann_cui_neg", ("cui", "negated"),
     "the cohort query ('affirmed mentions of CUI X') hits this"),
    # Only these two of the five attributes get an index: they are the ones a
    # cohort query filters on ("patient's own history of X"). The other three
    # are either never populated by umlsmatch or too rare for an index to pay
    # for itself -- `conditional` is 0.2% of mentions even in cTAKES' output.
    ("ix_ann_subject", ("subject",), ""),
    ("ix_ann_history", ("history_of",), ""),
)

#: Attributes loaded as nullable 0/1. `negated` is not among them: it is always
#: assessed, so it goes through `_as_int_flag` and defaults to 0.
_NULLABLE_FLAGS = ("history_of", "uncertain", "conditional", "generic")


def annotation_columns(omit: Iterable[str] = ()) -> tuple[str, ...]:
    """Annotation column names in schema order, minus `omit`."""
    dropped = set(omit)
    return tuple(c for c in ANNOTATION_COLUMNS if c not in dropped)


def schema(omit: Iterable[str] = ()) -> str:
    """``CREATE TABLE`` for both tables, without the columns in `omit`."""
    width = max(len(c) for c in ANNOTATION_COLUMNS)
    lines = ["    id             INTEGER PRIMARY KEY,",
             "    doc_id         INTEGER NOT NULL REFERENCES documents(doc_id),"]
    columns = annotation_columns(omit)
    for i, name in enumerate(columns):
        declaration, comment = ANNOTATION_COLUMNS[name]
        comma = "," if i < len(columns) - 1 else ""
        line = f"    {name:<{width}} {declaration}{comma}"
        lines.append(f"{line}  -- {comment}" if comment else line)
    body = "\n".join(lines)
    return f"{DOCUMENTS_SCHEMA}\nCREATE TABLE IF NOT EXISTS annotations (\n{body}\n);\n"


def indexes(omit: Iterable[str] = ()) -> str:
    """``CREATE INDEX`` statements that the columns in this table support."""
    present = set(annotation_columns(omit)) | {"doc_id"}
    out = []
    for name, columns, comment in _INDEXES:
        if not present.issuperset(columns):
            continue
        if comment:
            out.append(f"-- {comment}")
        out.append(
            f"CREATE INDEX IF NOT EXISTS {name} ON annotations({', '.join(columns)});"
        )
    return "\n".join(out) + "\n"


def _as_int_flag(value) -> int:
    """CSV gives '0'/'1'; JSON gives real booleans. Normalize both."""
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "t"} else 0


def _as_nullable_flag(value) -> int | None:
    """Like :func:`_as_int_flag`, but preserves "not assessed" as NULL.

    JSON gives ``None`` directly. CSV cannot: it has no null, so an empty cell
    is the only available spelling and is read as one here. That does mean a
    CSV round-trip cannot distinguish an unassessed attribute from a blank one
    -- which is the same thing, so nothing is lost.
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_int_flag(value)


class Loader:
    """Batched inserts into whatever shape the `annotations` table actually has.

    The column list is read from the table rather than assumed, because the
    table is not always the full one: ``parse_to_sqlite.py`` omits the
    attributes its pipeline never assesses. Building the INSERT from the
    database means one loader fills either shape, and in particular that
    ``--append`` still works across the two.

    Values for a column the table lacks are dropped, and counted in
    :attr:`unstorable` so the caller can say so. That case is real: appending a
    cTAKES silver standard -- which assesses all six attributes -- into a
    narrow table would otherwise discard assessments in silence.
    """

    def __init__(self, conn: sqlite3.Connection, *, basename: bool = False) -> None:
        self.conn = conn
        self.basename = basename
        self.doc_ids: dict[str, int] = {
            src: did for did, src in conn.execute("SELECT doc_id, source FROM documents")
        }
        present = {r[1] for r in conn.execute("PRAGMA table_info(annotations)")}
        self.columns = ("doc_id", *(c for c in ANNOTATION_COLUMNS if c in present))
        self.missing = tuple(c for c in ANNOTATION_COLUMNS if c not in present)
        self.insert_sql = (
            f"INSERT INTO annotations ({','.join(self.columns)}) "
            f"VALUES ({','.join('?' * len(self.columns))})"
        )
        #: Column -> how many non-null values were dropped for want of a column.
        self.unstorable: dict[str, int] = {}
        self.batch: list[tuple] = []
        self.rows = 0

    def doc_id(self, source: str) -> int:
        if self.basename:
            source = PurePath(source.replace("\\", "/")).name
        hit = self.doc_ids.get(source)
        if hit is None:
            cur = self.conn.execute("INSERT INTO documents (source) VALUES (?)", (source,))
            hit = cur.lastrowid
            self.doc_ids[source] = hit
        return hit

    def add(self, source: str, ann: dict) -> None:
        row = {
            "doc_id": self.doc_id(source),
            "cui": ann.get("cui"),
            "preferred_text": ann.get("preferred_text"),
            # accept either spelling on the way in
            "semantic_group": ann.get("semantic_group") or ann.get("group"),
            "negated": _as_int_flag(ann.get("negated", 0)),
            "subject": ann.get("subject") or None,
            **{f: _as_nullable_flag(ann.get(f)) for f in _NULLABLE_FLAGS},
            "start_offset": ann.get("start_offset", ann.get("start")),
            "end_offset": ann.get("end_offset", ann.get("end")),
            "text": ann.get("text"),
            "term": ann.get("term"),
        }
        for column in self.missing:
            if row.get(column) is not None:
                self.unstorable[column] = self.unstorable.get(column, 0) + 1
        self.batch.append(tuple(row[c] for c in self.columns))
        self.rows += 1
        if len(self.batch) >= BATCH:
            self.flush()

    def flush(self) -> None:
        if self.batch:
            self.conn.executemany(self.insert_sql, self.batch)
            self.batch.clear()

    def report_unstorable(self, stream=sys.stderr) -> None:
        """Warn about values dropped for want of a column, if any."""
        for column, n in sorted(self.unstorable.items()):
            print(
                f"WARNING: {n:,} '{column}' values were dropped -- this database "
                f"has no {column} column. Rebuild it without --append to keep them.",
                file=stream,
            )


def load_csv(path: Path, loader: Loader) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            sys.exit(f"error: {path} has no header row")
        if "document" not in reader.fieldnames:
            sys.exit(
                f"error: {path} lacks a 'document' column -- is it from parse_to_csv.py?"
            )
        for row in reader:
            loader.add(row["document"], row)


def _silver_rows(mentions: list[dict]) -> Iterator[dict]:
    """Flatten Java cTAKES mentions into annotation-shaped dicts.

    A mention may carry several UMLS concepts (a drug mention typically has one
    per RxNorm form), so this yields one row per concept. The semantic group is
    derived from the concept's TUI using the same table the matcher uses, rather
    than guessed from the mention's Java class name.
    """
    from umlsmatch.umls.semantic_tui import group_for_tui

    for m in mentions:
        for concept in m.get("concepts") or ():
            cui = concept.get("cui")
            if not cui:
                continue
            tui = concept.get("tui")
            yield {
                "cui": cui,
                "preferred_text": concept.get("preferred_text"),
                "semantic_group": group_for_tui(tui).name if tui else "",
                "negated": m.get("negated", False),
                # cTAKES assesses all six, so none of these is NULL on a silver
                # load -- which is exactly what makes the diff against a
                # umlsmatch load informative about what we do not attempt.
                "subject": m.get("subject"),
                "history_of": m.get("history_of"),
                "uncertain": m.get("uncertain"),
                "conditional": m.get("conditional"),
                "generic": m.get("generic"),
                "start": m.get("begin"),
                "end": m.get("end"),
                "text": m.get("text"),
                "term": None,
            }


def load_jsonl(path: Path, loader: Loader) -> None:
    seen_records = 0
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"error: {path}:{n} is not valid JSON ({exc})")
            seen_records += 1
            source = record.get("source") or record.get("source_file") or f"{path.name}:{n}"

            if "annotations" in record:
                rows = record["annotations"]
            elif "mentions" in record:
                rows = _silver_rows(record["mentions"])
            else:
                sys.exit(
                    f"error: {path}:{n} has neither an 'annotations' nor a 'mentions' key "
                    f"(found: {', '.join(sorted(record)) or 'nothing'}).\n"
                    "Expected an export from examples 02/06, or a silver standard from "
                    "tools/run_java_ctakes.py."
                )
            for ann in rows:
                loader.add(source, ann)

    if seen_records == 0:
        sys.exit(f"error: {path} contained no JSON records")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="annotations .csv or .jsonl")
    ap.add_argument("database", type=Path, help="SQLite file to create or extend")
    ap.add_argument(
        "--append",
        action="store_true",
        help="add to an existing database instead of replacing it",
    )
    ap.add_argument(
        "--basename",
        action="store_true",
        help="key documents by file name, not full path (use for a silver "
             "standard, `python -m umlsmatch --json` output, or any export "
             "predating the shared file-name convention)",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"error: not found: {args.input}")

    suffix = args.input.suffix.lower()
    if suffix not in {".csv", ".jsonl", ".json"}:
        sys.exit(f"error: expected .csv or .jsonl, got {suffix!r}")

    if args.database.exists() and not args.append:
        args.database.unlink()
    args.database.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.database)
    conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
    # The full table: this loader's other input is a cTAKES silver standard,
    # which assesses all six attributes. `CREATE TABLE IF NOT EXISTS` makes
    # this a no-op when --append targets a narrower table written by
    # parse_to_sqlite.py, and the Loader then fills whatever is there.
    conn.executescript(schema())

    loader = Loader(conn, basename=args.basename)
    print(f"loading {args.input} ({args.input.stat().st_size/1e6:.1f} MB) ...", flush=True)

    try:
        if suffix == ".csv":
            load_csv(args.input, loader)
        else:
            load_jsonl(args.input, loader)
        loader.flush()

        # Writing an empty database and reporting success is worse than
        # failing: the zeros scroll past and the mistake resurfaces later as a
        # confusing empty query result.
        if loader.rows == 0:
            sys.exit(
                f"error: no annotations found in {args.input} -- nothing was written.\n"
                "The file parsed, but every record was empty. If it is a silver "
                "standard, check that its mentions carry resolved concepts."
            )
    except SystemExit:
        # Don't leave a half-written or empty database behind for the next run
        # to pick up and trust.
        conn.close()
        if not args.append:
            args.database.unlink(missing_ok=True)
        raise

    # Denormalized count, maintained here so the common "how big is this
    # document" question needs no join.
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
    has_text = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE text IS NOT NULL AND text != ''"
    ).fetchone()[0]

    print(f"\n  documents        {docs:,}")
    print(f"  annotations      {anns:,}")
    print(f"  distinct CUIs    {cuis:,}")
    # Guard the percentage, not the whole call: with the conditional on
    # `print(...)`'s argument, a run that produced no annotations prints a
    # stray blank line instead of skipping the row.
    if anns:
        print(f"  negated          {negated:,} ({100*negated/anns:.1f}%)")

    # Report each attribute's assessed count alongside its positive count. A
    # bare "conditional 0" would read as "nothing was conditional"; "0 of 0
    # assessed" reads as what it is.
    print("\n  assertion attributes:")
    for column in _NULLABLE_FLAGS:
        if column in loader.missing:
            # A missing column is a fact about *this database*, not about the
            # input: a silver standard assesses all six, and appending one into
            # a narrow table drops those values -- which is what the warning
            # after the counts says, and why this line does not claim the
            # attribute was never assessed.
            print(f"    {column:<20}{'no column in this database':>28}")
            continue
        assessed, positive = conn.execute(
            f"SELECT COUNT({column}), COALESCE(SUM({column}), 0) FROM annotations"
        ).fetchone()
        rate = f"{100 * positive / assessed:.1f}%" if assessed else "-"
        print(f"    {column:<20}{positive:>8,} of {assessed:,} assessed  {rate}")
    for subject, n in conn.execute(
        "SELECT COALESCE(subject, '(not assessed)'), COUNT(*) FROM annotations "
        "GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"    {'subject=' + subject:<20}{n:>8,}")

    print("\n  by semantic group:")
    for group, n in conn.execute(
        "SELECT semantic_group, COUNT(*) c FROM annotations GROUP BY 1 ORDER BY c DESC"
    ):
        print(f"    {group!s:<14}{n:>8,}")

    conn.close()
    print(f"\nwrote {args.database} ({args.database.stat().st_size/1e6:.1f} MB)")
    # Said after the counts, where it cannot be scrolled past: a value the input
    # carried and this table cannot hold is the one loss this script can cause.
    loader.report_unstorable(sys.stdout)
    if has_text:
        print("NOTE: the `text` column contains verbatim note content (PHI).")
    print(f"\nQuery it:  python examples/sqlite_browser.py {args.database}")
    print(f"       or:  sqlite3 {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

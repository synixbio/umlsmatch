#!/usr/bin/env python3
"""Compare annotation exports of one corpus: do the formats actually agree?

    python examples/compare_exports.py
    python examples/compare_exports.py out/runs/a/annotations.csv out/runs/b/annotations.db

With no arguments it picks the newest CSV, JSONL, Parquet and SQLite export
under `out/runs/` and compares them. Name paths explicitly to compare particular
runs -- two runs of the same format is a valid comparison too, and is how you
check that a configuration change did what you expected.

There are five ways to write these annotations (`parse_to_csv.py`,
`parse_to_jsonl.py`, `parse_to_parquet.py`, `parse_to_sqlite.py`,
`parse_to_jsonl_batch.py`) and they are supposed to produce the same rows. That
claim is only worth making if something checks it, because the ways they can
drift are quiet: a column added to one exporter and not the others, a document
keyed by path here and by file name there, an attribute written as 0 in a format
that cannot spell null. Each of those survives a row-count check and changes
what a query returns.

So this normalizes every export into one shape and compares them exactly:

  * Column renames are undone -- SQLite's `semantic_group` and
    `start_offset`/`end_offset` exist because `GROUP` and `END` are reserved
    words, not because the data differs.
  * Each format's spelling of an optional boolean -- `""`, `0`/`1`, `true`,
    `NULL` -- collapses to True/False/None, so "not assessed" compares equal to
    "not assessed" and unequal to "assessed false".
  * Fields one export lacks are excluded from the comparison rather than
    failing it, and the report says which. Comparing a `--no-text` CSV against
    a full Parquet export is a reasonable thing to do; it just cannot say
    anything about `text`.

It also reports what each format *costs*: bytes on disk per annotation, how long
a full read takes, and how long the query these exports exist for takes --
affirmed mentions per concept -- asked in each format's own idiom. That last one
is where columnar and indexed storage separate from the text formats, and it is
the number worth choosing a format on, since writing costs roughly the same
either way.

Exit status is 0 when the exports agree, 1 when they differ, 2 when there was
nothing to compare. A differing exit status is the point: this is runnable as a
check after changing an exporter, not only as a report to read.

NOTE: reads PHI (every export quotes note text) but prints none of it. A row
that differs is reported by document, CUI and offsets; pass --show-text to
include the matched span when that is what differs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.runs import DEFAULT_RUN_ROOT, MANIFEST_NAME, SQLITE_SUFFIXES, find_artifacts

#: The canonical row, in report order. Every loader below maps into this, and
#: the comparison runs over whichever of these fields all inputs carry.
FIELDS = (
    "document",
    "cui",
    "preferred_text",
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
)

#: Fields that identify a row well enough to report a difference without
#: quoting note text. `text` is deliberately absent -- see --show-text.
IDENTIFYING = ("document", "cui", "start", "end")

#: Attributes that may be "not assessed". The whole reason this script
#: normalizes rather than compares raw values: CSV writes an empty cell here,
#: JSON writes null, SQLite writes NULL, and all three mean the same thing.
NULLABLE = ("history_of", "uncertain", "conditional", "generic")

#: Suffixes to look for when no paths are given, newest first within each.
DISCOVERY = {
    "csv": (".csv",),
    "jsonl": (".jsonl",),
    "parquet": (".parquet",),
    "sqlite": tuple(SQLITE_SUFFIXES),
}


@dataclass
class Export:
    """One loaded export, normalized."""

    path: Path
    kind: str
    rows: list[dict]
    #: Canonical fields this format actually carried. A `--no-text` export is
    #: missing `text`; an export written before `term` was added lacks that.
    fields: tuple[str, ...]
    load_seconds: float
    size: int
    manifest: dict | None = None
    notes: list[str] = field(default_factory=list)
    #: What the report calls this export. Assigned by :func:`name_exports`.
    display: str = ""

    @property
    def label(self) -> str:
        """Run id plus file name -- enough to tell two runs apart in a table."""
        return f"{self.path.parent.name}/{self.path.name}"


def name_exports(exports: list[Export]) -> None:
    """Give each export a name the report can repeat unambiguously.

    The format alone is enough when the formats differ, which is the common
    case. It is actively misleading when they do not -- comparing two runs of
    the same exporter is a normal thing to do, and "13,108 rows only in csv,
    0 only in csv" is a line nobody can act on. Those get their run directory
    attached instead.
    """
    counts = Counter(e.kind for e in exports)
    for export in exports:
        export.display = (
            export.kind if counts[export.kind] == 1
            else f"{export.kind}:{export.path.parent.name}"
        )


class Unreadable(Exception):
    """This file is not an annotation export of a shape we understand."""


# --- loaders ------------------------------------------------------------------
#
# Each returns (rows, fields). They are deliberately separate functions rather
# than one dispatcher with branches: the per-format quirks are the subject of
# this script, and burying them in `if` arms would hide exactly what a reader
# came here to see.


def _flag(value) -> bool | None:
    """Every format's spelling of an optional boolean -> True/False/None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t"}
    return bool(value)


def _row(source: dict, *, fields: tuple[str, ...]) -> dict:
    """Canonicalize one record, keeping only `fields`."""
    out = {}
    for name in fields:
        value = source.get(name)
        if name in NULLABLE:
            value = _flag(value)
        elif name == "negated":
            value = bool(_flag(value))
        elif name in ("start", "end"):
            value = int(value)
        elif name == "subject":
            value = value or None
        else:
            value = "" if value is None else str(value)
        out[name] = value
    return out


def load_csv(path: Path) -> tuple[list[dict], tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "cui" not in reader.fieldnames:
            raise Unreadable("no 'cui' column -- not an annotations CSV")
        fields = tuple(f for f in FIELDS if f in reader.fieldnames)
        return [_row(r, fields=fields) for r in reader], fields


def load_jsonl(path: Path) -> tuple[list[dict], tuple[str, ...]]:
    rows: list[dict] = []
    fields: tuple[str, ...] = ()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "annotations" not in record:
                # A Java cTAKES silver standard keys its rows differently and
                # carries no `term`; load_to_sqlite.py is the tool for those.
                raise Unreadable(
                    f"line {n} has no 'annotations' key "
                    f"(found: {', '.join(sorted(record)) or 'nothing'})"
                )
            document = record.get("source") or record.get("source_file") or ""
            for a in record["annotations"]:
                if not fields:
                    # Inferred from the first annotation: every record in one
                    # file comes from the same exporter, so they agree.
                    present = set(a) | {"document"}
                    fields = tuple(f for f in FIELDS if f in present)
                rows.append(_row({**a, "document": document}, fields=fields))
    if not rows:
        raise Unreadable("no annotations in any record")
    return rows, fields


def load_parquet(path: Path) -> tuple[list[dict], tuple[str, ...]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise Unreadable(f"pyarrow is not installed ({exc}); pip install pyarrow") from exc
    table = pq.read_table(path)
    fields = tuple(f for f in FIELDS if f in table.column_names)
    if "cui" not in fields:
        raise Unreadable("no 'cui' column -- not an annotations table")
    columns = {f: table.column(f).to_pylist() for f in fields}
    return (
        [_row({f: columns[f][i] for f in fields}, fields=fields)
         for i in range(table.num_rows)],
        fields,
    )


#: SQLite's column names differ from the canonical ones for a reason -- `GROUP`
#: and `END` are reserved words -- so the mapping is undone here rather than
#: treated as a difference in the data.
_DB_COLUMNS = {
    "document": "d.source",
    "group": "a.semantic_group",
    "start": "a.start_offset",
    "end": "a.end_offset",
}


def load_sqlite(path: Path) -> tuple[list[dict], tuple[str, ...]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = {"annotations", "documents"} - tables
        if missing:
            # The newest database under out/runs/ may be one notes_of_interest.py
            # wrote, which has a different shape entirely.
            raise Unreadable(f"no {'/'.join(sorted(missing))} table")
        available = {f"a.{r[1]}" for r in conn.execute("PRAGMA table_info(annotations)")}
        # `document` is `documents.source`, reached by the join below, so it is
        # never in `annotations`' own column list.
        available |= {f"d.{r[1]}" for r in conn.execute("PRAGMA table_info(documents)")}
        fields = tuple(f for f in FIELDS if _DB_COLUMNS.get(f, f"a.{f}") in available)
        select = ", ".join(f'{_DB_COLUMNS.get(f, f"a.{f}")} AS "{f}"' for f in fields)
        rows = conn.execute(
            f"SELECT {select} FROM annotations a JOIN documents d USING(doc_id)"
        ).fetchall()
        return [_row(dict(zip(fields, r, strict=True)), fields=fields) for r in rows], fields
    finally:
        conn.close()


LOADERS = {"csv": load_csv, "jsonl": load_jsonl, "parquet": load_parquet,
           "sqlite": load_sqlite}


def kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SQLITE_SUFFIXES:
        return "sqlite"
    return {".csv": "csv", ".jsonl": "jsonl", ".json": "jsonl",
            ".parquet": "parquet"}.get(suffix, "")


def load(path: Path) -> Export:
    kind = kind_of(path)
    if not kind:
        raise Unreadable(f"unrecognized extension {path.suffix!r}")
    started = time.perf_counter()
    rows, fields = LOADERS[kind](path)
    elapsed = time.perf_counter() - started
    manifest_path = path.parent / MANIFEST_NAME
    manifest = None
    if manifest_path.is_file():
        # A manifest is provenance, not data: an unreadable one costs a line of
        # the report, not the comparison.
        with contextlib.suppress(json.JSONDecodeError):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Export(path, kind, rows, fields, elapsed, path.stat().st_size, manifest)


def discover(root: Path) -> list[Export]:
    """Newest readable export of each format under `root`.

    Newest *readable*, not simply newest: run directories accumulate databases
    and JSONL files that are not annotation exports, so each candidate list is
    walked until one loads. What was skipped is reported rather than silently
    passed over -- "it compared a different file than I meant" is the failure
    mode worth making loud.
    """
    found: list[Export] = []
    for kind, suffixes in DISCOVERY.items():
        skipped: list[str] = []
        for candidate in find_artifacts(suffixes, root=root):
            try:
                export = load(candidate)
            except (Unreadable, OSError, json.JSONDecodeError, sqlite3.Error) as exc:
                skipped.append(f"{kind} candidate {candidate.parent.name}/"
                               f"{candidate.name}: {exc}")
                continue
            export.notes = [f"skipped {s}" for s in skipped]
            found.append(export)
            break
    return found


# --- comparison ---------------------------------------------------------------


def keys(export: Export, fields: tuple[str, ...], *, basename: bool) -> list[tuple]:
    """Rows as sortable tuples over `fields`, for exact comparison."""
    out = []
    for row in export.rows:
        values = tuple(
            Path(str(row[f]).replace("\\", "/")).name
            if (basename and f == "document") else row[f]
            for f in fields
        )
        out.append(values)
    # Sorted, not set: duplicate rows are legitimate (the same concept can be
    # matched twice at the same span by different dictionary terms) and a set
    # would silently call two exports equal when one dropped a duplicate.
    return sorted(out, key=repr)


def describe(row: tuple, fields: tuple[str, ...], *, show_text: bool) -> str:
    """One differing row, without quoting note text unless asked."""
    shown = fields if show_text else tuple(f for f in fields if f != "text")
    pairs = {f: v for f, v in zip(fields, row, strict=True) if f in shown}
    head = " ".join(f"{f}={pairs[f]!r}" for f in IDENTIFYING if f in pairs)
    rest = " ".join(f"{f}={v!r}" for f, v in pairs.items() if f not in IDENTIFYING)
    return f"{head}  {rest}"


def compare(exports: list[Export], *, basename: bool, show_text: bool,
            examples: int) -> bool:
    """Print the agreement report. True when every export matched."""
    common = tuple(f for f in FIELDS if all(f in e.fields for e in exports))
    excluded = [f for f in FIELDS if f not in common]

    print("\ncomparing on", len(common), "fields:", ", ".join(common))
    if excluded:
        # Not a failure: comparing a --no-text CSV against a full Parquet
        # export is reasonable, it just cannot say anything about `text`.
        for f in excluded:
            lacking = [e.display for e in exports if f not in e.fields]
            # Naming all of them is noise when the answer is "none of them" --
            # which is the usual case for an attribute no pipeline assessed.
            who = "none of them" if len(lacking) == len(exports) else ", ".join(lacking)
            print(f"  excluded {f!r}: carried by {who}")

    reference = exports[0]
    ref_keys = keys(reference, common, basename=basename)
    agreed = True

    print()
    for export in exports[1:]:
        other = keys(export, common, basename=basename)
        if other == ref_keys:
            print(f"  {reference.display:>14} == {export.display:<14} identical "
                  f"({len(other):,} rows)")
            continue

        agreed = False
        print(f"  {reference.display:>14} != {export.display:<14} DIFFER "
              f"({len(ref_keys):,} vs {len(other):,} rows)")

        # Diagnose the difference that this project actually hit, rather than
        # leaving someone to infer it from a dump of 16,000 rows: two exports
        # that key documents differently disagree on every single row.
        if (not basename and "document" in common
                and keys(reference, common, basename=True)
                == keys(export, common, basename=True)):
            print("           -> identical apart from how documents are keyed "
                  "(path vs file name). Re-run with --basename to compare "
                  "the rest.")
            continue

        only_ref = Counter(map(repr, ref_keys)) - Counter(map(repr, other))
        only_other = Counter(map(repr, other)) - Counter(map(repr, ref_keys))
        print(f"           {sum(only_ref.values()):,} rows only in "
              f"{reference.display}, {sum(only_other.values()):,} only in {export.display}")
        for label, rows in ((reference.display, ref_keys), (export.display, other)):
            source = only_ref if label == reference.display else only_other
            shown = 0
            for row in rows:
                if shown >= examples:
                    break
                if source.get(repr(row)):
                    print(f"             only in {label}: "
                          f"{describe(row, common, show_text=show_text)}")
                    shown += 1
    return agreed


# --- cost ---------------------------------------------------------------------


def affirmed_by_cui(export: Export) -> Counter:
    """Count affirmed mentions per concept, in each format's own idiom.

    The query these exports exist for, and the one that separates them: CSV and
    JSONL must parse every row to answer it, Parquet reads two columns, SQLite
    walks an index. Written four times on purpose -- routing them all through
    the normalized rows above would measure this script instead of the formats.
    """
    if export.kind == "csv":
        with export.path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            cui_at, neg_at = header.index("cui"), header.index("negated")
            return Counter(r[cui_at] for r in reader if r[neg_at] in ("0", "", "false"))

    if export.kind == "jsonl":
        counts: Counter = Counter()
        with export.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    for a in json.loads(line)["annotations"]:
                        if not a["negated"]:
                            counts[a["cui"]] += 1
        return counts

    if export.kind == "parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(export.path, columns=["cui", "negated"])
        cui = table.column("cui").to_pylist()
        negated = table.column("negated").to_pylist()
        return Counter(c for c, n in zip(cui, negated, strict=True) if not n)

    conn = sqlite3.connect(f"file:{export.path}?mode=ro", uri=True)
    try:
        return Counter(dict(conn.execute(
            "SELECT cui, COUNT(*) FROM annotations WHERE negated = 0 GROUP BY cui")))
    finally:
        conn.close()


def cost_report(exports: list[Export], *, repeat: int) -> None:
    print("\ncost")
    print(f"  {'format':<15}{'size':>11}{'per row':>10}{'normalize':>12}"
          f"{'top-CUI query':>16}")
    for export in exports:
        rows = len(export.rows) or 1
        best = None
        counts = None
        for _ in range(max(1, repeat)):
            started = time.perf_counter()
            counts = affirmed_by_cui(export)
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min(best, elapsed)
        print(f"  {export.display:<15}{export.size:>11,}B{export.size / rows:>9.0f}B"
              f"{export.load_seconds * 1000:>10.0f}ms{best * 1000:>14.1f}ms"
              f"   ({len(counts or ()):,} CUIs)")
    # Two different things, and conflating them would be the easy mistake to
    # make from this table: `normalize` is the cost of *this script's* loader,
    # which builds a Python dict per row for every field and so says more about
    # the loader than the file. The query column is the one to choose a format
    # on -- each format answers it the way that format is meant to be asked.
    print(f"  normalize = this script's row-by-row load (not a property of the format)"
          f"\n  top-CUI query = best of {repeat}, asked in each format's own idiom")


def provenance(exports: list[Export]) -> None:
    """Print where each export came from, and flag mismatched corpora.

    The likeliest way to misread this report is to compare exports of two
    different corpora, which the run manifests can rule out and a row count
    cannot.
    """
    print("\ninputs")
    for export in exports:
        manifest = export.manifest or {}
        script = manifest.get("script", "?")
        created = manifest.get("created", "?")
        print(f"  {export.display:<15}{export.label:<40}{script:<24}{created}")
        for note in export.notes:
            print(f"           {note}")

    corpora = {
        str((e.manifest or {}).get("input_dir"))
        for e in exports if (e.manifest or {}).get("input_dir")
    }
    if len(corpora) > 1:
        print("\n  WARNING: these runs name different input corpora --")
        for corpus in sorted(corpora):
            print(f"           {corpus}")
        print("           differences below are expected and mean nothing.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "exports",
        nargs="*",
        type=Path,
        help=f"annotation exports to compare. Omit to take the newest of each "
             f"format under {DEFAULT_RUN_ROOT}.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"where to look when no paths are given (default: {DEFAULT_RUN_ROOT})",
    )
    ap.add_argument(
        "--basename",
        action="store_true",
        help="compare documents by file name, for exports that keyed them by path",
    )
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="include the matched span when reporting a differing row (PHI)",
    )
    ap.add_argument("--examples", type=int, default=3,
                    help="differing rows to show per side (default: 3)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="timing runs per format, best taken (default: 3)")
    ap.add_argument("--no-cost", action="store_true", help="skip the timing report")
    args = ap.parse_args()

    if args.exports:
        exports = []
        for path in args.exports:
            try:
                exports.append(load(path))
            except (Unreadable, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
                print(f"error: {path}: {exc}", file=sys.stderr)
                return 2
    else:
        exports = discover(args.root)

    if len(exports) < 2:
        where = "the paths given" if args.exports else f"{args.root}"
        print(f"nothing to compare: found {len(exports)} readable export(s) in {where}",
              file=sys.stderr)
        return 2

    name_exports(exports)
    provenance(exports)
    agreed = compare(exports, basename=args.basename, show_text=args.show_text,
                     examples=args.examples)
    if not args.no_cost:
        cost_report(exports, repeat=args.repeat)

    print("\n" + ("all exports agree" if agreed else "exports DIFFER -- see above"))
    return 0 if agreed else 1


if __name__ == "__main__":
    raise SystemExit(main())

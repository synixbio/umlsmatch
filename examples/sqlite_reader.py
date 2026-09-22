#!/usr/bin/env python3
"""Browse a SQLite database — what is in it, and what the rows look like.

    python examples/sqlite_reader.py                       # newest db under out/runs/
    python examples/sqlite_reader.py out/annotations.db
    python examples/sqlite_reader.py --table annotations --limit 20
    python examples/sqlite_reader.py --schema

**With no path it opens the most recently written database under `out/runs/`.**
Run directories are named by timestamp so that nothing overwrites anything,
which is exactly what makes the name tedious to retype -- so the common case of
"show me what I just built" needs no argument at all. Newest is judged by
modification time, not by run id: a long corpus job finishes after a short one
that started later, and what you mean by "the last one" is the one that
finished.

**`sqlite_reader.py` vs `sqlite_browser.py`** -- near-identical names, different
jobs, and the distinction is the whole reason both exist:

  * `sqlite_browser.py` answers *clinical* questions with fixed reports -- top
    concepts, negation rates, co-occurrence. It knows the schema this project
    writes and would fail on any other.
  * `sqlite_reader.py` (this file) answers *structural* ones: what tables
    exist, what columns, what a row actually looks like. It assumes no schema,
    so it also reads a database this project did not build.

Reach for this one when the database came from somewhere else, or when a query
is returning something surprising and you need to see the rows behind it.

Opened read-only, like `sqlite_browser.py`: browsing should never be able to
modify the corpus.

NOTE: the `annotations.text` column is verbatim note text, so row output is
PHI. Pass --no-text to omit any column named `text`.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from umlsmatch.runs import DEFAULT_RUN_ROOT, SQLITE_SUFFIXES, latest_artifact

#: Longest cell printed before truncating. Wide enough for a CUI, a semantic
#: group or a short span; note text is truncated hard, which is a readability
#: measure and emphatically not a de-identification one -- use --no-text.
CELL = 32


def _tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """(name, type) for every user table and view, in alphabetical order.

    `sqlite_%` is filtered out: those are SQLite's own bookkeeping, and listing
    them alongside the real tables invites someone to query one.
    """
    return list(
        conn.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    )


def _count(conn: sqlite3.Connection, table: str) -> int | None:
    """Row count, or None if the table cannot be counted.

    A view can reference a table that no longer exists; that is worth showing
    as an unknown count rather than aborting the listing of everything else.
    """
    try:
        # The name comes from sqlite_master, not from the caller, so it cannot
        # be a parameter -- SQLite takes no placeholder for an identifier.
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.Error:
        return None


def _print_rows(cur: sqlite3.Cursor, *, no_text: bool) -> bool:
    """Print a cursor as a fixed-width table. True if any column was elided."""
    cols = [c[0] for c in cur.description]
    keep = [i for i, c in enumerate(cols) if not (no_text and c == "text")]
    elided = len(keep) != len(cols)

    rows = cur.fetchall()
    if not rows:
        print("  (no rows)")
        return elided

    widths = [
        min(max(len(cols[i]), *(len(str(r[i])) for r in rows)), CELL) for i in keep
    ]
    header = "  ".join(cols[i][:w].ljust(w) for i, w in zip(keep, widths, strict=True))
    print("  " + header)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        cells = (
            "NULL" if r[i] is None else str(r[i]).replace("\n", " ")
            for i in keep
        )
        print("  " + "  ".join(c[:w].ljust(w) for c, w in zip(cells, widths, strict=True)))
    return elided


def _resolve(given: Path | None, root: Path) -> Path:
    """The database to open: `given`, or the newest one under `root`."""
    if given is not None:
        if not given.is_file():
            sys.exit(f"error: {given} not found.")
        return given

    found = latest_artifact(SQLITE_SUFFIXES, root=root)
    if found is None:
        sys.exit(
            f"error: no database found under {root}/\n"
            "Build one from a run's annotations, writing it beside them:\n"
            "  python examples/parse_to_jsonl.py free_texts/synthetic\n"
            "  python examples/load_to_sqlite.py \\\n"
            f"      {root}/<run-id>/annotations.jsonl \\\n"
            f"      {root}/<run-id>/annotations.db\n"
            "...or pass a path directly:\n"
            "  python examples/sqlite_reader.py path/to.db"
        )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "database",
        type=Path,
        nargs="?",
        help=f"SQLite file. Omit to open the newest one under {DEFAULT_RUN_ROOT}.",
    )
    ap.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"where to look for the newest database (default: {DEFAULT_RUN_ROOT})",
    )
    ap.add_argument("--table", default=None, help="show rows from this table")
    ap.add_argument("--limit", type=int, default=10, help="rows to show (default: 10)")
    ap.add_argument("--schema", action="store_true", help="print the full DDL")
    ap.add_argument(
        "--no-text",
        action="store_true",
        help="omit any column named `text` -- it holds verbatim note content",
    )
    args = ap.parse_args()

    if args.limit < 1:
        ap.error("--limit must be >= 1")

    path = _resolve(args.database, args.runs)

    # Read-only, for the same reason as 08: a browser should not be able to
    # write to the thing it is browsing, however it is invoked.
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.DatabaseError as exc:
        sys.exit(f"error: {path} is not a readable SQLite database ({exc})")

    try:
        stat = path.stat()
        when = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        chose = "" if args.database is not None else f"  (newest under {args.runs})"
        print(f"{path}{chose}")
        print(f"{stat.st_size / 1e6:,.1f} MB, modified {when}\n")

        tables = _tables(conn)
        if not tables:
            print("(no tables)")
            return 0

        if args.schema:
            for name, _kind in tables:
                ddl = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
                ).fetchone()[0]
                print(f"{ddl};\n")
            return 0

        if args.table is None:
            width = max(len(n) for n, _ in tables)
            print("TABLES")
            for name, kind in tables:
                n = _count(conn, name)
                shown = "?" if n is None else f"{n:,}"
                suffix = "" if kind == "table" else f"  ({kind})"
                print(f"  {name.ljust(width)}  {shown:>12} rows{suffix}")
            print(
                f"\n  --table NAME   browse rows        --schema   full DDL"
                f"\n  --limit N      rows to show (default: {args.limit})"
            )
            return 0

        known = {n for n, _ in tables}
        if args.table not in known:
            # Listing the alternatives beats "no such table": the usual cause
            # is a plural/singular slip, and the answer is on screen already.
            sys.exit(
                f"error: no table named {args.table!r} in {path}\n"
                f"       available: {', '.join(sorted(known))}"
            )

        print(f"{args.table}, first {args.limit} row(s)\n")
        cur = conn.execute(f'SELECT * FROM "{args.table}" LIMIT ?', (args.limit,))
        elided = _print_rows(cur, no_text=args.no_text)

        if elided:
            print("\n  (`text` column omitted: --no-text)")
        elif any(c[0] == "text" for c in cur.description):
            print(
                "\nNOTE: the `text` column is verbatim note content (PHI). "
                "Pass --no-text to omit it.",
                file=sys.stderr,
            )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

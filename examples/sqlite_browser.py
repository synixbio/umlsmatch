#!/usr/bin/env python3
"""Useful queries against a database built by load_to_sqlite.py.

    python examples/sqlite_browser.py                       # newest db under out/runs/
    python examples/sqlite_browser.py out/annotations.db
    python examples/sqlite_browser.py --cui C0011849
    python examples/sqlite_browser.py --sql "SELECT ..."

**With no path it opens the most recently written database under `out/runs/`**,
the same default as `sqlite_reader.py` -- run ids are timestamps so nothing
overwrites anything, which is exactly what makes them tedious to retype.
Newest is by modification time, not run id: a long job finishes after a short
one that started later.

Unlike `sqlite_reader.py`, this script *knows the schema* -- every report below
assumes the `annotations` and `documents` tables `load_to_sqlite.py` writes. A
database without them is reported as such rather than failing mid-report, which
is the one thing the auto-default makes newly likely: the newest database under
out/runs/ may be one `notes_of_interest.py` wrote, which has a different shape
entirely.

Each report below prints the SQL it ran, so the script doubles as a cookbook --
copy a query into your own code or into `sqlite3` directly.

The queries worth studying are the last two:

  * **Negation rate per concept** shows which concepts are usually *ruled out*
    rather than asserted. In clinical notes that set is large and predictable
    (chest pain, fever, shortness of breath), which is exactly why counting raw
    mentions overstates prevalence.
  * **Co-occurrence** finds concepts appearing in the same documents -- the
    starting point for comorbidity analysis.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from umlsmatch.runs import DEFAULT_RUN_ROOT, SQLITE_SUFFIXES, find_artifacts

#: Tables every report here reads. Checked up front rather than discovered when
#: the third report happens to be the first one to touch `annotations`.
REQUIRED_TABLES = ("annotations", "documents")

TOP_CONCEPTS = """
SELECT cui,
       COALESCE(preferred_text, '?')          AS concept,
       semantic_group                         AS grp,
       COUNT(*)                               AS mentions,
       COUNT(DISTINCT doc_id)                 AS documents
FROM annotations
WHERE negated = 0
GROUP BY cui
ORDER BY documents DESC, mentions DESC
LIMIT ?
"""

BY_GROUP = """
SELECT semantic_group                          AS grp,
       COUNT(*)                                AS mentions,
       COUNT(DISTINCT cui)                     AS distinct_concepts,
       SUM(negated)                            AS negated
FROM annotations
GROUP BY semantic_group
ORDER BY mentions DESC
"""

NEGATION_RATE = """
SELECT cui,
       COALESCE(preferred_text, '?')                        AS concept,
       COUNT(*)                                             AS mentions,
       SUM(negated)                                         AS negated,
       ROUND(100.0 * SUM(negated) / COUNT(*), 1)            AS pct_negated
FROM annotations
GROUP BY cui
HAVING COUNT(*) >= ?
ORDER BY pct_negated DESC, mentions DESC
LIMIT ?
"""

# Self-join on doc_id, with a.cui < b.cui so each pair appears once.
CO_OCCURRENCE = """
SELECT a.cui AS cui_a, COALESCE(a.preferred_text,'?') AS concept_a,
       b.cui AS cui_b, COALESCE(b.preferred_text,'?') AS concept_b,
       COUNT(DISTINCT a.doc_id) AS documents
FROM annotations a
JOIN annotations b
  ON a.doc_id = b.doc_id
 AND a.cui < b.cui
WHERE a.negated = 0 AND b.negated = 0
  AND a.semantic_group = 'DISORDER'
  AND b.semantic_group = 'DISORDER'
GROUP BY a.cui, b.cui
HAVING documents >= 2
ORDER BY documents DESC
LIMIT ?
"""

COHORT = """
SELECT d.source,
       SUM(CASE WHEN a.negated = 0 THEN 1 ELSE 0 END) AS affirmed,
       SUM(a.negated)                                 AS negated
FROM annotations a
JOIN documents d ON d.doc_id = a.doc_id
WHERE a.cui = ?
GROUP BY d.doc_id
ORDER BY affirmed DESC, d.source
"""

DOC_SIZES = """
SELECT source, n_annotations
FROM documents
ORDER BY n_annotations DESC
LIMIT ?
"""


def show(conn: sqlite3.Connection, title: str, sql: str, params=(), *, widths=None) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}")
    print("\n".join("  " + ln for ln in sql.strip().splitlines()))
    print()
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    if not rows:
        print("  (no rows)")
        return
    widths = widths or [
        min(max(len(str(c)), *(len(str(r[i])) for r in rows)), 38)
        for i, c in enumerate(cols)
    ]
    # strict: one width per column, always. A mismatch means the caller's
    # `widths` or the query's column count is wrong, and silently truncating
    # the table would hide it.
    print("  " + "  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths, strict=True)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(v)[:w].ljust(w) for v, w in zip(r, widths, strict=True)))


def tables_of(path: Path) -> set[str] | None:
    """Table and view names in `path`, or None if it cannot be read at all.

    None rather than an exception because this runs over *candidate* files
    during auto-selection: a `.db` that turns out to be something else should
    drop out of the running quietly, not abort the search.
    """
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return {
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def missing_tables(path: Path) -> list[str] | None:
    """Which of :data:`REQUIRED_TABLES` `path` lacks; None if unreadable."""
    present = tables_of(path)
    if present is None:
        return None
    return [t for t in REQUIRED_TABLES if t not in present]


def resolve(given: Path | None, root: Path) -> Path:
    """The database to open: `given`, or the newest *usable* one under `root`.

    The distinction matters. An explicit path is a decision -- if it is the
    wrong shape, say so rather than quietly reporting on some other file.
    But with no path the caller asked for "the newest database", and every
    report here needs a particular schema, so the newest one that *cannot* be
    reported on is not an answer to that question. `notes_of_interest.py`
    writes a differently-shaped .sqlite into these same run directories, so
    refusing at the first incompatible file would mean a search that lands on
    one is simply broken until the user works out why.

    So: skip what cannot be read, and only complain once nothing is left.
    """
    if given is not None:
        if not given.is_file():
            sys.exit(
                f"error: {given} not found.\n"
                "Build it with:  python examples/load_to_sqlite.py <export> <db>"
            )
        missing = missing_tables(given)
        if missing is None:
            sys.exit(f"error: {given} is not a readable SQLite database.")
        if missing:
            present = tables_of(given) or set()
            sys.exit(
                f"error: {given} is not a database this script can report on.\n"
                f"       missing table(s): {', '.join(missing)}\n"
                f"       it has: {', '.join(sorted(present)) or '(nothing)'}\n"
                "Build one with:  python examples/load_to_sqlite.py <export> <db>\n"
                "Or browse this one's actual shape:\n"
                f"  python examples/sqlite_reader.py {given}"
            )
        return given

    candidates = find_artifacts(SQLITE_SUFFIXES, root=root)
    for path in candidates:
        if missing_tables(path) == []:
            return path

    if not candidates:
        sys.exit(
            f"error: no database found under {root}/\n"
            "Export a corpus, then load it into a run directory:\n"
            "  python -m umlsmatch free_texts/synthetic --json -o out/annotations.jsonl\n"
            "  python examples/load_to_sqlite.py --basename \\\n"
            "      out/annotations.jsonl \\\n"
            f"      {root}/<run-id>/annotations.db\n"
            "...or pass a path directly:\n"
            "  python examples/sqlite_browser.py path/to.db"
        )
    # There are databases, just none with the tables these reports read. Name
    # the newest so the message is actionable rather than a bare count.
    sys.exit(
        f"error: found {len(candidates)} database(s) under {root}/, but none has "
        f"the tables this script reports on ({', '.join(REQUIRED_TABLES)}).\n"
        f"       newest was {candidates[0]}\n"
        "Build one with:\n"
        "  python examples/load_to_sqlite.py \\\n"
        f"      {root}/<run-id>/annotations.jsonl \\\n"
        f"      {root}/<run-id>/annotations.db\n"
        "Or browse what is there:\n"
        f"  python examples/sqlite_reader.py {candidates[0]}"
    )


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
    ap.add_argument("--top", type=int, default=12, help="rows per report (default: 12)")
    ap.add_argument("--min-mentions", type=int, default=10, help="for the negation report")
    ap.add_argument("--cui", default=None, help="show the per-document cohort for one CUI")
    ap.add_argument("--sql", default=None, help="run an arbitrary read-only query instead")
    args = ap.parse_args()

    chosen = args.database is None
    path = resolve(args.database, args.runs)

    # `resolve` has already established that this opens and carries the tables
    # every report reads, whether it was named or selected.
    #
    # Read-only on purpose: this is an analysis tool, and a stray UPDATE/DELETE
    # typed into --sql should bounce off rather than mutate the extracted corpus.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    if chosen:
        print(f"({path} -- newest usable under {args.runs})", file=sys.stderr)
    try:
        if args.sql:
            try:
                show(conn, "custom query", args.sql)
            except sqlite3.OperationalError as exc:
                if "readonly" in str(exc):
                    print(
                        f"error: this database is opened read-only, so that query "
                        f"cannot run ({exc}).\nUse the sqlite3 CLI if you intend to "
                        f"modify it.",
                        file=sys.stderr,
                    )
                else:
                    print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0

        if args.cui:
            show(conn, f"Documents mentioning {args.cui}", COHORT, (args.cui.upper(),))
            return 0

        n_docs, n_ann = conn.execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM annotations)"
        ).fetchone()
        print(f"{path}: {n_docs:,} documents, {n_ann:,} annotations")

        show(conn, "Annotations by semantic group", BY_GROUP)
        show(conn, f"Top {args.top} affirmed concepts by document frequency",
             TOP_CONCEPTS, (args.top,))
        show(conn, f"Largest {args.top} documents", DOC_SIZES, (args.top,))
        show(
            conn,
            f"Most-often-negated concepts (>= {args.min_mentions} mentions)",
            NEGATION_RATE,
            (args.min_mentions, args.top),
        )
        show(conn, f"Top {args.top} co-occurring disorder pairs", CO_OCCURRENCE, (args.top,))

        print(
            "\nReminder: negation precision is ~0.50 (docs/USER_GUIDE.md S:7), so the\n"
            "negation report indicates where to look, not a settled answer."
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

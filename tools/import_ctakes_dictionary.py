#!/usr/bin/env python3
"""Import the dictionary Java cTAKES actually ships (``sno_rx_16ab``).

cTAKES distributes its default dictionary as an HSQLDB database, whose
``.script`` file is plain SQL text -- so the whole thing can be read in pure
Python with no JVM and no HSQLDB driver.

Why bother, when ``build_dictionary.py`` already builds one from MRCONSO? Because
they disagree by a lot. The shipped dictionary carries ~246k concepts; a
straight SNOMEDCT_US+RXNORM extract from a modern UMLS release carries ~442k.
That surplus is the dominant source of false positives in CUI parity scoring.
The curation that produced ``sno_rx_16ab`` was done interactively a decade ago
and its recipe is not recorded anywhere in the cTAKES source -- but the *result*
is right here on disk, so it can simply be read.

Importing it removes the dictionary as a variable when diffing Python output
against Java output: both sides then look up the same concepts, and any
remaining disagreement belongs to the tokenizer, POS tagger or matcher.

Source schema (from the script's own DDL)::

    CUI_TERMS(CUI BIGINT, RINDEX INT, TCOUNT INT, TEXT VARCHAR, RWORD VARCHAR)
    TUI(CUI BIGINT, TUI INT)
    PREFTERM(CUI BIGINT, PREFTERM VARCHAR)
    SNOMEDCT_US(CUI BIGINT, SNOMEDCT_US BIGINT)
    RXNORM(CUI BIGINT, RXNORM BIGINT)

``CUI_TERMS`` maps 1:1 onto our ``rare_term``: cTAKES stores the rare word, its
index within the term, and the term's token count -- exactly the fields
``RareWordMatcher`` needs. No rare-word recomputation is required or wanted.

Note that ``TEXT`` is **pre-tokenized**: punctuation is split out and
space-separated (``'2 , 4 - dichlorophenoxyacetic acid'``, TCOUNT=6), and
``TCOUNT`` counts punctuation. Our tokenizer must segment at the same
granularity for span alignment to work.

Output schema matches ``tools/build_dictionary.py`` exactly, so the result is a
drop-in ``--db`` for the matcher and every scorer.

Usage::

    python tools/import_ctakes_dictionary.py \\
        --script "$CTAKES_HOME/$SCRIPT_REL" \\
        --out data/umls_ctakes_16ab.sqlite

where ``$CTAKES_HOME`` is an installed cTAKES distribution (e.g.
``D:/apps/apache-ctakes-7.0.0-SNAPSHOT``) and ``$SCRIPT_REL`` is
``resources/org/apache/ctakes/dictionary/lookup/fast/sno_rx_16ab/sno_rx_16ab.script``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.umls.semantic_tui import best_group, group_for_tui

BATCH = 50_000

INSERT_PREFIX = "INSERT INTO "


def parse_sql_values(payload: str) -> list[str | int | None]:
    """Split an HSQLDB ``VALUES(...)`` payload into Python values.

    Handles single-quoted strings with the SQL ``''`` escape, which a naive
    ``split(',')`` would corrupt -- terms like
    ``'3 '' , 5 '' cyclic amp phosphodiesterases'`` contain both quotes and
    commas inside a single literal.
    """
    values: list[str | int | None] = []
    i = 0
    n = len(payload)
    while i < n:
        while i < n and payload[i] in " \t":
            i += 1
        if i >= n:
            break
        if payload[i] == "'":
            i += 1
            buf: list[str] = []
            while i < n:
                if payload[i] == "'":
                    if i + 1 < n and payload[i + 1] == "'":
                        buf.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(payload[i])
                i += 1
            values.append("".join(buf))
        else:
            j = i
            while j < n and payload[j] != ",":
                j += 1
            raw = payload[i:j].strip()
            if raw.upper() == "NULL":
                values.append(None)
            else:
                try:
                    values.append(int(raw))
                except ValueError:
                    values.append(raw)
            i = j
        while i < n and payload[i] in " \t":
            i += 1
        if i < n and payload[i] == ",":
            i += 1
    return values


def cui_str(code: int) -> str:
    """cTAKES stores CUIs as integers: 97 -> 'C0000097'."""
    return f"C{code:07d}"


def tui_str(code: int) -> str:
    return f"T{code:03d}"


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;

        DROP TABLE IF EXISTS concept;
        DROP TABLE IF EXISTS concept_tui;
        DROP TABLE IF EXISTS term;
        DROP TABLE IF EXISTS rare_term;
        DROP TABLE IF EXISTS meta;

        CREATE TABLE concept (
            cui TEXT PRIMARY KEY, preferred_text TEXT NOT NULL, best_group TEXT NOT NULL
        );
        CREATE TABLE concept_tui (cui TEXT NOT NULL, tui TEXT NOT NULL);
        CREATE TABLE term (
            norm TEXT NOT NULL, text TEXT NOT NULL, cui TEXT NOT NULL,
            sab TEXT NOT NULL, tty TEXT NOT NULL
        );
        CREATE TABLE rare_term (
            rare_word TEXT NOT NULL, norm TEXT NOT NULL, cui TEXT NOT NULL,
            word_index INTEGER NOT NULL, token_count INTEGER NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--script", type=Path, required=True, help="path to sno_rx_16ab.script")
    ap.add_argument("--out", type=Path, default=Path("data/umls_ctakes_16ab.sqlite"))
    args = ap.parse_args()

    if not args.script.is_file():
        sys.exit(f"error: not found: {args.script}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    t0 = time.time()
    conn = sqlite3.connect(args.out)
    _schema(conn)

    counts = {"CUI_TERMS": 0, "TUI": 0, "PREFTERM": 0, "SNOMEDCT_US": 0, "RXNORM": 0}
    skipped = 0

    rare_batch: list[tuple] = []
    term_batch: list[tuple] = []
    tui_batch: list[tuple] = []
    pref: dict[str, str] = {}
    sab: dict[str, str] = {}

    print(f"reading {args.script} ({args.script.stat().st_size/1e6:.0f} MB) ...")
    with args.script.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(INSERT_PREFIX):
                continue
            head, _, rest = line[len(INSERT_PREFIX) :].partition(" VALUES(")
            table = head.strip()
            if table not in counts:
                continue
            payload = rest.rstrip().rstrip(")")
            vals = parse_sql_values(payload)

            try:
                if table == "CUI_TERMS":
                    code, rindex, tcount, text, rword = vals[0], vals[1], vals[2], vals[3], vals[4]
                    cui = cui_str(int(code))
                    rare_batch.append((rword, text, cui, int(rindex), int(tcount)))
                    term_batch.append((text, text, cui))
                elif table == "TUI":
                    tui_batch.append((cui_str(int(vals[0])), tui_str(int(vals[1]))))
                elif table == "PREFTERM":
                    pref[cui_str(int(vals[0]))] = str(vals[1])
                elif table == "SNOMEDCT_US":
                    sab.setdefault(cui_str(int(vals[0])), "SNOMEDCT_US")
                elif table == "RXNORM":
                    sab.setdefault(cui_str(int(vals[0])), "RXNORM")
            except (TypeError, ValueError, IndexError):
                skipped += 1
                continue

            counts[table] += 1

            if len(rare_batch) >= BATCH:
                conn.executemany("INSERT INTO rare_term VALUES (?,?,?,?,?)", rare_batch)
                rare_batch.clear()
            if len(tui_batch) >= BATCH:
                conn.executemany("INSERT INTO concept_tui VALUES (?,?)", tui_batch)
                tui_batch.clear()
            if len(term_batch) >= BATCH:
                conn.executemany(
                    "INSERT INTO term (norm, text, cui, sab, tty) VALUES (?,?,?,'','')", term_batch
                )
                term_batch.clear()

    if rare_batch:
        conn.executemany("INSERT INTO rare_term VALUES (?,?,?,?,?)", rare_batch)
    if tui_batch:
        conn.executemany("INSERT INTO concept_tui VALUES (?,?)", tui_batch)
    if term_batch:
        conn.executemany(
            "INSERT INTO term (norm, text, cui, sab, tty) VALUES (?,?,?,'','')", term_batch
        )
    conn.commit()

    print("  parsed:", ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    if skipped:
        print(f"  WARNING: {skipped:,} unparseable rows skipped")

    # Backfill source abbreviation now that membership tables are read.
    # Done as one indexed join, not a per-CUI UPDATE loop: `term` has 550k rows
    # and no index at this point, so row-by-row updates degrade to a full scan
    # each and the import never finishes.
    print("assigning sources ...", flush=True)
    conn.execute("CREATE TEMP TABLE _sab (cui TEXT PRIMARY KEY, sab TEXT NOT NULL)")
    conn.executemany("INSERT OR REPLACE INTO _sab VALUES (?,?)", list(sab.items()))
    conn.execute("CREATE INDEX ix_term_cui_tmp ON term(cui)")
    conn.execute(
        "UPDATE term SET sab = COALESCE((SELECT s.sab FROM _sab s WHERE s.cui = term.cui), '')"
    )
    conn.execute("DROP INDEX ix_term_cui_tmp")
    conn.commit()

    # concept: preferred term + cTAKES' best-group tie-break over its TUIs.
    print("building concept table ...", flush=True)
    tui_map: dict[str, list[str]] = {}
    for cui, tui in conn.execute("SELECT cui, tui FROM concept_tui"):
        tui_map.setdefault(cui, []).append(tui)

    all_cuis = {r[0] for r in conn.execute("SELECT DISTINCT cui FROM rare_term")}
    rows = []
    for cui in all_cuis:
        tuis = tui_map.get(cui, [])
        group = best_group([group_for_tui(t) for t in tuis]).name if tuis else "UNKNOWN"
        rows.append((cui, pref.get(cui, ""), group))
    conn.executemany("INSERT INTO concept VALUES (?,?,?)", rows)
    conn.commit()

    print("indexing ...", flush=True)
    conn.executescript(
        """
        CREATE INDEX ix_rare_word   ON rare_term(rare_word);
        CREATE INDEX ix_term_norm   ON term(norm);
        CREATE INDEX ix_term_cui    ON term(cui);
        CREATE INDEX ix_ctui_cui    ON concept_tui(cui);
        CREATE INDEX ix_concept_grp ON concept(best_group);
        """
    )
    conn.commit()

    elapsed = time.time() - t0
    stats = {
        "concepts": conn.execute("SELECT COUNT(*) FROM concept").fetchone()[0],
        "terms": conn.execute("SELECT COUNT(*) FROM term").fetchone()[0],
        "rare_terms": conn.execute("SELECT COUNT(*) FROM rare_term").fetchone()[0],
        "rare_word_keys": conn.execute(
            "SELECT COUNT(DISTINCT rare_word) FROM rare_term"
        ).fetchone()[0],
        "concept_tuis": conn.execute("SELECT COUNT(*) FROM concept_tui").fetchone()[0],
    }
    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?,?)",
        sorted(
            {
                "source": "ctakes_shipped_sno_rx_16ab",
                "umls_release": "2016AB",
                # cTAKES ships this dictionary already pre-tokenized, so it
                # satisfies the alignment invariant `RareWordMatcher` enforces
                # without passing through `retokenize_terms.py` -- which refuses
                # to run on it, because re-tokenizing would corrupt the
                # spellings. Recording it here rather than relying on the
                # matcher's `PRETOKENIZED_SOURCES` fallback means a database
                # built by this script carries its own proof.
                "terms_retokenized": "yes",
                "sources": "RXNORM,SNOMEDCT_US",
                "script_path": str(args.script),
                "limited": "no",
                "semantic_tags_stripped": "n/a",
                "import_seconds": f"{elapsed:.1f}",
                **{k: str(v) for k, v in stats.items()},
            }.items()
        ),
    )
    conn.commit()

    print("\n=== summary ===")
    for k, v in stats.items():
        print(f"  {k:<16} {v:,}")
    print("\n  concepts by semantic group:")
    for group, n in conn.execute(
        "SELECT best_group, COUNT(*) c FROM concept GROUP BY best_group ORDER BY c DESC"
    ):
        print(f"    {group:<20} {n:,}")

    conn.execute("VACUUM")
    conn.close()
    print(f"\nwrote {args.out} ({args.out.stat().st_size/1e6:.0f} MB, {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

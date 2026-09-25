#!/usr/bin/env python3
"""Build a cTAKES-equivalent UMLS lookup dictionary from MRCONSO/MRSTY.

Reproduces the filtering behind cTAKES' default ``sno_rx_16ab`` dictionary:

  * ``MRCONSO.RRF`` restricted to English, non-suppressed atoms from
    SNOMEDCT_US and RXNORM
  * ``MRSTY.RRF`` restricted to the 136 TUIs cTAKES maps to semantic groups
  * concepts with no mapped TUI are dropped entirely
  * terms whose entire text is a bare function word or hand-curated
    ambiguous term (e.g. "past", "date") are dropped -- ported from
    cTAKES' own dictionary-build tool, see ``umlsmatch.dictionary.exclusions``

Output is a SQLite database:

  ``concept``      cui, preferred_text, best_group
  ``concept_tui``  cui, tui
  ``term``         norm, text, cui, sab, tty   -- the matchable surface forms
  ``meta``         provenance: UMLS release, sources, counts, build time

Both input files are streamed; the CUI join is done by SQLite rather than in
Python, so peak memory stays flat regardless of release size.

Usage::

    python tools/build_dictionary.py \\
        --umls-dir "D:/xye/Medical.terms/umls-2021AB-metathesaurus/2021AB/META" \\
        --out data/umls_sno_rx.sqlite

Smoke-test against the first N lines of each file before committing to a full
run (which reads ~2.1 GB)::

    python tools/build_dictionary.py --umls-dir ... --out /tmp/t.sqlite --limit 200000
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.dictionary.exclusions import EXCLUDED_TERM_TEXTS
from umlsmatch.umls.ctakes_tuis import CTAKES_TUIS
from umlsmatch.umls.semantic_tui import (
    TUI_TO_GROUP,
    best_group,
    group_for_tui,
)

# --- MRCONSO.RRF / MRSTY.RRF column indices, per MRFILES.RRF -----------------
# MRCONSO: CUI,LAT,TS,LUI,STT,SUI,ISPREF,AUI,SAUI,SCUI,SDUI,SAB,TTY,CODE,STR,
#          SRL,SUPPRESS,CVF  (18 columns)
MRCONSO_CUI, MRCONSO_LAT, MRCONSO_ISPREF = 0, 1, 6
MRCONSO_SAB, MRCONSO_TTY, MRCONSO_STR, MRCONSO_SUPPRESS = 11, 12, 14, 16
MRCONSO_NCOLS = 18

# MRSTY: CUI,TUI,STN,STY,ATUI,CVF  (6 columns)
MRSTY_CUI, MRSTY_TUI = 0, 1
MRSTY_NCOLS = 6

# cTAKES' default dictionary sources -- what it RECORDS CODES from.
DEFAULT_SOURCES = ("SNOMEDCT_US", "RXNORM")

# What cTAKES' dictionary builder READS SYNONYMS from by default. Its GUI has
# two separate checkbox columns ("Read Synonyms" / "Record Codes"); this is the
# wider one. See ctakes-gui .../umls/SourceTableModel.java:CTAKES_SOURCES.
CTAKES_SYNONYM_SOURCES = ("SNOMEDCT_US", "RXNORM", "MTH", "MSH", "LNC", "CHV", "HPO")

BATCH = 50_000


def normalize(text: str) -> str:
    """Lookup normalization: casefold and collapse internal whitespace.

    Deliberately conservative. cTAKES additionally excludes lookup *anchors* by
    POS tag and enforces a minimum span, but those are matcher-time concerns --
    the dictionary keeps every surface form so the matcher stays free to decide.
    """
    return " ".join(text.casefold().split())


# SNOMED fully-specified names carry a trailing semantic tag, e.g.
# "Pneumonia (disorder)" or "Aspirin (substance)". These never occur in clinical
# narrative, so keeping them verbatim inflates the dictionary by ~a third and
# skews the rare-word token statistics that drive lookup. Strip the tag and keep
# the bare form, which IS matchable.
SEMANTIC_TAG_RE = re.compile(r"^(?P<base>.+?)\s*\([^()]{1,40}\)$")


def strip_semantic_tag(norm: str) -> str | None:
    """Return the tag-stripped form of a SNOMED FSN, or None if not applicable."""
    m = SEMANTIC_TAG_RE.match(norm)
    if not m:
        return None
    base = m.group("base").strip()
    # Guard against terms that are mostly tag, or whose parenthetical is part of
    # the concept name rather than a semantic tag.
    if len(base) < 3:
        return None
    return base


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS concept;
        DROP TABLE IF EXISTS concept_tui;
        DROP TABLE IF EXISTS term;
        DROP TABLE IF EXISTS meta;
        DROP TABLE IF EXISTS _sty_raw;
        DROP TABLE IF EXISTS _term_raw;
        DROP TABLE IF EXISTS _code_cui;

        CREATE TABLE _sty_raw  (cui TEXT NOT NULL, tui TEXT NOT NULL);
        -- CUIs carrying a code in one of the `code_sources` vocabularies.
        CREATE TABLE _code_cui (cui TEXT NOT NULL);
        CREATE TABLE _term_raw (
            cui  TEXT NOT NULL,
            norm TEXT NOT NULL,
            text TEXT NOT NULL,
            sab  TEXT NOT NULL,
            tty  TEXT NOT NULL,
            ispref TEXT NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )


def _tune(conn: sqlite3.Connection) -> None:
    # Bulk-load settings. This database is a build artifact regenerated from
    # source files, so durability during the build buys nothing.
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
        """
    )


def load_mrsty(
    conn: sqlite3.Connection,
    path: Path,
    limit: int | None,
    allowed_tuis: frozenset[str] | None = None,
) -> tuple[int, int]:
    """Stream MRSTY.RRF, keeping rows whose TUI is in `allowed_tuis`.

    Defaults to every TUI cTAKES maps to a semantic group (all 136). Passing
    ``CTAKES_TUIS`` narrows this to the 49 the shipped dictionary indexes, which
    is what makes a modern-release build behave like cTAKES.
    """
    allowed = allowed_tuis if allowed_tuis is not None else frozenset(TUI_TO_GROUP)
    read = kept = 0
    batch: list[tuple[str, str]] = []
    started = time.time()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if limit and read >= limit:
                break
            read += 1
            cols = line.rstrip("\n").split("|")
            if len(cols) < MRSTY_NCOLS:
                continue
            tui = cols[MRSTY_TUI]
            if tui not in allowed:
                continue
            batch.append((cols[MRSTY_CUI], tui))
            kept += 1
            if len(batch) >= BATCH:
                conn.executemany("INSERT INTO _sty_raw VALUES (?,?)", batch)
                batch.clear()
            if read % 1_000_000 == 0:
                print(f"  MRSTY  {read:>10,} read  {kept:>10,} kept  {time.time()-started:5.0f}s")

    if batch:
        conn.executemany("INSERT INTO _sty_raw VALUES (?,?)", batch)
    conn.commit()
    return read, kept


def load_mrconso(
    conn: sqlite3.Connection,
    path: Path,
    sources: set[str] | None,
    limit: int | None,
    strip_tags: bool = True,
    exclude_texts: bool = True,
    code_sources: set[str] | None = None,
) -> tuple[int, int, int, int]:
    """Stream MRCONSO.RRF, keeping English non-suppressed atoms.

    Two independent filters, mirroring the two checkbox columns in cTAKES'
    dictionary builder ("Read Synonyms" / "Record Codes"):

    * `sources` -- read *synonyms* from these vocabularies; ``None`` means every
      vocabulary in the release. More synonyms means more ways to match the same
      concept. It does **not** change which concepts exist.
    * `code_sources` -- keep a concept only if it carries a code in one of
      these. This is what decides the concept inventory. ``None`` disables the
      restriction, admitting concepts that exist only in vocabularies without a
      SNOMED/RxNorm code.

    Separating them matters: conflating the two means broadening synonyms also
    silently balloons the concept set.
    """
    read = kept = stripped = excluded = 0
    code_batch: list[tuple[str]] = []
    batch: list[tuple[str, str, str, str, str, str]] = []
    started = time.time()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if limit and read >= limit:
                break
            read += 1
            cols = line.rstrip("\n").split("|")
            if len(cols) < MRCONSO_NCOLS:
                continue
            if cols[MRCONSO_LAT] != "ENG":
                continue
            if cols[MRCONSO_SUPPRESS] != "N":
                continue

            sab = cols[MRCONSO_SAB]
            # Record code-bearing CUIs before the synonym filter: a concept
            # qualifies on its codes even when this particular atom's
            # vocabulary is not one we read synonyms from.
            if code_sources is not None and sab in code_sources:
                code_batch.append((cols[MRCONSO_CUI],))
                if len(code_batch) >= BATCH:
                    conn.executemany("INSERT INTO _code_cui VALUES (?)", code_batch)
                    code_batch.clear()

            if sources is not None and sab not in sources:
                continue
            text = cols[MRCONSO_STR].strip()
            if not text:
                continue
            norm = normalize(text)
            if not norm:
                continue
            # Replace the normalized form of a SNOMED fully-specified name with
            # its tag-stripped form. The tagged form is unmatchable in narrative
            # text, so indexing it is pure dead weight; `text` still carries the
            # original string, so provenance is not lost.
            if strip_tags and cols[MRCONSO_TTY] == "FN":
                base = strip_semantic_tag(norm)
                if base and base != norm:
                    norm = base
                    stripped += 1

            # Drop terms whose entire normalized text is a bare function word or
            # a hand-curated ambiguous term (e.g. "past", "date") -- ported from
            # cTAKES' own dictionary-build tool; see umlsmatch.dictionary.exclusions.
            if exclude_texts and norm in EXCLUDED_TERM_TEXTS:
                excluded += 1
                continue

            batch.append(
                (
                    cols[MRCONSO_CUI],
                    norm,
                    text,
                    cols[MRCONSO_SAB],
                    cols[MRCONSO_TTY],
                    cols[MRCONSO_ISPREF],
                )
            )
            kept += 1
            if len(batch) >= BATCH:
                conn.executemany("INSERT INTO _term_raw VALUES (?,?,?,?,?,?)", batch)
                batch.clear()
            if read % 2_000_000 == 0:
                print(f"  MRCONSO {read:>10,} read  {kept:>10,} kept  {time.time()-started:5.0f}s")

    if batch:
        conn.executemany("INSERT INTO _term_raw VALUES (?,?,?,?,?,?)", batch)
    if code_batch:
        conn.executemany("INSERT INTO _code_cui VALUES (?)", code_batch)
    conn.commit()
    return read, kept, stripped, excluded


def join_and_finalize(conn: sqlite3.Connection) -> dict[str, int]:
    """Drop concepts lacking a mapped TUI, assign best group, build final tables."""
    print("  indexing staging tables ...")
    conn.executescript(
        """
        CREATE INDEX ix_sty_cui  ON _sty_raw(cui);
        CREATE INDEX ix_term_cui ON _term_raw(cui);
        CREATE INDEX ix_code_cui ON _code_cui(cui);
        """
    )
    conn.commit()

    # Only concepts present in BOTH staging tables survive: a term with no
    # cTAKES-mapped semantic type is not a clinical concept as far as the
    # pipeline is concerned.
    print("  building concept_tui ...")
    conn.executescript(
        """
        CREATE TABLE concept_tui AS
        SELECT DISTINCT s.cui, s.tui
        FROM _sty_raw s
        WHERE EXISTS (SELECT 1 FROM _term_raw t WHERE t.cui = s.cui)
          AND (NOT EXISTS (SELECT 1 FROM _code_cui LIMIT 1)
               OR EXISTS (SELECT 1 FROM _code_cui c WHERE c.cui = s.cui));

        CREATE INDEX ix_ctui_cui ON concept_tui(cui);
        """
    )
    conn.commit()

    print("  building term ...")
    conn.executescript(
        """
        CREATE TABLE term AS
        SELECT DISTINCT t.norm, t.text, t.cui, t.sab, t.tty
        FROM _term_raw t
        WHERE EXISTS (SELECT 1 FROM concept_tui c WHERE c.cui = t.cui);

        CREATE INDEX ix_term_norm ON term(norm);
        CREATE INDEX ix_term_cui2 ON term(cui);
        """
    )
    conn.commit()

    # Preferred text: prefer an ISPREF='Y' atom, else the shortest surface form,
    # which is a reasonable stand-in for a canonical label.
    print("  building concept ...")
    conn.executescript(
        """
        CREATE TABLE concept (
            cui            TEXT PRIMARY KEY,
            preferred_text TEXT NOT NULL,
            best_group     TEXT NOT NULL
        );

        INSERT INTO concept (cui, preferred_text, best_group)
        SELECT r.cui,
               (SELECT t.text FROM _term_raw t
                 WHERE t.cui = r.cui
                 ORDER BY (t.ispref='Y') DESC, LENGTH(t.text) ASC, t.text ASC
                 LIMIT 1),
               ''
        FROM (SELECT DISTINCT cui FROM concept_tui) r;
        """
    )
    conn.commit()

    # best_group is cTAKES' multi-TUI tie-break; compute it in Python so the
    # generated SemanticTui tables stay the single source of truth.
    print("  assigning best_group ...")
    cur = conn.execute("SELECT cui, GROUP_CONCAT(tui) FROM concept_tui GROUP BY cui")
    updates: list[tuple[str, str]] = []
    for cui, tuis in cur:
        groups = [group_for_tui(t) for t in tuis.split(",")]
        updates.append((best_group(groups).name, cui))
        if len(updates) >= BATCH:
            conn.executemany("UPDATE concept SET best_group=? WHERE cui=?", updates)
            updates.clear()
    if updates:
        conn.executemany("UPDATE concept SET best_group=? WHERE cui=?", updates)
    conn.commit()

    conn.executescript(
        """
        CREATE INDEX ix_concept_group ON concept(best_group);
        DROP TABLE _sty_raw;
        DROP TABLE _term_raw;
        DROP TABLE _code_cui;
        """
    )
    conn.commit()

    counts = {
        "concepts": conn.execute("SELECT COUNT(*) FROM concept").fetchone()[0],
        "terms": conn.execute("SELECT COUNT(*) FROM term").fetchone()[0],
        "distinct_norms": conn.execute("SELECT COUNT(DISTINCT norm) FROM term").fetchone()[0],
        "concept_tuis": conn.execute("SELECT COUNT(*) FROM concept_tui").fetchone()[0],
    }
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--umls-dir", type=Path, required=True, help="UMLS META directory")
    ap.add_argument("--out", type=Path, default=Path("data/umls_sno_rx.sqlite"))
    ap.add_argument(
        "--sources",
        default="all",
        help=(
            "vocabularies to READ SYNONYMS from (default: all). Reading widely "
            "raises recall a lot: on a real-note corpus not included here, 2 sources gave R=0.610 "
            "and all sources R=0.863. cTAKES' own builder reads from "
            f"{','.join(CTAKES_SYNONYM_SOURCES)}, which scored the best F1 "
            "(0.806 vs 0.775) -- pass that list explicitly to prefer precision."
        ),
    )
    ap.add_argument(
        "--code-sources",
        default=",".join(DEFAULT_SOURCES),
        help=(
            "a concept is kept only if it carries a code in one of these; 'any' "
            f"disables the restriction (default: {','.join(DEFAULT_SOURCES)}). "
            "This decides the concept inventory -- widening --sources alone adds "
            "synonyms without adding concepts."
        ),
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="read only N lines per file (smoke test)"
    )
    ap.add_argument(
        "--keep-semantic-tags",
        action="store_true",
        help=(
            "index SNOMED fully-specified names as-is. By default an FSN's "
            "lookup form REPLACES the trailing semantic tag, so 'Pneumonia "
            "(disorder)' is indexed as 'pneumonia'; the tagged spelling never "
            "occurs in narrative. The original string is kept in term.text "
            "either way."
        ),
    )
    ap.add_argument(
        "--keep-excluded-texts",
        action="store_true",
        help="do not drop terms matching umlsmatch.dictionary.exclusions.EXCLUDED_TERM_TEXTS",
    )
    ap.add_argument(
        "--tui-set",
        choices=("ctakes", "all"),
        default="ctakes",
        help=(
            "'ctakes' (default): only the 49 TUIs cTAKES' shipped dictionary indexes -- "
            "far higher precision. 'all': every TUI mapped to a semantic group (136), "
            "which yields ~1.8x the concepts and many generic false positives."
        ),
    )
    args = ap.parse_args()

    mrconso = args.umls_dir / "MRCONSO.RRF"
    mrsty = args.umls_dir / "MRSTY.RRF"
    for f in (mrconso, mrsty):
        if not f.is_file():
            sys.exit(f"error: not found: {f}")

    sources = (
        None
        if args.sources.strip().lower() == "all"
        else {s.strip().upper() for s in args.sources.split(",") if s.strip()}
    )
    code_sources = (
        None
        if args.code_sources.strip().lower() == "any"
        else {s.strip().upper() for s in args.code_sources.split(",") if s.strip()}
    )
    release = args.umls_dir.resolve().parent.name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    print(f"UMLS dir : {args.umls_dir}")
    print(f"release  : {release}")
    print(f"synonyms : {'ALL vocabularies' if sources is None else ', '.join(sorted(sources))}")
    print(f"codes    : {'any' if code_sources is None else ', '.join(sorted(code_sources))}")
    print(f"output   : {args.out}")
    if args.limit:
        print(f"LIMIT    : {args.limit:,} lines/file (smoke test -- output is NOT complete)")
    print()

    t0 = time.time()
    conn = sqlite3.connect(args.out)
    _tune(conn)
    _schema(conn)

    allowed_tuis = CTAKES_TUIS if args.tui_set == "ctakes" else frozenset(TUI_TO_GROUP)

    print("loading MRSTY.RRF ...")
    sty_read, sty_kept = load_mrsty(conn, mrsty, args.limit, allowed_tuis)
    print(
        f"  {sty_read:,} read, {sty_kept:,} kept "
        f"(tui-set={args.tui_set}, {len(allowed_tuis)} TUIs)\n"
    )

    print("loading MRCONSO.RRF ...")
    con_read, con_kept, con_stripped, con_excluded = load_mrconso(
        conn,
        mrconso,
        sources,
        args.limit,
        strip_tags=not args.keep_semantic_tags,
        exclude_texts=not args.keep_excluded_texts,
        code_sources=code_sources,
    )
    print(
        f"  {con_read:,} read, {con_kept:,} kept "
        f"({con_stripped:,} FSN tag-stripped forms, {con_excluded:,} excluded texts dropped)\n"
    )

    print("joining ...")
    counts = join_and_finalize(conn)

    elapsed = time.time() - t0
    meta = {
        "umls_release": release,
        "sources": "ALL" if sources is None else ",".join(sorted(sources)),
        "code_sources": "ANY" if code_sources is None else ",".join(sorted(code_sources)),
        "tui_set": args.tui_set,
        "tui_count": str(len(allowed_tuis)),
        "mrsty_rows_read": str(sty_read),
        "mrsty_rows_kept": str(sty_kept),
        "mrconso_rows_read": str(con_read),
        "mrconso_rows_kept": str(con_kept),
        "fsn_tags_stripped": str(con_stripped),
        "semantic_tags_stripped": "no" if args.keep_semantic_tags else "yes",
        "excluded_texts_dropped": str(con_excluded),
        "excluded_texts_applied": "no" if args.keep_excluded_texts else "yes",
        "build_seconds": f"{elapsed:.1f}",
        "limited": "yes" if args.limit else "no",
        **{k: str(v) for k, v in counts.items()},
    }
    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", sorted(meta.items()))
    conn.commit()

    print("\n=== summary ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:<16} {v:,}")

    print("\n  concepts by semantic group:")
    for group, n in conn.execute(
        "SELECT best_group, COUNT(*) c FROM concept GROUP BY best_group ORDER BY c DESC"
    ):
        print(f"    {group:<20} {n:,}")

    conn.execute("VACUUM")
    conn.close()

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:,.0f} MB, {elapsed:.0f}s)")
    if args.limit:
        print("NOTE: --limit was set; this dictionary is a partial smoke-test artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

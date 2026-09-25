#!/usr/bin/env python3
"""Add cTAKES' rare-word lookup index to the built UMLS dictionary.

cTAKES does not use an Aho-Corasick automaton. ``RareWordTermMapCreator``
indexes every dictionary term by its *rarest* constituent token, storing that
token's offset within the term and the term's token count. At lookup time each
eligible token in the window fetches candidate terms keyed by that token, and
each candidate is verified by aligning it against the surrounding tokens using
the stored offset.

The payoff is that the index is an ordinary B-tree (here, a SQLite index)
rather than a resident automaton, so lookup memory stays flat.

This script reproduces that build:

  * token counts over all dictionary terms, skipping tokens that are not
    "rarable" (length <= 1, no letter, or a known function word)
  * per term, the rarable token with the lowest corpus count wins; ties go to
    the earliest such token, matching the Java ``<`` comparison
  * single-token terms index on their only token regardless of rarability

``BAD_POS_TERMS`` is vendored in ``umlsmatch.dictionary.exclusions`` as
``RARE_WORD_BAD_POS_TERMS``, transcribed line-for-line from the Java in the
style the TUI tables use -- so building an index does not need a clone of
Apache cTAKES just to read a list of 114 function words. Pass ``--ctakes-root``
to re-parse the Java and fail if the two have drifted apart.

Note that this is *not* ``exclusions.BAD_POS_TERM_SET``, which comes from the
GUI build tool and is a larger superset; see the comment on either constant.

Usage::

    python tools/build_rare_word_index.py --db data/umls_sno_rx.sqlite
    python tools/build_rare_word_index.py --db data/umls_sno_rx.sqlite \\
        --ctakes-root ../ctakes-java        # additionally verify against Java
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.dictionary.exclusions import RARE_WORD_BAD_POS_TERMS

RARE_WORD_JAVA = (
    "ctakes-dictionary-lookup-fast/src/main/java/org/apache/ctakes/"
    "dictionary/lookup2/dictionary/RareWordTermMapCreator.java"
)

BATCH = 50_000


def parse_bad_pos_terms(java_src: str) -> frozenset[str]:
    """Extract the BAD_POS_TERMS function-word list from the Java source."""
    m = re.search(
        r"BAD_POS_TERMS\s*=\s*new\s+HashSet<>\(\s*Arrays\.asList\((?P<body>.*?)\)\s*\)\s*;",
        java_src,
        re.DOTALL,
    )
    if not m:
        sys.exit("error: could not locate BAD_POS_TERMS in RareWordTermMapCreator.java")
    terms = frozenset(re.findall(r'"([^"]*)"', m.group("body")))
    if not terms:
        sys.exit("error: BAD_POS_TERMS parsed but empty")
    return terms


def is_rarable(token: str, bad_pos_terms: frozenset[str]) -> bool:
    """Port of ``RareWordTermMapCreator.isRarableToken``."""
    if len(token) <= 1:
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return token not in bad_pos_terms


def choose_rare_word(
    tokens: list[str], counts: dict[str, int], bad_pos_terms: frozenset[str]
) -> tuple[str, int]:
    """Return (rare_word, word_index), porting ``getRareWord`` + ``getWordIndex``.

    Java seeds ``bestWord = tokens[0]`` with ``bestCount = MAX_VALUE`` and keeps
    a candidate only on strictly ``<``, so the earliest minimum wins and a term
    with no rarable token falls back to its first token.
    """
    if len(tokens) == 1:
        return tokens[0], 0

    best_word = tokens[0]
    best_count = None
    for token in tokens:
        if not is_rarable(token, bad_pos_terms):
            continue
        count = counts.get(token)
        if count is None:
            continue
        if best_count is None or count < best_count:
            best_word = token
            best_count = count

    # getWordIndex returns the FIRST positional match for the chosen word.
    return best_word, tokens.index(best_word)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/umls_sno_rx.sqlite"))
    ap.add_argument(
        "--ctakes-root", type=Path, default=None, metavar="DIR",
        help="clone of apache/ctakes. Optional: BAD_POS_TERMS is vendored in "
             "umlsmatch.dictionary.exclusions, and passing this re-parses the "
             "Java and fails if the two have diverged.",
    )
    args = ap.parse_args()

    if not args.db.is_file():
        sys.exit(f"error: dictionary not found: {args.db} (run tools/build_dictionary.py first)")

    bad_pos_terms = RARE_WORD_BAD_POS_TERMS
    print(f"BAD_POS_TERMS: {len(bad_pos_terms)} function words (vendored)")

    if args.ctakes_root is not None:
        java_path = args.ctakes_root / RARE_WORD_JAVA
        if not java_path.is_file():
            sys.exit(f"error: cTAKES source not found: {java_path}")
        from_java = parse_bad_pos_terms(java_path.read_text(encoding="utf-8"))
        if from_java != bad_pos_terms:
            only_java = sorted(from_java - bad_pos_terms)
            only_here = sorted(bad_pos_terms - from_java)
            sys.exit(
                "error: vendored BAD_POS_TERMS has diverged from the Java source.\n"
                f"  only in Java:     {only_java}\n"
                f"  only in vendored: {only_here}\n"
                "  Update umlsmatch.dictionary.exclusions.RARE_WORD_BAD_POS_TERMS."
            )
        print(f"  verified against {java_path}")

    t0 = time.time()
    conn = sqlite3.connect(args.db)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
        DROP TABLE IF EXISTS rare_term;
        CREATE TABLE rare_term (
            rare_word   TEXT NOT NULL,
            norm        TEXT NOT NULL,
            cui         TEXT NOT NULL,
            word_index  INTEGER NOT NULL,
            token_count INTEGER NOT NULL
        );
        """
    )

    # Pass 1 -- corpus token counts over distinct terms.
    # cTAKES counts per CuiTerm; we count per distinct normalized surface form so
    # the statistic is not skewed by a term repeated across many CUIs.
    print("counting tokens ...")
    counts: dict[str, int] = {}
    n_terms = 0
    for (norm,) in conn.execute("SELECT DISTINCT norm FROM term"):
        n_terms += 1
        for token in norm.split(" "):
            if is_rarable(token, bad_pos_terms):
                counts[token] = counts.get(token, 0) + 1
    print(f"  {n_terms:,} distinct terms, {len(counts):,} rarable token types")

    # Pass 2 -- assign each (norm, cui) pair its rare word.
    print("assigning rare words ...")
    batch: list[tuple[str, str, str, int, int]] = []
    rows = 0
    fallback = 0
    for norm, cui in conn.execute("SELECT DISTINCT norm, cui FROM term"):
        tokens = norm.split(" ")
        rare_word, word_index = choose_rare_word(tokens, counts, bad_pos_terms)
        if len(tokens) > 1 and not is_rarable(rare_word, bad_pos_terms):
            fallback += 1
        batch.append((rare_word, norm, cui, word_index, len(tokens)))
        rows += 1
        if len(batch) >= BATCH:
            conn.executemany("INSERT INTO rare_term VALUES (?,?,?,?,?)", batch)
            batch.clear()
    if batch:
        conn.executemany("INSERT INTO rare_term VALUES (?,?,?,?,?)", batch)
    conn.commit()

    print("  indexing ...")
    conn.execute("CREATE INDEX ix_rare_word ON rare_term(rare_word)")
    conn.commit()

    elapsed = time.time() - t0
    distinct_keys = conn.execute("SELECT COUNT(DISTINCT rare_word) FROM rare_term").fetchone()[0]
    worst = conn.execute(
        "SELECT rare_word, COUNT(*) c FROM rare_term GROUP BY rare_word ORDER BY c DESC LIMIT 5"
    ).fetchall()
    avg = rows / distinct_keys if distinct_keys else 0

    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?,?)",
        [
            ("rare_terms", str(rows)),
            ("rare_word_keys", str(distinct_keys)),
            ("bad_pos_terms", str(len(bad_pos_terms))),
            ("rare_index_seconds", f"{elapsed:.1f}"),
        ],
    )
    conn.commit()

    print("\n=== summary ===")
    print(f"  rare_term rows        {rows:,}")
    print(f"  distinct rare words   {distinct_keys:,}")
    print(f"  avg candidates/key    {avg:.2f}")
    print(f"  no-rarable fallbacks  {fallback:,}")
    print("  worst-case keys:")
    for word, c in worst:
        print(f"    {word:<24} {c:,}")
    print(f"\ndone in {elapsed:.0f}s")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

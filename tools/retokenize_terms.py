#!/usr/bin/env python3
"""Align dictionary term spellings with the tokenizer used at match time.

``RareWordMatcher`` verifies a candidate by joining the *document's* token norms
with single spaces and comparing to the dictionary's stored ``norm``. So a term
only ever matches if the dictionary spells it the way the tokenizer segments it.

``build_dictionary.py`` normalizes with casefold + whitespace collapse only, so
it stores ``"chest x-ray"``. spaCy tokenizes that text as
``['chest', 'x', '-', 'ray']`` -> ``"chest x - ray"``. Those never compare equal,
so every hyphenated term is dead on arrival -- ~96k terms, roughly 10% of a
2026AA build.

cTAKES sidesteps this by storing **pre-tokenized** text in its own dictionary
(``'2 , 4 - dichlorophenoxyacetic acid'``). This script does the same thing for
ours, but derives the spelling from the very tokenizer the matcher will use, so
alignment holds by construction rather than by matching cTAKES' tokenizer rules.

Run after ``build_dictionary.py`` and *before* ``build_rare_word_index.py`` --
rare-word statistics must be computed over the final spellings.

Usage::

    python tools/build_dictionary.py --umls-dir <META>
    python tools/retokenize_terms.py --db data/umls_sno_rx.sqlite
    python tools/build_rare_word_index.py --db data/umls_sno_rx.sqlite --ctakes-root ../ctakes-java

Requires the optional ``nlp`` extra (spaCy). The shipped-cTAKES dictionary
imported by ``import_ctakes_dictionary.py`` is already pre-tokenized and must
NOT be passed through this.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BATCH = 20_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=Path("data/umls_sno_rx.sqlite"))
    ap.add_argument("--model", default=None, help="spaCy model (default: tokenizer default)")
    ap.add_argument("--force", action="store_true", help="re-run even if already re-tokenized")
    ap.add_argument(
        "--max-passes",
        type=int,
        default=4,
        help="iterate to a tokenization fixed point, up to this many passes (default: 4)",
    )
    args = ap.parse_args()

    if not args.db.is_file():
        sys.exit(f"error: {args.db} not found -- run tools/build_dictionary.py first")

    try:
        from umlsmatch.pipeline.tokenizer import DEFAULT_MODEL, load_model
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        sys.exit(f"error: spaCy not available ({exc}). Install the 'nlp' extra.")

    conn = sqlite3.connect(args.db)
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    if meta.get("source") == "ctakes_shipped_sno_rx_16ab":
        sys.exit(
            "error: this database is cTAKES' shipped dictionary, which is already "
            "pre-tokenized. Re-tokenizing it would corrupt the spellings."
        )
    if meta.get("terms_retokenized") == "yes" and not args.force:
        print("already re-tokenized; nothing to do (pass --force to re-run).")
        return 0

    nlp = load_model(args.model or DEFAULT_MODEL)
    tokenizer = nlp.tokenizer  # tokenizer only: no tagger/parser needed here

    norms = [r[0] for r in conn.execute("SELECT DISTINCT norm FROM term")]
    print(f"re-tokenizing {len(norms):,} distinct terms ...", flush=True)

    def retokenize(values: list[str]) -> list[str]:
        return [
            " ".join(t.text for t in doc if not t.is_space)
            for doc in tokenizer.pipe(values, batch_size=2000)
        ]

    # A single pass is not always a fixed point: re-tokenizing its own output can
    # shift again (e.g. ".alpha . ,.beta . '-diglycerin"). Since the matcher
    # compares against token-join spelling, anything short of a fixed point
    # leaves terms permanently unmatchable -- so iterate until stable.
    t0 = time.time()
    current = list(norms)
    for p in range(1, args.max_passes + 1):
        nxt = retokenize(current)
        # strict: `retokenize` returns one spelling per input, so a length
        # mismatch means the tokenizer dropped a term -- which would silently
        # misalign every later pair and corrupt the remap.
        moved = sum(1 for a, b in zip(current, nxt, strict=True) if a != b)
        print(f"  pass {p}: {moved:,} changed ({time.time()-t0:.0f}s)", flush=True)
        current = nxt
        if moved == 0:
            break
    else:
        unstable = sum(1 for a, b in zip(current, retokenize(current), strict=True) if a != b)
        if unstable:
            print(f"  WARNING: {unstable:,} terms still unstable after {args.max_passes} passes")

    mapping = [(new, old) for old, new in zip(norms, current, strict=True) if new and new != old]
    changed = len(mapping)
    print(f"  {changed:,} spellings changed ({100*changed/max(len(norms),1):.1f}%)", flush=True)

    print("applying ...", flush=True)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TEMP TABLE _remap (new TEXT NOT NULL, old TEXT PRIMARY KEY);
        """
    )
    for i in range(0, len(mapping), BATCH):
        conn.executemany("INSERT OR REPLACE INTO _remap VALUES (?,?)", mapping[i : i + BATCH])
    conn.commit()

    conn.execute(
        "UPDATE term SET norm = COALESCE("
        "(SELECT r.new FROM _remap r WHERE r.old = term.norm), norm)"
    )
    conn.commit()

    # rare_term is derived from norms; force a rebuild rather than leave it stale.
    conn.execute("DROP TABLE IF EXISTS rare_term")
    conn.execute(
        "INSERT OR REPLACE INTO meta VALUES ('terms_retokenized','yes'), "
        "('retokenize_model',?), ('retokenize_changed',?)",
        (args.model or DEFAULT_MODEL, str(changed)),
    )
    conn.commit()

    distinct = conn.execute("SELECT COUNT(DISTINCT norm) FROM term").fetchone()[0]
    print(f"\ndistinct norms now {distinct:,}  ({time.time()-t0:.0f}s)")
    print("rare_term dropped -- re-run tools/build_rare_word_index.py")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

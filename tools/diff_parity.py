#!/usr/bin/env python3
"""Show which CUIs the Python matcher over/under-generates vs. Java cTAKES.

Aggregates false positives (Python found a CUI, Java didn't) and false
negatives (Java found a CUI, Python didn't) across a silver-standard corpus,
ranked by frequency, with example matched text and -- for false positives --
the POS tags spaCy assigned across the matched span. Complements
tools/score_parity.py, which only reports aggregate P/R/F1; this answers
"on which concepts" rather than "how much".

Usage::

    python tools/diff_parity.py --jsonl free_texts/json/silver.jsonl --top 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.diff import CuiEvidence, diff_corpus, top


def _print_section(evidence: list[CuiEvidence], title: str) -> None:
    print(f"=== {title} ({len(evidence)} shown) ===")
    for e in evidence:
        label = e.preferred_text or "(no preferred_text)"
        print(f"{e.cui:10s} n={e.count:4d} docs={len(e.docs):3d}  {label}")
        # strict: CuiEvidence.add appends to both lists together, so unequal
        # lengths mean that invariant broke.
        for text, pos in zip(e.example_texts, e.example_pos, strict=True):
            pos_str = f"  pos={list(pos)}" if pos else ""
            print(f"    e.g. {text!r}{pos_str}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Silver-standard file (JSONL)")
    parser.add_argument(
        "--db", type=Path, default=Path("data/umls_sno_rx.sqlite"),
        help="Python UMLS dictionary SQLite DB (default: data/umls_sno_rx.sqlite)",
    )
    parser.add_argument("--top", type=int, default=20, help="How many CUIs to show per section")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        sys.exit(f"error: not found: {args.jsonl}")

    false_positives, false_negatives = diff_corpus(args.jsonl, args.db)

    print(
        f"{len(false_positives)} distinct false-positive CUIs, "
        f"{len(false_negatives)} distinct false-negative CUIs\n"
    )
    _print_section(top(false_positives, args.top), "FALSE POSITIVES (python found, java didn't)")
    _print_section(top(false_negatives, args.top), "FALSE NEGATIVES (java found, python didn't)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

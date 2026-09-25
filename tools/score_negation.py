#!/usr/bin/env python3
"""Score negation agreement: Python rule-based negation vs. Java cTAKES polarity.

Measures the negation target README's Validation section sets
("(greater-eq)90% F1 vs. cTAKES"). See ``umlsmatch.eval.negation`` for the
scoring definition -- restricted to CUIs both sides found in a document, so
this isolates polarity agreement from the separate CUI-matching problem
``tools/score_parity.py`` already characterizes.

Usage::

    python tools/score_negation.py --jsonl free_texts/json/silver.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.negation import aggregate, score_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Silver-standard file (JSONL)")
    parser.add_argument(
        "--db", type=Path, default=Path("data/umls_sno_rx.sqlite"),
        help="Python UMLS dictionary SQLite DB (default: data/umls_sno_rx.sqlite)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-document lines")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        sys.exit(f"error: not found: {args.jsonl}")

    scores = score_jsonl(args.jsonl, args.db)
    if not scores:
        sys.exit(f"error: no records scored from {args.jsonl}")

    if not args.quiet:
        for s in scores:
            print(
                f"{s.source_file:45s} P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f} "
                f"acc={s.accuracy:.3f}  (common={len(s.common_cuis)} "
                f"gold_neg={len(s.gold_negated)} "
                f"py_neg={len(s.python_negated)} tp={len(s.true_positives)})"
            )
        print()

    agg = aggregate(scores)
    print(f"{agg.n_docs} documents, {agg.n_common_cuis} commonly-found CUIs")
    print(
        f"micro  P={agg.micro_precision:.3f} R={agg.micro_recall:.3f} F1={agg.micro_f1:.3f}  "
        f"accuracy={agg.micro_accuracy:.3f}   (target: >=90% F1)"
    )
    print(f"macro  P={agg.macro_precision:.3f} R={agg.macro_recall:.3f} F1={agg.macro_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

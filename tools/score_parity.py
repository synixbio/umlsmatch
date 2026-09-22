#!/usr/bin/env python3
"""Score the Python dictionary matcher against Java cTAKES silver-standard output.

This is the parity harness README's Validation section sets the
concept-extraction target on, scoped to CUI-level entity extraction.
See ``umlsmatch.eval.parity`` for the scoring definition and its
caveats (UMLS release mismatch, general-domain vs. clinical POS tagger).

Usage::

    python tools/score_parity.py --jsonl free_texts/json/silver.jsonl

    # with a non-default dictionary build:
    python tools/score_parity.py --jsonl free_texts/json/silver.jsonl --db data/umls_sno_rx.sqlite

Silver-standard JSONL comes from ``tools/run_java_ctakes.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.parity import aggregate, score_jsonl


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
                f"{s.source_file:45s} P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}  "
                f"(silver={len(s.silver_cuis)} python={len(s.python_cuis)} "
                f"tp={len(s.true_positives)})"
            )
        print()

    agg = aggregate(scores)
    print(f"{agg.n_docs} documents")
    print(f"micro  P={agg.micro_precision:.3f} R={agg.micro_recall:.3f} F1={agg.micro_f1:.3f}")
    print(f"macro  P={agg.macro_precision:.3f} R={agg.macro_recall:.3f} F1={agg.macro_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score negation per *mention* instead of per document-collapsed CUI.

``tools/score_negation.py`` collapses polarity to one boolean per CUI per
document. This scores each gold mention-concept occurrence against the Python
match aligned to it (same CUI, overlapping span), so a difference in how many
mentions each side emits cannot move the number.

Run both and compare: if precision here is materially higher, the document
collapse is inflating the false-positive count and rule tuning aimed at the
per-document figure is aimed at an artifact. See
``umlsmatch.eval.negation_spans``.

Usage::

    python tools/score_negation_spans.py --jsonl free_texts/json/silver.jsonl
    python tools/score_negation_spans.py --jsonl free_texts/json/silver.jsonl \\
        --db data/umls_ctakes_16ab.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.negation_spans import aggregate, score_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", type=Path, required=True, help="silver-standard JSONL")
    ap.add_argument(
        "--db", type=Path, default=Path("data/umls_sno_rx.sqlite"),
        help="dictionary SQLite DB (default: data/umls_sno_rx.sqlite)",
    )
    ap.add_argument(
        "--max-scope", type=int, default=None, metavar="N",
        help="override the negation scope cap, in tokens",
    )
    ap.add_argument("--quiet", action="store_true", help="suppress per-document lines")
    args = ap.parse_args()

    if not args.jsonl.is_file():
        sys.exit(f"error: not found: {args.jsonl}")
    if not args.db.is_file():
        sys.exit(f"error: not found: {args.db}")

    kwargs = {} if args.max_scope is None else {"max_scope": args.max_scope}
    scores = score_jsonl(args.jsonl, args.db, **kwargs)
    if not scores:
        sys.exit(f"error: no records scored from {args.jsonl}")

    if not args.quiet:
        for s in scores:
            print(
                f"{Path(s.source_file).name:<45} P={s.precision:.3f} R={s.recall:.3f} "
                f"F1={s.f1:.3f} acc={s.accuracy:.3f}  "
                f"(aligned={s.aligned} unaligned={s.unaligned} "
                f"gold_neg={len(s.gold_negated)} py_neg={len(s.python_negated)})"
            )
        print()

    agg = aggregate(scores)
    print(f"{agg.n_docs} documents, {agg.n_aligned:,} aligned mention-concepts")
    print(
        f"  ({agg.n_unaligned:,} gold occurrences had no Python counterpart -- "
        "concept recall, not polarity; see tools/score_parity.py)"
    )
    print(
        f"micro  P={agg.micro_precision:.3f} R={agg.micro_recall:.3f} "
        f"F1={agg.micro_f1:.3f}  accuracy={agg.micro_accuracy:.3f}   (target: >=90% F1)"
    )
    print(
        f"macro  P={agg.macro_precision:.3f} R={agg.macro_recall:.3f} F1={agg.macro_f1:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Show which negation flags are wrong and which trigger phrase caused each.

``tools/score_negation_spans.py`` reports that ~half of all negation flags are
wrong. This decomposes that number by trigger phrase, which is what decides
where the remaining work goes: a few phrases dominating means fix the lexicon,
errors spread thin across many means the rules are at their ceiling and the
trained classifier is the honest path. See ``umlsmatch.eval.negation_diff``.

Usage::

    python tools/diff_negation.py --jsonl free_texts/json/silver.jsonl
    python tools/diff_negation.py --jsonl free_texts/json/silver.jsonl \\
        --db data/umls_ctakes_16ab.sqlite --top 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.negation_diff import collect_errors, top_false_negatives, top_triggers


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
    ap.add_argument("--top", type=int, default=20, help="rows per section (default: 20)")
    ap.add_argument("--examples", action="store_true", help="show example sentences")
    args = ap.parse_args()

    for path in (args.jsonl, args.db):
        if not path.is_file():
            sys.exit(f"error: not found: {path}")

    kwargs = {} if args.max_scope is None else {"max_scope": args.max_scope}
    errors = collect_errors(args.jsonl, args.db, **kwargs)

    total_flags = errors.n_true_positives + errors.n_false_positives
    precision = errors.n_true_positives / total_flags if total_flags else 0.0
    print(
        f"{total_flags:,} negation flags: {errors.n_true_positives:,} right, "
        f"{errors.n_false_positives:,} wrong (P={precision:.3f}); "
        f"{errors.n_false_negatives:,} of Java's negations missed\n"
    )

    ranked = top_triggers(errors, args.top)
    if ranked:
        cumulative = 0
        print(f"=== WRONG FLAGS BY TRIGGER (top {len(ranked)}) ===")
        print(f"{'trigger':<26}{'wrong':>7}{'fired':>7}{'prec':>7}{'cum%':>7}")
        for t in ranked:
            cumulative += t.wrong
            share = 100 * cumulative / errors.n_false_positives if errors.n_false_positives else 0
            print(
                f"{t.trigger[:25]:<26}{t.wrong:>7,}{t.fired:>7,}"
                f"{t.precision:>7.2f}{share:>6.0f}%"
            )
        print(f"\n{len(errors.by_trigger)} distinct trigger phrases fired at all.")

        if args.examples:
            print("\n--- examples of wrong flags ---")
            for t in ranked[:5]:
                if not t.examples:
                    continue
                print(f"\n  [{t.trigger}]")
                for mention, sentence in t.examples:
                    print(f"    {mention!r} in: {sentence[:150]}")
        print()

    print(f"=== MISSED NEGATIONS: why (of {errors.n_false_negatives:,}) ===")
    for cause, n in errors.false_negative_causes.most_common(args.top):
        print(f"  {n:>6,}  {cause[:90]}")

    print(f"\n=== MISSED NEGATIONS: most frequent mentions (top {args.top}) ===")
    for mention, n in top_false_negatives(errors, args.top):
        print(f"  {n:>6,}  {mention[:70]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

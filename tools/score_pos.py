#!/usr/bin/env python3
"""Score POS agreement between the Python tagger and Java cTAKES.

Reports raw tag agreement and -- more usefully -- *anchor* agreement: whether
the two taggers agree on which tokens may anchor a dictionary lookup. Only the
latter affects matcher output. See ``umlsmatch.eval.pos``.

Usage::

    python tools/score_pos.py --jsonl free_texts/json/silver.jsonl
    python tools/score_pos.py --jsonl free_texts/json/silver.jsonl --top 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.pos import aggregate, score_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", type=Path, required=True, help="silver-standard JSONL")
    ap.add_argument("--top", type=int, default=15, help="confusions to list (default: 15)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-document lines")
    args = ap.parse_args()

    if not args.jsonl.is_file():
        sys.exit(f"error: not found: {args.jsonl}")

    scores = score_jsonl(args.jsonl)
    if not scores:
        sys.exit("error: no records scored")

    if not args.quiet:
        for s in scores:
            print(
                f"{Path(s.source_file).name:<48} "
                f"tag={s.tag_agreement:.3f}  anchor={s.anchor_agreement:.3f}  n={s.compared}"
            )
        print()

    agg = aggregate(scores)
    if not agg.compared:
        sys.exit("error: no tokens aligned -- check tokenization")

    print(f"{agg.documents} documents, {agg.compared:,} span-aligned tokens")
    if agg.unaligned:
        pct = 100 * agg.unaligned / (agg.compared + agg.unaligned)
        print(f"  ({agg.unaligned:,} cTAKES tokens unaligned, {pct:.1f}% -- tokenization diff)")
    print()
    print(f"  tag agreement    {agg.tag_agreement:.3f}")
    print(f"  anchor agreement {agg.anchor_agreement:.3f}   <- the one that affects matching")
    print()

    extra = agg.anchor_flips.get("python_anchors_extra", 0)
    missed = agg.anchor_flips.get("python_misses_anchor", 0)
    total_flips = extra + missed
    print(f"  anchor-eligibility flips: {total_flips:,}")
    if total_flips:
        print(f"    python anchors where cTAKES would not : {extra:,}  (-> false positives)")
        print(f"    python skips where cTAKES would anchor: {missed:,}  (-> false negatives)")
    print()

    print(f"  top {args.top} tag confusions (cTAKES -> python):")
    for (gold, py), n in agg.confusions.most_common(args.top):
        print(f"    {gold!s:<6} -> {py!s:<6} {n:>7,}")
    print()

    detailed = [
        (k, v)
        for k, v in agg.anchor_flips.items()
        if isinstance(k, tuple) and len(k) == 3
    ]
    detailed.sort(key=lambda kv: -kv[1])
    print(f"  top {args.top} confusions that flip anchor eligibility:")
    for (direction, gold, py), n in detailed[: args.top]:
        arrow = "+anchor" if direction == "python_anchors_extra" else "-anchor"
        print(f"    {arrow}  {gold!s:<6} -> {py!s:<6} {n:>7,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score token- and sentence-boundary agreement: Python tokenizer vs. Java cTAKES.

Measures the boundary targets README's Validation section sets ("(greater-eq)95%
token-boundary agreement and (greater-eq)90% sentence-boundary agreement with
cTAKES"). See ``umlsmatch.eval.boundaries`` for
the scoring definition (exact-span match, same philosophy as
tools/score_parity.py's exact-CUI match).

Note the sentence target is retained as a **diagnostic**, not a gate: it
reads ~0.54 and was measured not to predict downstream quality. A low
sentence number here is expected and is not a release blocker.

Usage::

    python tools/score_boundaries.py --jsonl free_texts/json/silver.jsonl

Requires a silver-standard JSONL exported with "sentences"/"tokens" fields
-- re-run tools/run_java_ctakes.py's --jsonl export (``--skip-run`` reuses
existing XMI, no need to re-invoke Java) if your file predates those fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.boundaries import aggregate, score_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Silver-standard file (JSONL)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-document lines")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        sys.exit(f"error: not found: {args.jsonl}")

    sentence_scores, token_scores = score_jsonl(args.jsonl)
    if not sentence_scores:
        sys.exit(f"error: no records scored from {args.jsonl}")

    if not args.quiet:
        # strict: score_jsonl documents these as index-aligned, one pair per
        # document; a mismatch would silently mis-pair every later line.
        for s, t in zip(sentence_scores, token_scores, strict=True):
            print(
                f"{s.source_file:45s} "
                f"sent F1={s.f1:.3f} (P={s.precision:.3f} R={s.recall:.3f})  "
                f"tok F1={t.f1:.3f} (P={t.precision:.3f} R={t.recall:.3f})"
            )
        print()

    s_agg = aggregate(sentence_scores)
    t_agg = aggregate(token_scores)
    print(f"{s_agg.n_docs} documents")
    print(
        f"sentence boundaries  micro P={s_agg.micro_precision:.3f} R={s_agg.micro_recall:.3f} "
        f"F1={s_agg.micro_f1:.3f}   (target: >=90% agreement)"
    )
    print(
        f"token boundaries     micro P={t_agg.micro_precision:.3f} R={t_agg.micro_recall:.3f} "
        f"F1={t_agg.micro_f1:.3f}   (target: >=95% agreement)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

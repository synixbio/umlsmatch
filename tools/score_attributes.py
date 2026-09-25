#!/usr/bin/env python3
"""Score an assertion attribute against the Java cTAKES silver standard.

    python tools/score_attributes.py --jsonl free_texts/json/silver.jsonl \\
        --db data/umls_ctakes_16ab.sqlite --attribute subject

Per-mention agreement, aligned on same CUI + overlapping span -- the same
alignment ``tools/score_negation_spans.py`` uses, so the numbers reconcile
rather than being two measurements of the same corpus.

Pass ``--attribute all`` to score every attribute the pipeline assesses.

**What this measures.** Agreement with cTAKES, not correctness. cTAKES is
measurably wrong on constructions that are common in these notes, and for the
rare attributes the positive count is too small to support a decision at all --
the tool prints the caveat that applies to each attribute next to its score, and
that sentence is part of the output, not decoration.

PHI: reads the silver standard, which embeds note text. The error-attribution
listing quotes matched spans (a few words each, not sentences); pass
``--no-examples`` to suppress it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.attributes import attribute_names, get_attribute
from umlsmatch.eval.attributes import scorable_attributes, score_corpus


def _report(agg, *, examples: bool, top: int) -> None:
    spec = agg.spec
    print(f"\n=== {spec.name} (cTAKES: {spec.ctakes_name}) ===")
    print(f"  {spec.note}\n")
    print(f"  aligned mentions      {agg.n_aligned:,}")
    print(f"  unaligned (not scored){agg.n_unaligned:>10,}")
    print(f"  gold positives        {agg.n_gold_positives:,}")
    print(f"  predicted positives   {agg.n_predicted_positives:,}")
    print(
        f"\n  micro  P {agg.micro_precision:.3f}  R {agg.micro_recall:.3f}  "
        f"F1 {agg.micro_f1:.3f}  acc {agg.micro_accuracy:.3f}"
    )
    print(
        f"  macro  P {agg.macro_precision:.3f}  R {agg.macro_recall:.3f}  "
        f"F1 {agg.macro_f1:.3f}   ({agg.n_docs} documents)"
    )

    if examples:
        for label, counter in (
            ("we said yes, cTAKES said no", agg.false_positive_texts),
            ("cTAKES said yes, we said no", agg.false_negative_texts),
        ):
            if not counter:
                continue
            print(f"\n  {label}:")
            for text, n in counter.most_common(top):
                print(f"    {n:>5}  {text[:60]}")

    print(f"\n  {agg.caveat()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", type=Path, required=True, help="silver-standard JSONL")
    ap.add_argument(
        "--db", type=Path, default=Path("data/umls_sno_rx.sqlite"),
        help="dictionary SQLite DB (default: %(default)s)",
    )
    ap.add_argument(
        "--attribute", default="all",
        choices=("all", *attribute_names()),
        help="attribute to score, or 'all' for every one the pipeline assesses",
    )
    ap.add_argument(
        "--no-examples", action="store_true",
        help="suppress the error-attribution listing, which quotes matched spans",
    )
    ap.add_argument("--top", type=int, default=15, help="examples to list (default: 15)")
    args = ap.parse_args()

    for path in (args.jsonl, args.db):
        if not path.is_file():
            sys.exit(f"error: not found: {path}")

    options: dict[str, object] = {}
    if args.attribute == "all":
        # Deliberately the narrow list: `all` is what fills a results table, and
        # a prototype has no result. Name one explicitly to score it.
        specs = scorable_attributes()
    else:
        spec = get_attribute(args.attribute)
        if not spec.assessable:
            sys.exit(
                f"error: {spec.name} is not implemented, deliberately -- "
                f"{spec.note} Scoring it would measure a column of False."
            )
        if spec.prototype and spec.pipeline_option:
            options[spec.pipeline_option] = True
        specs = (spec,)

    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(args.db, **options) as nlp:
        for spec in specs:
            agg = score_corpus(args.jsonl, nlp, spec)
            _report(agg, examples=not args.no_examples, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a blind review file for human adjudication of an assertion attribute.

cTAKES is not truth. Its polarity is wrong on Review-of-Systems negatives; its
``subject`` labels EHR family-history tables inconsistently; its ``historyOf``
marks the same mention text both ways (6 distinct texts on the 20-note corpus,
113 on a real-note corpus not included here). Every number
``tools/score_attributes.py`` prints is therefore agreement with a faulty
oracle. This produces the human gold standard that replaces it -- without
labelling every mention in the corpus, by adjudicating every case where the two
systems disagree and sampling the cases where they agree. See
``umlsmatch.eval.adjudication``.

Writes two files:

  * ``review.csv``  -- give this to the annotator. Blind: it shows the sentence,
    the mention and the question, and deliberately NOT what either system
    predicted.
  * ``design.json`` -- the sampling design. Do not show it to the annotator;
    ``score_adjudicated.py`` needs it to reweight the verdicts.

Usage::

    python tools/make_adjudication_set.py --jsonl free_texts/json/silver.jsonl \\
        --db data/umls_ctakes_16ab.sqlite --attribute subject

Then fill in the ``verdict`` column with the attribute's own words (the tool
prints them) and run ``tools/score_adjudicated.py``.

PHI: the review file quotes whole sentences of note text. It defaults to
``free_texts/adjudication/``, which is git-ignored, and the pre-commit hook
refuses to commit ``*.csv``. Keep it under the same controls as the notes
themselves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.attributes import attribute_names, get_attribute
from umlsmatch.eval.adjudication import (
    BOTH_NEGATIVE,
    BOTH_POSITIVE,
    DEFAULT_SAMPLE_SIZES,
    STRATA,
    collect_cases,
    sample_cases,
    write_review_csv,
)


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
        "--attribute", default="negated", choices=attribute_names(),
        help="attribute to adjudicate (default: %(default)s)",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="where to write review.csv and design.json "
             "(default: free_texts/adjudication/<attribute>)",
    )
    ap.add_argument(
        "--agree-positive", type=int, default=DEFAULT_SAMPLE_SIZES[BOTH_POSITIVE],
        help="sample size where both systems say yes (default: %(default)s)",
    )
    ap.add_argument(
        "--agree-negative", type=int, default=DEFAULT_SAMPLE_SIZES[BOTH_NEGATIVE],
        help=(
            "sample size where both say no (default: %(default)s). This is what "
            "makes recall estimable -- a positive both systems missed appears in "
            "no disagreement."
        ),
    )
    ap.add_argument(
        "--max-scope", type=int, default=None, metavar="N",
        help="override the negation scope cap, in tokens",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    args = ap.parse_args()

    for path in (args.jsonl, args.db):
        if not path.is_file():
            sys.exit(f"error: not found: {path}")

    spec = get_attribute(args.attribute)
    if not spec.assessable:
        sys.exit(
            f"error: {spec.name} is not assessed by this pipeline, so two of the "
            f"four strata would be empty and the design could not reweight.\n"
            f"       {spec.note}"
        )
    out_dir = args.out_dir or Path("free_texts/adjudication") / spec.name

    # The pipeline's own options go into the design: an estimate reweights the
    # strata that *this* configuration produced, and a design that does not say
    # which configuration that was cannot be reproduced later.
    options = {} if args.max_scope is None else {"max_scope": args.max_scope}
    if spec.prototype and spec.pipeline_option:
        # A prototype is off by default, and adjudicating it is the whole
        # reason it is allowed here -- so turn it on rather than making the
        # caller know the flag. It lands in the design with everything else,
        # which is what keeps the resulting estimate attributable to the rules
        # that produced it if they later change.
        options[spec.pipeline_option] = True
        print(
            f"note: {spec.name} is a prototype; building the pipeline with "
            f"{spec.pipeline_option}=True. {spec.note}"
        )

    from umlsmatch import ClinicalPipeline

    print(f"collecting aligned mentions from {args.jsonl} for {spec.name} ...")
    with ClinicalPipeline(args.db, **options) as nlp:
        cases = collect_cases(args.jsonl, nlp, spec)
    if not cases:
        sys.exit("error: no aligned mentions found")

    sizes = dict(DEFAULT_SAMPLE_SIZES)
    sizes[BOTH_POSITIVE] = args.agree_positive
    sizes[BOTH_NEGATIVE] = args.agree_negative

    sampled, design = sample_cases(
        cases, sizes, seed=args.seed, pipeline_options=options
    )

    review = out_dir / "review.csv"
    design_path = out_dir / "design.json"
    write_review_csv(sampled, review)
    design.to_json(design_path)

    print(f"\n{len(cases):,} aligned mentions in the corpus\n")
    print(f"  {'stratum':<16}{'population':>12}{'sampled':>10}{'rate':>8}")
    for s in STRATA:
        pop = design.population[s]
        got = design.sampled[s]
        rate = f"{100 * got / pop:.0f}%" if pop else "-"
        print(f"  {s:<16}{pop:>12,}{got:>10,}{rate:>8}")

    total = len(sampled)
    print(f"\n  {total:,} cases to adjudicate")
    print(f"  ~{total * 12 / 3600:.1f} hours at 12 seconds each")
    print(f"\nwrote {review}")
    print(f"      {design_path}")
    print(
        f"\nThe question is: {spec.question}\n"
        f"Fill in the 'verdict' column with: {' / '.join(spec.verdicts)}\n"
        "The file is deliberately blind -- it does not show what either system\n"
        "predicted, because seeing a machine label anchors the judgement it is\n"
        "meant to overrule.\n"
        f"\nThen:  python tools/score_adjudicated.py --review {review}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

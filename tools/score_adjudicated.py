#!/usr/bin/env python3
"""Score an assertion attribute against human verdicts instead of against cTAKES.

Reads a review file filled in by ``tools/make_adjudication_set.py`` and reports
corpus-level precision and recall, reweighted for the stratified sampling. Works
for any attribute the pipeline assesses; the design records which one.

This is the number to use once you have it. ``tools/score_attributes.py``
measures agreement with cTAKES, which is wrong on Review-of-Systems negatives,
inconsistent on family-history tables, and self-contradictory on 113 distinct
mention texts for ``historyOf``. Against human labels the two numbers can
legitimately diverge, and where they do, this one is the accuracy.

Usage::

    python tools/score_adjudicated.py \\
        --review free_texts/adjudication/subject/review.csv

Partial progress is fine: blank verdicts are skipped, so this can be run
part-way through annotation to watch the intervals tighten. Treat an interval
that still straddles a decision boundary as "keep annotating", not as a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.attributes import get_attribute
from umlsmatch.eval.adjudication import (
    BOTH_NEGATIVE,
    STRATA,
    Design,
    estimate,
    load_verdicts,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--review", type=Path, required=True,
        help="filled-in review file, e.g. free_texts/adjudication/subject/review.csv",
    )
    ap.add_argument(
        "--design", type=Path, default=None,
        help="sampling design (default: design.json beside --review)",
    )
    ap.add_argument(
        "--bootstrap", type=int, default=2000,
        help="bootstrap resamples for the intervals (default: %(default)s)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    design_path = args.design or args.review.parent / "design.json"
    for path in (args.review, design_path):
        if not path.is_file():
            sys.exit(f"error: not found: {path}")

    design = Design.from_json(design_path)
    spec = get_attribute(design.attribute)
    verdicts = load_verdicts(args.review, spec)
    if not verdicts:
        sys.exit(
            f"error: no verdicts filled in yet in {args.review}\n"
            f"Fill the 'verdict' column with {' / '.join(spec.verdicts)}."
        )

    est = estimate(design, verdicts, bootstrap=args.bootstrap, seed=args.seed)

    total_sampled = sum(design.sampled.values())
    done = len(verdicts)
    print(f"attribute: {spec.name}  ({spec.question})")
    print(f"{done:,}/{total_sampled:,} cases adjudicated ({100 * done / total_sampled:.0f}%)")
    if est.n_unclear:
        print(
            f"  {est.n_unclear:,} marked unclear and excluded "
            f"({100 * est.n_unclear / max(done, 1):.1f}%)"
        )
    print()

    header = spec.positive_label[:9]
    print(f"  {'stratum':<16}{header:>9}{'judged':>9}{'rate':>8}{'population':>12}")
    for s in STRATA:
        positive, judged, pop = est.per_stratum[s]
        rate = f"{positive / judged:.2f}" if judged else "-"
        print(f"  {s:<16}{positive:>9,}{judged:>9,}{rate:>8}{pop:>12,}")

    print("\n  corpus estimate (stratified, 95% bootstrap interval)")
    print(
        f"    precision  {est.precision:.3f}  "
        f"[{est.precision_lo:.3f}, {est.precision_hi:.3f}]"
    )
    print(
        f"    recall     {est.recall:.3f}  "
        f"[{est.recall_lo:.3f}, {est.recall_hi:.3f}]"
    )
    print(f"    F1         {est.f1:.3f}")

    # Recall extrapolates a small rate over a large stratum, so its interval is
    # usually the widest thing here and the one that decides whether more
    # annotation is worth buying. Say which knob moves it.
    recall_width = est.recall_hi - est.recall_lo
    if recall_width > 0.15:
        _, judged, pop = est.per_stratum[BOTH_NEGATIVE]
        print(
            f"\n  note: the recall interval spans {recall_width:.2f}. It is dominated by\n"
            f"  extrapolating {judged:,} adjudicated '{BOTH_NEGATIVE}' cases over a stratum\n"
            f"  of {pop:,}. Raise --agree-negative when generating the set if recall\n"
            "  matters; precision is unaffected by that stratum."
        )

    incomplete = [s for s in STRATA if est.per_stratum[s][1] == 0 and design.sampled.get(s)]
    if incomplete:
        print(
            "\n  warning: no verdicts yet for "
            + ", ".join(incomplete)
            + f"\n  The estimate treats those strata as containing no true "
            f"{spec.positive_label}\n  cases, which is almost certainly wrong -- "
            "finish them before quoting this."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score a *changed* pipeline against verdicts collected for the shipped one.

``tools/score_adjudicated.py`` reads each case's prediction out of its stratum
-- ``python_only`` and ``both_positive`` mean the pipeline said yes. That is
exact and free, and it stops being true the moment the pipeline changes. Scoring
a variant with it would compare new verdicts against old predictions.

This re-runs the pipeline per case instead and reweights with the *same*
``design.json``. That is valid because a stratified design is a sampling frame:
the inclusion probability of each case is known and fixed, so a
Horvitz-Thompson estimate of any corpus quantity is unbiased regardless of what
the strata were originally defined by. The strata came from the old pipeline;
the estimate does not have to.

    population[s] / adjudicated[s]  is the weight each case carries -- the
    denominator counts cases actually judged, not cases sampled, because
    ``unclear`` verdicts leave the numerator.

Validated by running it with the shipped settings: it reproduces
``score_adjudicated.py`` to three decimals on ``history_of``.

**This cannot tell you a variant is better.** It tells you what a variant does
against whatever verdicts you have. If those verdicts are a model
pre-annotation, tuning against them fits the model's conventions and reports the
fit as an improvement. Use it to size an effect you already have a reason to
expect, and confirm on human verdicts.

Usage::

    python tools/score_variant_adjudicated.py \\
        --review free_texts/adjudication/history_of/review_model_prefill.csv \\
        --db data/umls_ctakes_16ab.sqlite --history-sections
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.attributes import get_attribute
from umlsmatch.eval.adjudication import Design
from umlsmatch.eval.attributes import predicted_occurrences


def _predictions(review_rows, db, **pipeline_kwargs) -> dict[str, bool]:
    """Run the pipeline once per document and read off each case's prediction."""
    from umlsmatch import ClinicalPipeline

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in review_rows:
        by_doc[r["document"]].append(r)

    attribute = review_rows[0]["attribute"]
    spec = get_attribute(attribute)
    notes = Path("free_texts/synthetic")
    out: dict[str, bool] = {}

    with ClinicalPipeline(db, **pipeline_kwargs) as nlp:
        for doc, rows in sorted(by_doc.items()):
            path = notes / doc
            if not path.is_file():
                raise SystemExit(f"note not found: {path}")
            # newline="" disables universal-newline translation. The review
            # file's offsets come from cTAKES' sofaString, which preserves CRLF;
            # read_text() would collapse it to LF, shift every offset past the
            # first line break, and match nothing -- reporting a silent all-zero
            # score rather than failing.
            with path.open(encoding="utf-8", errors="replace", newline="") as fh:
                anns = nlp.analyze(fh.read())
            # Same reader the scorer uses, so "unassessed counts as false"
            # stays one decision in one place.
            index = {(s, e, cui): v for s, e, cui, v in predicted_occurrences(anns, spec)}
            for r in rows:
                key = (int(r["start"]), int(r["end"]), r["cui"])
                out[r["case_id"]] = index.get(key, False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=None,
                    help="default: design.json beside the review file")
    ap.add_argument("--db", type=Path, default=Path("data/umls_ctakes_16ab.sqlite"))
    ap.add_argument("--history-sections", action="store_true",
                    help="enable the Past Medical History section rule")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.review.open(encoding="utf-8-sig", newline="")))
    design = Design.from_json(args.design or args.review.parent / "design.json")
    attribute = rows[0]["attribute"]
    positive = get_attribute(attribute).positive_label

    judged = [r for r in rows if (r.get("verdict") or "").strip()
              and r["verdict"].strip() != "unclear"]
    if not judged:
        raise SystemExit("no adjudicated verdicts in that review file")

    preds = _predictions(judged, args.db, history_sections=args.history_sections)

    # Weight by cases actually ADJUDICATED per stratum, not by cases sampled.
    # `unclear` verdicts leave the numerator, so leaving them in the denominator
    # would shrink every stratum's contribution by its unclear rate -- which for
    # history_of is 41% -- enough to understate recall by about a third.
    # Renormalizing within stratum is the same thing score_adjudicated.py
    # does implicitly when it takes positives/judged as the stratum rate.
    adjudicated: dict[str, int] = defaultdict(int)
    for r in judged:
        adjudicated[design.case_stratum[r["case_id"]]] += 1

    tp = fp = fn = 0.0
    for r in judged:
        stratum = design.case_stratum[r["case_id"]]
        weight = design.population[stratum] / adjudicated[stratum]
        gold = r["verdict"].strip() == positive
        pred = preds[r["case_id"]]
        if pred and gold:
            tp += weight
        elif pred:
            fp += weight
        elif gold:
            fn += weight

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"attribute: {attribute}   variant: history_sections={args.history_sections}")
    print(f"{len(judged):,} adjudicated cases "
          f"({len(rows) - len(judged):,} unclear, excluded)")
    print("  corpus estimate (stratified, Horvitz-Thompson)")
    print(f"    TP {tp:>9,.0f}   FP {fp:>9,.0f}   FN {fn:>9,.0f}")
    print(f"    precision  {precision:.3f}")
    print(f"    recall     {recall:.3f}")
    print(f"    F1         {f1:.3f}")
    print("\n  Verdicts, not truth. If they are a model pre-annotation, this "
          "measures\n  agreement with that model -- see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

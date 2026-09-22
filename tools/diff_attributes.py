#!/usr/bin/env python3
"""Attribute every assertion disagreement to the rule and cue that caused it.

    python tools/diff_attributes.py --jsonl free_texts/json/silver.jsonl \\
        --db data/umls_ctakes_16ab.sqlite --attribute subject

``tools/score_attributes.py`` says how much a rule set disagrees with cTAKES.
This says why: per rule (section / self-reference / cue window), then per cue
phrase and direction, ranked by wrong calls.

Read the per-rule table first. A rule at precision 0.85 and one at 0.51 call for
opposite investments, and no aggregate F1 distinguishes them -- this
decomposition is what showed that a forward window from a kinship term was
reaching into the next row of an EHR family-history table.

The counterpart for polarity is ``tools/diff_negation.py``, which predates this
and is kept: negation's parse-driven rules have no analogue here.

PHI: the examples quote whole sentences of note text. Pass ``--no-examples`` to
suppress them, and treat the output as you would the notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.history import CANDIDATE_HISTORY_SECTIONS
from umlsmatch.dictionary.matcher import RareWordMatcher
from umlsmatch.eval.attribute_diff import collect_errors, top_cues
from umlsmatch.eval.explain import EXPLAINERS


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
        "--attribute", default="subject", choices=sorted(EXPLAINERS),
        help="attribute to attribute errors for (default: %(default)s)",
    )
    ap.add_argument(
        "--history-sections", action="store_true",
        help="history_of only: claim a PMH/FH section body wholesale, as the "
             "clinical_recall profile does. Off by default, matching the "
             "shipped pipeline -- attributing errors under a rule set the "
             "pipeline is not running is a report on the wrong thing.",
    )
    ap.add_argument("--top", type=int, default=20, help="cues to list (default: 20)")
    ap.add_argument(
        "--no-examples", action="store_true",
        help="suppress the example sentences, which are verbatim note text",
    )
    args = ap.parse_args()

    for path in (args.jsonl, args.db):
        if not path.is_file():
            sys.exit(f"error: not found: {path}")

    # Rejected rather than ignored: a flag that silently does nothing on the
    # other attributes would be read as "measured, no effect". This is argument
    # validation, not a per-attribute dispatch branch -- EXPLAINERS stays keyed
    # by name and the call below stays uniform.
    if args.history_sections and args.attribute != "history_of":
        sys.exit("error: --history-sections applies only to --attribute history_of")
    options = (
        {"history_sections": CANDIDATE_HISTORY_SECTIONS} if args.history_sections else {}
    )

    with RareWordMatcher(args.db) as matcher:
        explain = EXPLAINERS[args.attribute](matcher, **options)
        errors = collect_errors(args.jsonl, None, args.attribute, explain)

    total = errors.n_true_positives + errors.n_false_positives
    print(f"\n{args.attribute}: {total:,} positive calls, "
          f"{errors.n_false_positives:,} of them disagreeing with cTAKES; "
          f"{errors.n_false_negatives:,} positives missed\n")

    print(f"  {'rule':<12}{'right':>8}{'wrong':>8}{'precision':>12}")
    for rule, (correct, wrong) in sorted(
        errors.by_rule().items(), key=lambda kv: -kv[1][1]
    ):
        fired = correct + wrong
        print(f"  {rule:<12}{correct:>8,}{wrong:>8,}{correct / fired:>12.3f}")

    print(f"\n  {'rule':<10}{'cue':<32}{'right':>7}{'wrong':>7}{'prec':>8}")
    for cue in top_cues(errors, args.top):
        print(
            f"  {cue.rule:<10}{cue.label[:31]:<32}"
            f"{cue.correct:>7,}{cue.wrong:>7,}{cue.precision:>8.3f}"
        )
        if args.no_examples:
            continue
        for mention, sentence in cue.examples[:1]:
            print(f"      {mention!r} in {sentence[:96]!r}")

    if errors.false_negatives:
        print("\n  most-missed mentions (cTAKES said yes, we said no):")
        for text, n in errors.false_negatives.most_common(args.top):
            print(f"    {n:>5}  {text[:60]}")

    print(
        "\n  Reference is cTAKES, which is not truth -- a 'wrong' here is a "
        "disagreement.\n  See docs/ADJUDICATION_RESULTS.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

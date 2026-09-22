#!/usr/bin/env python3
"""Per-trigger precision for `negated`, against adjudicated verdicts.

``tools/diff_attributes.py`` answers the same question against cTAKES. That is
the right tool when cTAKES is a sound reference, and for ``negated`` it is not:
it marks Review-of-Systems negatives affirmed, so it penalises the pipeline for
the construction it gets right and cannot see the ones it gets wrong.

This attributes each *adjudicated* false positive to the trigger phrase that
caused it, by re-running :func:`~umlsmatch.assertion.negation.negation_evidence`
on the sentence. Counts are reweighted to corpus scale with the sampling design,
so a trigger that fires twice in a rarely-sampled stratum is not mistaken for a
rare problem.

Only mentions the pipeline actually negated appear, so the table is a precision
breakdown. It says nothing about what the rules miss.

    python tools/diff_negation_adjudicated.py \\
        --review free_texts/adjudication/negated/review_model_prefill.csv \\
        --db data/umls_ctakes_16ab.sqlite

PHI: prints trigger phrases and matched mention text, never sentences.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.assertion.negation import MAX_SCOPE_TOKENS, negation_evidence
from umlsmatch.assertion.sections import track_sections
from umlsmatch.dictionary.matcher import RareWordMatcher
from umlsmatch.eval.adjudication import Design
from umlsmatch.pipeline.tokenizer import annotate_sentences

#: Strata in which the shipped pipeline predicted "negated".
PIPELINE_POSITIVE = frozenset({"both_positive", "python_only"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=Path("data/umls_ctakes_16ab.sqlite"))
    ap.add_argument("--notes", type=Path, default=Path("free_texts/synthetic"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--mentions-for",
        default=None,
        metavar="TRIGGER",
        help="also list the mention texts this trigger wrongly negates",
    )
    args = ap.parse_args()

    rows = [
        r
        for r in csv.DictReader(args.review.open(encoding="utf-8-sig", newline=""))
        if (r.get("verdict") or "").strip() and r["verdict"].strip() != "unclear"
    ]
    if not rows:
        raise SystemExit("no adjudicated verdicts in that review file")
    if rows[0]["attribute"] != "negated":
        raise SystemExit(f"this tool is for negated, not {rows[0]['attribute']}")

    design = Design.from_json(args.review.parent / "design.json")
    adjudicated: Counter[str] = Counter(
        design.case_stratum[r["case_id"]] for r in rows
    )

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[r["document"]].append(r)

    fp: Counter[str] = Counter()
    tp: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    matcher = RareWordMatcher(args.db)
    try:
        for doc, drows in sorted(by_doc.items()):
            text = (args.notes / doc).read_text(encoding="utf-8", errors="replace")
            want = {(int(r["start"]), int(r["end"]), r["cui"]): r for r in drows}
            for tokens, section in track_sections(annotate_sentences(text)):
                evidence = negation_evidence(
                    tokens,
                    matcher.match(tokens),
                    # Mirror the pipeline: a negative-findings section uncaps.
                    max_scope=None if section is not None else MAX_SCOPE_TOKENS,
                )
                for match, triggers in evidence.items():
                    row = want.get((match.start, match.end, match.cui))
                    if row is None:
                        continue
                    stratum = design.case_stratum[row["case_id"]]
                    if stratum not in PIPELINE_POSITIVE:
                        continue
                    weight = design.population[stratum] / adjudicated[stratum]
                    phrases = sorted(
                        " ".join(t.norm for t in tokens[s.start : s.end])
                        for s in triggers
                    )
                    label = " | ".join(phrases)
                    if row["verdict"].strip() == "negated":
                        tp[label] += weight
                    else:
                        fp[label] += weight
                        if args.mentions_for and args.mentions_for in phrases:
                            mentions[row["mention"].casefold()] += 1
    finally:
        matcher.close()

    print(f"{'trigger':<34}{'FP (est.)':>11}{'TP (est.)':>11}{'precision':>11}")
    for label, wrong in fp.most_common(args.top):
        right = tp.get(label, 0.0)
        print(f"{label[:33]:<34}{wrong:>11,.0f}{right:>11,.0f}"
              f"{right / (right + wrong):>11.2f}")
    print(f"\nestimated false positives across all triggers: {sum(fp.values()):,.0f}")

    if args.mentions_for:
        print(f"\nmentions {args.mentions_for!r} wrongly negates:")
        for mention, n in mentions.most_common(args.top):
            print(f"  {n:>3}  {mention}")
    print("\nVerdicts, not truth -- see docs/ADJUDICATION_RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

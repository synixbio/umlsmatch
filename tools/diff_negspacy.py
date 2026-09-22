#!/usr/bin/env python3
"""Diff this pipeline's negation against negspaCy's NegEx -- an independent second opinion.

The negation rules have no trustworthy external check: the human
gold standard is built but unannotated, and cTAKES polarity is
wrong on exactly the constructions that matter. Two independent NegEx
implementations disagreeing does not say which is right, but it says *where to
look*, and it costs no annotation to run.

Treat the output as a work queue, not a score. See
``umlsmatch.eval.negspacy_diff`` for the three caveats that shape it -- chiefly
that only non-overlapping mentions can be compared, because negspaCy decides
polarity for ``doc.ents`` and spaCy entities may not overlap.

Usage::

    python tools/diff_negspacy.py --input-dir free_texts/synthetic
    python tools/diff_negspacy.py --input-dir free_texts/synthetic --top 30

    # isolate what the new dependency rules changed:
    python tools/diff_negspacy.py --input-dir free_texts/synthetic --no-coordination

Needs the optional comparison extra::

    pip install -e ".[compare]"

PHI note: mention text and sentences come from the notes. By default this
prints aggregate counts and CUIs only. ``--examples`` adds the surrounding
sentence and must not be used to produce anything that leaves the machine.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.eval.negspacy_diff import compare_documents


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="folder of .txt notes")
    ap.add_argument("--db", type=Path, default=None, help="dictionary SQLite path")
    ap.add_argument("--top", type=int, default=20, help="rows per section (default: 20)")
    ap.add_argument(
        "--examples",
        action="store_true",
        help="print the sentence for each disagreement (PHI -- screen only)",
    )
    ap.add_argument(
        "--no-coordination",
        action="store_true",
        help="disable coordinate-list scope, to isolate what it changed",
    )
    ap.add_argument(
        "--no-clause-bounding", action="store_true", help="disable clause bounding"
    )
    ap.add_argument("--no-sections", action="store_true", help="disable section scope")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: not a directory: {args.input_dir}", file=sys.stderr)
        return 2

    notes = sorted(args.input_dir.glob("*.txt"))
    if not notes:
        print(f"error: no .txt files in {args.input_dir}", file=sys.stderr)
        return 2

    texts = [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in notes]

    try:
        report = compare_documents(
            texts,
            db_path=args.db,
            keep_sentences=args.examples,
            coordination=not args.no_coordination,
            clause_bounding=not args.no_clause_bounding,
            sections=not args.no_sections,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report.summary())

    for label, rows in (
        ("we negate, negspaCy does not", report.we_negate_they_do_not),
        ("negspaCy negates, we do not", report.they_negate_we_do_not),
    ):
        if not rows:
            continue
        print(f"\n  {label} ({len(rows)}):")
        by_mention = collections.Counter(d.mention.casefold() for d in rows)
        for mention, n in by_mention.most_common(args.top):
            print(f"    {n:5}  {mention}")

    if args.examples:
        print("\n  examples (PHI -- screen only):")
        for d in report.disagreements[: args.top]:
            side = "ours" if d.ours else "negspaCy"
            print(f"    [{side} negates] {d.mention!r} ({d.cui})")
            print(f"        {d.sentence[:160]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

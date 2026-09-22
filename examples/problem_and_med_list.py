#!/usr/bin/env python3
"""Build a per-document problem list and medication list.

    python examples/problem_and_med_list.py free_texts/synthetic
    python examples/problem_and_med_list.py free_texts/synthetic --limit 3 --all-mentions

Demonstrates the normal shape of downstream use: group annotations by semantic
group, drop negated mentions, and deduplicate by CUI so "diabetes", "DM" and
"diabetes mellitus type 2" collapse into one problem rather than three.

Deduplicating by CUI (not by text) is the point. The same concept appears in
many surface forms across a note; the CUI is what makes them one thing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection, Iterable
from pathlib import Path

from umlsmatch import Annotation, ClinicalPipeline
from umlsmatch.corpus import iter_text_files, read_text

PROBLEM_GROUPS = {"DISORDER", "FINDING"}
MED_GROUPS = {"DRUG"}
PROCEDURE_GROUPS = {"PROCEDURE"}


def summarize(
    annotations: Iterable[Annotation],
    groups: Collection[str],
    *,
    include_negated: bool = False,
) -> list[tuple[str, tuple[str, int]]]:
    """CUI -> (preferred label, mention count), restricted to `groups`."""
    out: dict[str, tuple[str, int]] = {}
    for a in annotations:
        if a.group not in groups:
            continue
        if a.negated and not include_negated:
            continue
        label = a.preferred_text or a.text
        prev = out.get(a.cui)
        out[a.cui] = (label, (prev[1] if prev else 0) + 1)
    # Most-mentioned first: a concept repeated through a note is usually the
    # one the note is actually about.
    return sorted(out.items(), key=lambda kv: (-kv[1][1], kv[1][0].lower()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--db", default=None)
    ap.add_argument("--limit", type=int, default=5, help="documents to show (default: 5)")
    ap.add_argument("--top", type=int, default=10, help="entries per list (default: 10)")
    ap.add_argument(
        "--all-mentions", action="store_true", help="include negated mentions too"
    )
    args = ap.parse_args()

    files = list(iter_text_files(args.input_dir))
    if not files:
        print(f"no .txt files under {args.input_dir}", file=sys.stderr)
        return 1
    files = files[: args.limit]

    with ClinicalPipeline(args.db) as nlp:
        for path in files:
            text = read_text(path)
            annotations = nlp.analyze(text)

            print(
                f"\n{'=' * 72}\n{path.name}  "
                f"({len(text):,} chars, {len(annotations)} annotations)"
            )

            for title, groups in (
                ("PROBLEMS", PROBLEM_GROUPS),
                ("MEDICATIONS", MED_GROUPS),
                ("PROCEDURES", PROCEDURE_GROUPS),
            ):
                entries = summarize(
                    annotations, groups, include_negated=args.all_mentions
                )
                print(f"\n  {title} ({len(entries)} distinct)")
                if not entries:
                    print("    -")
                    continue
                for cui, (label, n) in entries[: args.top]:
                    count = f" x{n}" if n > 1 else ""
                    print(f"    {cui}  {label}{count}")
                if len(entries) > args.top:
                    print(f"    ... {len(entries) - args.top} more")

            # Negated findings are clinically meaningful in their own right --
            # "denies chest pain" is a documented negative, not an absence of
            # information.
            ruled_out = summarize(
                [a for a in annotations if a.negated], PROBLEM_GROUPS, include_negated=True
            )
            print(f"\n  DOCUMENTED NEGATIVES ({len(ruled_out)} distinct)")
            for cui, (label, n) in ruled_out[: args.top]:
                print(f"    {cui}  {label}{' x' + str(n) if n > 1 else ''}")
            if not ruled_out:
                print("    -")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

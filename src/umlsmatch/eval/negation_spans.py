"""Negation agreement scored per *mention*, not per document-collapsed CUI.

``umlsmatch.eval.negation`` collapses polarity to one boolean per CUI per
document: if any mention carrying that CUI is negated, the CUI counts negated.
That mirrors how ``eval.parity`` collapses CUI sets, and it is fine for a
concept-level question -- but it is a suspect yardstick for *polarity*, because
the two sides do not emit comparable numbers of mentions.

The Python matcher returns every overlapping hit by design (cTAKES does too, and
suppressing them measurably lowers recall), so a single phrase
can yield "chest", "chest pain" and "pain" where Java contributes one chunk-based
mention. Under an OR-collapse, whichever side emits more mention occurrences gets
more chances for at least one to land inside a negation scope. That biases the
*predicted-negated* set upward independently of whether the negation rules are
right, and it inflates exactly the quantity that looks broken: precision.

This module removes that degree of freedom. Each gold (mention, concept) pair is
aligned to a Python match by **same CUI and overlapping character span**, and
polarity is compared on the aligned pair. One gold occurrence, one prediction,
one comparison -- so a difference in mention counts cannot move the score.

Read the two together:

  * ``eval.negation`` answers "per document, do we agree this concept was
    negated somewhere?"
  * this module answers "at this specific mention, do we agree?"

If per-mention precision is materially higher than the per-document figure, the
document collapse -- not the negation lexicon -- is responsible for the
difference, and rule tuning aimed at the per-document number is aimed at an
artifact.

Alignment is deliberately strict on CUI and loose on span: cTAKES segments
mentions by noun-phrase chunk while the matcher anchors on rare words, so exact
span equality would discard most true pairs (the same reason ``eval.parity``
scores CUI sets rather than spans). Gold pairs with no Python counterpart are
counted and reported separately rather than scored -- they are a concept-recall
problem, already measured by ``eval.parity``, and folding them in here would
conflate two different failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from umlsmatch.assertion.negation import MAX_SCOPE_TOKENS, negated_matches
from umlsmatch.dictionary.matcher import RareWordMatcher
from umlsmatch.eval.metrics import PrfAggregate, PrfScore, aggregate_prf, safe_div
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "AggregateSpanNegationScore",
    "SpanNegationScore",
    "aggregate",
    "align_mentions",
    "gold_mention_polarity",
    "python_match_polarity",
    "score_document",
    "score_jsonl",
]

#: (begin, end, cui, negated) for one mention-concept occurrence.
Occurrence = tuple[int, int, str, bool]


@dataclass(frozen=True)
class SpanNegationScore(PrfScore):
    """Per-mention polarity agreement for one document.

    ``gold_negated``/``python_negated`` hold indices into the aligned-pair list,
    so two occurrences of the same CUI stay distinct -- which is the whole point
    of scoring here rather than in :mod:`umlsmatch.eval.negation`.
    """

    source_file: str
    #: Gold occurrences that found a Python counterpart.
    aligned: int
    #: Gold occurrences with no same-CUI overlapping Python match. Not scored;
    #: these are concept-recall misses, see the module docstring.
    unaligned: int
    gold_negated: frozenset[int]
    python_negated: frozenset[int]

    @property
    def gold_set(self) -> frozenset[int]:
        return self.gold_negated

    @property
    def predicted_set(self) -> frozenset[int]:
        return self.python_negated

    @property
    def accuracy(self) -> float:
        """Polarity agreement across all aligned occurrences."""
        if not self.aligned:
            return 0.0
        agree = sum(
            1
            for i in range(self.aligned)
            if (i in self.gold_negated) == (i in self.python_negated)
        )
        return agree / self.aligned


@dataclass(frozen=True)
class AggregateSpanNegationScore(PrfAggregate):
    """:class:`PrfAggregate` plus the occurrence counts specific to this scorer."""

    n_aligned: int = 0
    n_unaligned: int = 0
    micro_accuracy: float = 0.0


def gold_mention_polarity(record: dict) -> list[Occurrence]:
    """One occurrence per (mention, concept) pair in a silver-standard record."""
    out: list[Occurrence] = []
    for mention in record["mentions"]:
        negated = bool(mention.get("negated", False))
        begin, end = int(mention["begin"]), int(mention["end"])
        for concept in mention["concepts"]:
            cui = concept.get("cui")
            if cui:
                out.append((begin, end, cui, negated))
    return out


def python_match_polarity(
    text: str, matcher: RareWordMatcher, *, max_scope: int | None = MAX_SCOPE_TOKENS
) -> list[Occurrence]:
    """One occurrence per Python match, with the polarity the rules assign it."""
    out: list[Occurrence] = []
    for tokens in annotate_sentences(text):
        matches = matcher.match(tokens)
        if not matches:
            continue
        negated = negated_matches(tokens, matches, max_scope=max_scope)
        out.extend((m.start, m.end, m.cui, m in negated) for m in matches)
    return out


def align_mentions(
    gold: list[Occurrence], predicted: list[Occurrence]
) -> tuple[list[tuple[Occurrence, Occurrence]], int]:
    """Pair gold occurrences with Python ones on same CUI + overlapping span.

    Returns ``(pairs, n_unaligned_gold)``. Each Python match is consumed at most
    once, so N gold occurrences of a CUI cannot all align to a single prediction
    and manufacture agreement. When several candidates overlap a gold span the
    one with the largest overlap wins; ties break on the earlier start for
    determinism.
    """
    by_cui: dict[str, list[int]] = {}
    for i, (_, _, cui, _) in enumerate(predicted):
        by_cui.setdefault(cui, []).append(i)

    used: set[int] = set()
    pairs: list[tuple[Occurrence, Occurrence]] = []
    unaligned = 0

    for g in gold:
        g_begin, g_end, g_cui, _ = g
        best_i = -1
        best_overlap = 0
        for i in by_cui.get(g_cui, ()):
            if i in used:
                continue
            p_begin, p_end, _, _ = predicted[i]
            overlap = min(g_end, p_end) - max(g_begin, p_begin)
            if overlap <= 0:
                continue
            # Larger overlap wins; equal overlap breaks on the earlier start.
            # `predicted[best_i]` is only reached once best_i is set, because
            # the first qualifying candidate always takes the first branch.
            if overlap > best_overlap or (
                overlap == best_overlap and p_begin < predicted[best_i][0]
            ):
                best_i, best_overlap = i, overlap
        if best_i < 0:
            unaligned += 1
            continue
        used.add(best_i)
        pairs.append((g, predicted[best_i]))

    return pairs, unaligned


def score_document(
    record: dict, matcher: RareWordMatcher, *, max_scope: int | None = MAX_SCOPE_TOKENS
) -> SpanNegationScore:
    gold = gold_mention_polarity(record)
    predicted = python_match_polarity(record["text"], matcher, max_scope=max_scope)
    pairs, unaligned = align_mentions(gold, predicted)

    gold_negated = frozenset(i for i, (g, _) in enumerate(pairs) if g[3])
    python_negated = frozenset(i for i, (_, p) in enumerate(pairs) if p[3])

    return SpanNegationScore(
        source_file=record["source_file"],
        aligned=len(pairs),
        unaligned=unaligned,
        gold_negated=gold_negated,
        python_negated=python_negated,
    )


def score_jsonl(
    jsonl_path: str | Path, db_path: str | Path, *, max_scope: int | None = MAX_SCOPE_TOKENS
) -> list[SpanNegationScore]:
    with RareWordMatcher(db_path) as matcher:
        return [
            score_document(record, matcher, max_scope=max_scope)
            for record in iter_records(jsonl_path)
        ]


def aggregate(scores: list[SpanNegationScore]) -> AggregateSpanNegationScore:
    """Micro (pooled counts) and macro (mean of per-doc scores) aggregates."""
    aligned = sum(s.aligned for s in scores)
    agree = sum(
        1
        for s in scores
        for i in range(s.aligned)
        if (i in s.gold_negated) == (i in s.python_negated)
    )
    return AggregateSpanNegationScore(
        **aggregate_prf(scores),
        n_aligned=aligned,
        n_unaligned=sum(s.unaligned for s in scores),
        micro_accuracy=safe_div(agree, aligned),
    )

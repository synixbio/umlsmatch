"""Any assertion attribute, scored per mention against the silver standard.

:mod:`umlsmatch.eval.negation_spans` does this for polarity, and does it the
right way: align each gold (mention, concept) pair to a Python match on same
CUI + overlapping span, then compare *on the aligned pair*, so a difference in
how many mentions each side emits cannot move the score. That reasoning is not
specific to polarity, and the four other attributes this pipeline assesses need
exactly the same treatment.

This module is that scorer, parameterized over
:class:`~umlsmatch.assertion.attributes.AttributeSpec`. Alignment is the same
function, so a number here reconciles with the negation figures rather than
being a second, subtly different measurement of the same corpus.

**Read every number here with its denominator.** The corpus carries 921
``negated`` positives and 27 ``conditional`` ones. A precision computed over 27
positives is not a smaller version of a precision computed over 921; it is a
number that will move by 0.04 if one mention is re-labelled.
:attr:`AttributeScore.gold_positives` is reported alongside P/R/F1 for that
reason, and :meth:`AggregateAttributeScore.caveat` spells out in words what the
count means -- the house style established by the negation work, where a metric
is published with a plain statement of what its reference is and is not good
for.

**The reference is cTAKES, and cTAKES is not truth.** The silver standard is
a sound development signal and an unsound acceptance gate.
Human labels are what :mod:`umlsmatch.eval.adjudication` exists to make
affordable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.assertion.attributes import (
    ATTRIBUTES,
    SUBJECT_FAMILY_MEMBER,
    AttributeSpec,
    get_attribute,
)
from umlsmatch.eval.metrics import PrfAggregate, PrfScore, aggregate_prf, safe_div
from umlsmatch.eval.negation_spans import Occurrence, align_mentions
from umlsmatch.eval.records import iter_records

__all__ = [
    "AggregateAttributeScore",
    "AttributeScore",
    "aggregate",
    "gold_occurrences",
    "predicted_occurrences",
    "score_corpus",
    "score_document",
]


@dataclass(frozen=True)
class AttributeScore(PrfScore):
    """One attribute's agreement with cTAKES over one document.

    ``gold_true``/``predicted_true`` hold indices into the aligned-pair list, so
    two occurrences of the same CUI stay distinct -- the same reason
    :mod:`umlsmatch.eval.negation_spans` scores positions rather than CUIs.
    """

    source_file: str
    attribute: str
    #: Gold occurrences that found a Python counterpart.
    aligned: int
    #: Gold occurrences with no same-CUI overlapping Python match. Not scored;
    #: these are concept-recall misses, measured by ``eval.parity``.
    unaligned: int
    gold_true: frozenset[int]
    predicted_true: frozenset[int]

    @property
    def gold_set(self) -> frozenset[int]:
        return self.gold_true

    @property
    def predicted_set(self) -> frozenset[int]:
        return self.predicted_true

    @property
    def gold_positives(self) -> int:
        return len(self.gold_true)

    @property
    def accuracy(self) -> float:
        """Agreement across all aligned occurrences.

        Nearly useless on its own for the rare attributes and reported anyway,
        because seeing 99.8% next to an F1 of 0.0 is the clearest statement
        available that the majority class is doing all the work.
        """
        if not self.aligned:
            return 0.0
        agree = sum(
            1
            for i in range(self.aligned)
            if (i in self.gold_true) == (i in self.predicted_true)
        )
        return agree / self.aligned


@dataclass(frozen=True)
class AggregateAttributeScore(PrfAggregate):
    """Corpus aggregate for one attribute, with the counts that qualify it."""

    attribute: str = ""
    n_aligned: int = 0
    n_unaligned: int = 0
    micro_accuracy: float = 0.0
    #: Gold positives pooled over the corpus. The denominator behind the recall.
    n_gold_positives: int = 0
    #: Predicted positives pooled over the corpus.
    n_predicted_positives: int = 0
    #: Disagreements by predicted/gold direction, for error attribution.
    false_positive_texts: Counter = field(default_factory=Counter)
    false_negative_texts: Counter = field(default_factory=Counter)

    @property
    def spec(self) -> AttributeSpec:
        return get_attribute(self.attribute)

    def caveat(self) -> str:
        """What this number is and is not good for, in words, for publication.

        Returned rather than logged so every caller prints the same sentence.
        A score for ``conditional`` that appears without one reads as a result.
        """
        spec = self.spec
        base = (
            f"Reference: Java cTAKES' own {spec.ctakes_name} on "
            f"{self.n_docs:,} notes ({self.n_gold_positives:,} positives among "
            f"{self.n_aligned:,} aligned mentions). Agreement with cTAKES, not "
            "correctness"
        )
        if spec.prototype:
            # Stronger than the unmeasurable caveat and deliberately so. That
            # one qualifies a number the project publishes; this one says the
            # number is not a result at all, which is the claim that has to
            # survive being copied out of a terminal into an issue.
            return (
                f"{base}. {spec.name} is a PROTOTYPE, off by default, and this "
                "figure is a liveness check only: it says whether the rules fire "
                "at a sane volume, nothing more. Do not quote it. Run "
                f"tools/make_adjudication_set.py --attribute {spec.name} for a "
                "number that means something."
            )
        if not spec.measurable_against_ctakes:
            # Deliberately does NOT blame the positive count. That was the
            # stated reason here for a long time and docs/ADJUDICATION_RESULTS.md
            # disproved it: for `uncertain` the reference is 97% noise, which is
            # a fact about cTAKES and not about the sample size. Printing the
            # wrong reason is worse than printing none, because it points the
            # next person at a bigger corpus instead of at adjudication.
            return (
                f"{base}. This number does not support a decision about the "
                f"rules -- see the note above for why, and "
                f"docs/ADJUDICATION_RESULTS.md for what a stratified sample of "
                f"{spec.name}'s own positives estimates instead. Do not gate on it."
            )
        return f"{base} -- the reference is cTAKES, which is not truth."


def gold_occurrences(record: dict, spec: AttributeSpec) -> list[Occurrence]:
    """One occurrence per (mention, concept) pair, carrying `spec`'s boolean."""
    out: list[Occurrence] = []
    for mention in record["mentions"]:
        value = spec.gold(mention)
        begin, end = int(mention["begin"]), int(mention["end"])
        for concept in mention["concepts"]:
            cui = concept.get("cui")
            if cui:
                out.append((begin, end, cui, value))
    return out


def predicted_occurrences(annotations, spec: AttributeSpec) -> list[Occurrence]:
    """One occurrence per annotation, carrying `spec`'s boolean.

    An annotation whose attribute is ``None`` counts as **false**, not as
    excluded. That is deliberate and it is the pessimistic reading: a pipeline
    that assesses nothing scores zero recall rather than scoring nothing at
    all. Silently dropping unassessed mentions would let an unimplemented
    attribute report a flattering precision over the handful of cases some
    other rule happened to touch.
    """
    out: list[Occurrence] = []
    for a in annotations:
        value = getattr(a, spec.name)
        if spec.name == "subject":
            value = value == SUBJECT_FAMILY_MEMBER
        out.append((a.start, a.end, a.cui, bool(value)))
    return out


def score_document(
    record: dict, annotations, spec: AttributeSpec
) -> tuple[AttributeScore, Counter, Counter]:
    """Score one document, and return the disagreeing mention texts with it.

    The two counters are false-positive and false-negative mention texts. They
    are collected here rather than in a second pass because the alignment is
    already in hand, and "which mentions do we get wrong" is the first question
    asked of any of these scores.
    """
    text = record["text"]
    gold = gold_occurrences(record, spec)
    predicted = predicted_occurrences(annotations, spec)
    pairs, unaligned = align_mentions(gold, predicted)

    gold_true = frozenset(i for i, (g, _) in enumerate(pairs) if g[3])
    predicted_true = frozenset(i for i, (_, p) in enumerate(pairs) if p[3])

    false_positives: Counter = Counter()
    false_negatives: Counter = Counter()
    for i, (_, p) in enumerate(pairs):
        if i in predicted_true and i not in gold_true:
            false_positives[text[p[0] : p[1]].casefold()] += 1
        elif i in gold_true and i not in predicted_true:
            false_negatives[text[p[0] : p[1]].casefold()] += 1

    score = AttributeScore(
        source_file=record["source_file"],
        attribute=spec.name,
        aligned=len(pairs),
        unaligned=unaligned,
        gold_true=gold_true,
        predicted_true=predicted_true,
    )
    return score, false_positives, false_negatives


def score_corpus(
    jsonl_path: str | Path,
    pipeline,
    attribute: str | AttributeSpec,
) -> AggregateAttributeScore:
    """Score one attribute over a silver-standard corpus.

    `pipeline` is anything with ``analyze(text) -> list[Annotation]``; it is
    passed in rather than built here so a caller can sweep a rule setting
    without re-opening a 589 MB dictionary for each variant.
    """
    spec = attribute if isinstance(attribute, AttributeSpec) else get_attribute(attribute)
    scores: list[AttributeScore] = []
    false_positives: Counter = Counter()
    false_negatives: Counter = Counter()

    for record in iter_records(jsonl_path):
        annotations = pipeline.analyze(record["text"])
        score, fp, fn = score_document(record, annotations, spec)
        scores.append(score)
        false_positives += fp
        false_negatives += fn

    return aggregate(scores, false_positives, false_negatives)


def aggregate(
    scores: list[AttributeScore],
    false_positive_texts: Counter | None = None,
    false_negative_texts: Counter | None = None,
) -> AggregateAttributeScore:
    """Micro (pooled counts) and macro (mean of per-doc scores) aggregates."""
    aligned = sum(s.aligned for s in scores)
    agree = sum(
        1
        for s in scores
        for i in range(s.aligned)
        if (i in s.gold_true) == (i in s.predicted_true)
    )
    return AggregateAttributeScore(
        **aggregate_prf(scores),
        attribute=scores[0].attribute if scores else "",
        n_aligned=aligned,
        n_unaligned=sum(s.unaligned for s in scores),
        micro_accuracy=safe_div(agree, aligned),
        n_gold_positives=sum(len(s.gold_true) for s in scores),
        n_predicted_positives=sum(len(s.predicted_true) for s in scores),
        false_positive_texts=false_positive_texts or Counter(),
        false_negative_texts=false_negative_texts or Counter(),
    )


def scorable_attributes() -> tuple[AttributeSpec, ...]:
    """The attributes worth running this over: the ones the pipeline assesses."""
    return tuple(a for a in ATTRIBUTES if a.supported)

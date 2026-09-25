"""CUI-level parity scoring: Python matcher output vs. Java cTAKES silver standard.

The parity harness README's Validation section sets the concept-extraction
target on ("(greater-equal)90% F1 against cTAKES on CUI-level entity
extraction").

Silver-standard records come from ``tools/run_java_ctakes.py`` (real Java
cTAKES output flattened to JSONL). This module runs the Python tokenizer +
``RareWordMatcher`` over the *same* raw text and compares the sets of CUIs
each side found, per document.

Deliberately CUI-level, not span-level: Java's DefaultFastPipeline and the
Python matcher segment mentions differently (chunk-based NP mentions vs.
sentence-windowed rare-word anchors), so span boundaries will disagree even
when both sides correctly identify the same concept. Comparing at the CUI
set level asks the question the target actually gates on -- "did we find the
same concepts" -- without span alignment noise. Treat this as a first-cut
metric: it will not catch a right-CUI-wrong-place error.

**Read the score against the dictionary it was produced with.** Both
confounds below have been measured, and the results invert the naive
reading of a low number:

  * **Dictionary release dominates.** Scored against cTAKES' own shipped
    2016AB dictionary -- the one that generated the silver standard --
    this reports F1 0.971 (P 0.982 / R 0.960), meeting the >=0.90
    target. The default build is UMLS 2026AA and scores 0.756-0.782
    depending on synonym sources. That spread is release and build
    configuration, not matcher defect; treat a matched-dictionary run as
    the measure of the algorithm and a modern-release run as the measure
    of the build.
  * **POS divergence is not the lever.** The bridge does use general-domain
    spaCy (`en_core_web_sm`) rather than a clinically-trained tagger, but
    substituting cTAKES' own tokens *and* POS moved precision by +0.009.
    Anchor-eligibility agreement is 0.967; the dominant
    `NN -> NNP` confusion is harmless because both tags anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from umlsmatch.dictionary.matcher import RareWordMatcher
from umlsmatch.eval.metrics import PrfAggregate, PrfScore, aggregate_prf
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "AggregateScore",
    "DocScore",
    "aggregate",
    "python_cuis",
    "score_document",
    "score_jsonl",
    "silver_cuis",
]


@dataclass(frozen=True)
class DocScore(PrfScore):
    """One document's CUI-set agreement. P/R/F1 come from :class:`PrfScore`."""

    source_file: str
    silver_cuis: frozenset[str]
    python_cuis: frozenset[str]

    @property
    def gold_set(self) -> frozenset[str]:
        return self.silver_cuis

    @property
    def predicted_set(self) -> frozenset[str]:
        return self.python_cuis


@dataclass(frozen=True)
class AggregateScore(PrfAggregate):
    """Corpus-level CUI-parity aggregate; fields are :class:`PrfAggregate`'s."""


def silver_cuis(record: dict) -> frozenset[str]:
    """CUIs Java cTAKES attached to any mention in one silver-standard record."""
    return frozenset(
        concept["cui"]
        for mention in record["mentions"]
        for concept in mention["concepts"]
        if concept.get("cui")
    )


def python_cuis(text: str, matcher: RareWordMatcher) -> frozenset[str]:
    """CUIs the Python matcher finds over the same raw text, sentence by sentence."""
    cuis: set[str] = set()
    for tokens in annotate_sentences(text):
        cuis.update(match.cui for match in matcher.match(tokens))
    return frozenset(cuis)


def score_document(record: dict, matcher: RareWordMatcher) -> DocScore:
    return DocScore(
        source_file=record["source_file"],
        silver_cuis=silver_cuis(record),
        python_cuis=python_cuis(record["text"], matcher),
    )


def score_jsonl(jsonl_path: str | Path, db_path: str | Path) -> list[DocScore]:
    """Score every record in a silver-standard JSONL file (one line per document)."""
    with RareWordMatcher(db_path) as matcher:
        return [score_document(record, matcher) for record in iter_records(jsonl_path)]


def aggregate(scores: list[DocScore]) -> AggregateScore:
    """Micro (pooled counts) and macro (mean of per-doc scores) aggregates."""
    return AggregateScore(**aggregate_prf(scores))

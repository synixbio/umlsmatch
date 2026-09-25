"""Negation agreement: Python rule-based negation vs. Java cTAKES polarity.

Measures the negation target README's Validation section sets
("(greater-eq)90% F1 vs. cTAKES"). Scoped like ``eval.parity``: CUI-level, and
restricted to CUIs *both* sides found in a document. The CUI-matching
problem itself is already characterized separately by
``tools/score_parity.py`` / ``tools/diff_parity.py`` -- this module isolates
a different question: of the concepts both sides agree exist, do we agree
on negated vs. affirmed?

A CUI's polarity is collapsed per document: if any mention carrying that CUI
is negated, the CUI counts as negated for that document (mirrors how
``eval.parity.silver_cuis`` already collapses per-mention info to a
per-document CUI set).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from umlsmatch.assertion.negation import negated_matches
from umlsmatch.dictionary.matcher import RareWordMatcher
from umlsmatch.eval.metrics import PrfAggregate, PrfScore, aggregate_prf, safe_div
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "AggregateNegationScore",
    "NegationScore",
    "aggregate",
    "gold_cui_polarity",
    "python_cui_polarity",
    "score_document",
    "score_jsonl",
]


@dataclass(frozen=True)
class NegationScore(PrfScore):
    """One document's polarity agreement over commonly-found CUIs.

    P/R/F1 come from :class:`PrfScore` and score the *negated* subsets;
    :attr:`accuracy` additionally credits agreeing on "affirmed", which P/R/F1
    by construction never see.
    """

    source_file: str
    #: CUIs both Java and Python found in this document at all.
    common_cuis: frozenset[str]
    #: Subset of `common_cuis` Java marked negated.
    gold_negated: frozenset[str]
    #: Subset of `common_cuis` Python marked negated.
    python_negated: frozenset[str]

    @property
    def gold_set(self) -> frozenset[str]:
        return self.gold_negated

    @property
    def predicted_set(self) -> frozenset[str]:
        return self.python_negated

    @property
    def accuracy(self) -> float:
        """Agreement rate on polarity across all commonly-found CUIs."""
        if not self.common_cuis:
            return 0.0
        agree = sum(
            1
            for cui in self.common_cuis
            if (cui in self.gold_negated) == (cui in self.python_negated)
        )
        return agree / len(self.common_cuis)


@dataclass(frozen=True)
class AggregateNegationScore(PrfAggregate):
    """:class:`PrfAggregate` plus the two figures specific to polarity scoring.

    Both default so the inherited fields keep their positional order; every
    construction site passes them by keyword.
    """

    #: Total CUIs both sides found, pooled across documents -- the denominator
    #: :attr:`micro_accuracy` is taken over.
    n_common_cuis: int = 0
    #: Pooled agreement rate on polarity, counting affirmed/affirmed matches.
    micro_accuracy: float = 0.0


def gold_cui_polarity(record: dict) -> dict[str, bool]:
    """CUI -> negated, collapsed across all mentions carrying that CUI in this document.

    A mention with no ``negated`` key counts as affirmed rather than raising:
    absent polarity is exactly what cTAKES' default (``polarity=1``) means, and
    aborting a whole corpus run over one legacy record helps nobody.
    """
    polarity: dict[str, bool] = {}
    for mention in record["mentions"]:
        negated = bool(mention.get("negated", False))
        for concept in mention["concepts"]:
            cui = concept.get("cui")
            if cui:
                polarity[cui] = polarity.get(cui, False) or negated
    return polarity


def python_cui_polarity(text: str, matcher: RareWordMatcher) -> dict[str, bool]:
    """CUI -> negated, collapsed across all Python matches carrying that CUI."""
    polarity: dict[str, bool] = {}
    for tokens in annotate_sentences(text):
        matches = matcher.match(tokens)
        negated = negated_matches(tokens, matches)
        for m in matches:
            polarity[m.cui] = polarity.get(m.cui, False) or (m in negated)
    return polarity


def score_document(record: dict, matcher: RareWordMatcher) -> NegationScore:
    gold_polarity = gold_cui_polarity(record)
    python_polarity = python_cui_polarity(record["text"], matcher)

    common = frozenset(gold_polarity) & frozenset(python_polarity)
    gold_negated = frozenset(c for c in common if gold_polarity[c])
    python_negated = frozenset(c for c in common if python_polarity[c])

    return NegationScore(record["source_file"], common, gold_negated, python_negated)


def score_jsonl(jsonl_path: str | Path, db_path: str | Path) -> list[NegationScore]:
    with RareWordMatcher(db_path) as matcher:
        return [score_document(record, matcher) for record in iter_records(jsonl_path)]


def aggregate(scores: list[NegationScore]) -> AggregateNegationScore:
    """Micro (pooled counts) and macro (mean of per-doc scores) aggregates."""
    common = sum(len(s.common_cuis) for s in scores)
    agree = sum(
        1
        for s in scores
        for cui in s.common_cuis
        if (cui in s.gold_negated) == (cui in s.python_negated)
    )
    return AggregateNegationScore(
        **aggregate_prf(scores),
        n_common_cuis=common,
        micro_accuracy=safe_div(agree, common),
    )

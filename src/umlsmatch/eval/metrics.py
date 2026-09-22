"""Precision/recall/F1 machinery shared by the set-comparison scorers.

Three modules in this package -- :mod:`~umlsmatch.eval.parity` (CUI sets),
:mod:`~umlsmatch.eval.boundaries` (span sets) and
:mod:`~umlsmatch.eval.negation` (negated-CUI sets) -- all score the same shape
of problem: a gold set against a predicted set, per document, then micro and
macro aggregates over the corpus. Defining ``_safe_div``, the
``precision``/``recall``/``f1`` properties and ``aggregate()`` once keeps
"what does macro-F1 mean here" from having three answers that merely happen to
agree, and means a fix to the averaging reaches every scorer at once.

:mod:`~umlsmatch.eval.pos` deliberately does not use this: it reports agreement
rates and confusion counts, not set-overlap P/R/F1, so there is no gold/predicted
pair to share.

Zero denominators return 0.0 rather than raising. A document where the matcher
found nothing, or one with no gold annotations, is a real and unremarkable case
in a corpus run; aborting the run over it helps nobody, and 0.0 is what both
micro-averaging and a human reader expect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["PrfAggregate", "PrfScore", "aggregate_prf", "safe_div"]


def safe_div(numerator: float, denominator: float) -> float:
    """Divide, treating a zero denominator as 0.0. See the module docstring."""
    return numerator / denominator if denominator else 0.0


class PrfScore:
    """Mixin supplying P/R/F1 to a per-document score over two sets.

    Subclasses store their two sets under whatever names read best in context
    (``silver_cuis``/``python_cuis``, ``gold``/``predicted``, ...) and expose
    them here via :attr:`gold_set` and :attr:`predicted_set`. Keeping the
    domain-specific names on the dataclass and the generic ones on the mixin
    means the scoring math is written once without forcing every scorer to call
    its CUIs "gold".
    """

    @property
    def gold_set(self) -> frozenset[Any]:
        raise NotImplementedError

    @property
    def predicted_set(self) -> frozenset[Any]:
        raise NotImplementedError

    @property
    def true_positives(self) -> frozenset[Any]:
        return self.gold_set & self.predicted_set

    @property
    def precision(self) -> float:
        return safe_div(len(self.true_positives), len(self.predicted_set))

    @property
    def recall(self) -> float:
        return safe_div(len(self.true_positives), len(self.gold_set))

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return safe_div(2 * p * r, p + r)


@dataclass(frozen=True)
class PrfAggregate:
    """Corpus-level micro and macro aggregates.

    Micro pools the raw counts across documents, so a long document weighs more
    than a short one. Macro averages the per-document scores, so every document
    weighs the same. They answer different questions and are both reported
    rather than one being picked for the reader.
    """

    n_docs: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def aggregate_prf(scores: Sequence[PrfScore]) -> dict[str, float]:
    """Micro/macro aggregate fields for `scores`, ready to splat into a dataclass.

    Returned as a dict rather than a :class:`PrfAggregate` so callers that add
    their own fields (``eval.negation`` carries an accuracy and a CUI count)
    can build their subclass in one expression.

    An empty `scores` yields all-zero values instead of raising -- scoring a
    corpus that produced no records is a reporting problem for the caller to
    surface, not an arithmetic error here.
    """
    tp = sum(len(s.true_positives) for s in scores)
    predicted = sum(len(s.predicted_set) for s in scores)
    gold = sum(len(s.gold_set) for s in scores)
    micro_p = safe_div(tp, predicted)
    micro_r = safe_div(tp, gold)
    n = len(scores) or 1
    return {
        "n_docs": len(scores),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": safe_div(2 * micro_p * micro_r, micro_p + micro_r),
        "macro_precision": sum(s.precision for s in scores) / n,
        "macro_recall": sum(s.recall for s in scores) / n,
        "macro_f1": sum(s.f1 for s in scores) / n,
    }

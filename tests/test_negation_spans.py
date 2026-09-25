"""Tests for per-mention negation scoring and its span alignment.

The alignment is the load-bearing part: the whole point of this scorer is that
mention-count differences between the two sides cannot move the score, and that
property only holds if each prediction is consumed at most once. These run on
synthetic occurrence tuples, so no dictionary or spaCy model is needed.
"""

from __future__ import annotations

from umlsmatch.eval.negation_spans import (
    SpanNegationScore,
    aggregate,
    align_mentions,
    gold_mention_polarity,
)

# --- alignment ---------------------------------------------------------------


def test_aligns_on_overlap_not_exact_span():
    """cTAKES chunks mentions, the matcher anchors on rare words -- spans differ."""
    gold = [(10, 25, "C1", True)]
    predicted = [(15, 20, "C1", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert len(pairs) == 1
    assert unaligned == 0


def test_does_not_align_across_different_cuis():
    gold = [(10, 20, "C1", True)]
    predicted = [(10, 20, "C2", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert pairs == []
    assert unaligned == 1


def test_does_not_align_disjoint_spans():
    gold = [(10, 20, "C1", True)]
    predicted = [(50, 60, "C1", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert pairs == []
    assert unaligned == 1


def test_touching_spans_do_not_count_as_overlapping():
    """end == begin is adjacency, not overlap."""
    gold = [(10, 20, "C1", True)]
    predicted = [(20, 30, "C1", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert pairs == []
    assert unaligned == 1


def test_largest_overlap_wins():
    gold = [(10, 20, "C1", True)]
    predicted = [(19, 30, "C1", False), (10, 20, "C1", True)]
    (pair,), _ = align_mentions(gold, predicted)
    assert pair[1] == (10, 20, "C1", True)


def test_each_prediction_is_consumed_at_most_once():
    """Two gold occurrences must not both align to one prediction.

    This is what stops the scorer from manufacturing agreement when the two
    sides emit different mention counts -- the bias this module exists to
    remove.
    """
    gold = [(10, 20, "C1", True), (10, 20, "C1", True)]
    predicted = [(10, 20, "C1", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert len(pairs) == 1
    assert unaligned == 1


def test_extra_predictions_do_not_create_pairs():
    """Overlapping matcher hits for one gold mention align once, not three."""
    gold = [(0, 10, "C1", False)]
    predicted = [(0, 5, "C1", True), (0, 10, "C1", False), (6, 10, "C1", True)]
    pairs, unaligned = align_mentions(gold, predicted)
    assert len(pairs) == 1
    assert unaligned == 0
    assert pairs[0][1][3] is False  # the best-overlap match, not the first


# --- gold extraction ---------------------------------------------------------


def test_gold_mention_polarity_emits_one_occurrence_per_concept():
    record = {
        "mentions": [
            {"begin": 0, "end": 5, "negated": True, "concepts": [{"cui": "C1"}, {"cui": "C2"}]},
            {"begin": 9, "end": 12, "negated": False, "concepts": [{"cui": "C1"}]},
        ]
    }
    assert gold_mention_polarity(record) == [
        (0, 5, "C1", True),
        (0, 5, "C2", True),
        (9, 12, "C1", False),
    ]


def test_gold_mention_polarity_keeps_repeat_cuis_distinct():
    """The per-document scorer collapses these to one; this one must not."""
    record = {
        "mentions": [
            {"begin": 0, "end": 5, "negated": True, "concepts": [{"cui": "C1"}]},
            {"begin": 9, "end": 14, "negated": False, "concepts": [{"cui": "C1"}]},
        ]
    }
    assert len(gold_mention_polarity(record)) == 2


def test_gold_mention_polarity_treats_missing_negated_as_affirmed():
    record = {"mentions": [{"begin": 0, "end": 5, "concepts": [{"cui": "C1"}]}]}
    assert gold_mention_polarity(record) == [(0, 5, "C1", False)]


def test_gold_mention_polarity_skips_concepts_without_a_cui():
    record = {
        "mentions": [{"begin": 0, "end": 5, "negated": False, "concepts": [{"cui": None}]}]
    }
    assert gold_mention_polarity(record) == []


# --- scoring math ------------------------------------------------------------


def _score(aligned, gold_neg, py_neg, unaligned=0):
    return SpanNegationScore("a.txt", aligned, unaligned, frozenset(gold_neg), frozenset(py_neg))


def test_perfect_agreement():
    s = _score(3, {0}, {0})
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.accuracy == 1.0


def test_over_negation_costs_precision_not_recall():
    s = _score(4, {0}, {0, 1, 2})
    assert s.recall == 1.0
    assert s.precision == 1 / 3
    assert s.accuracy == 0.5  # indices 0,3 agree; 1,2 do not


def test_no_negation_on_either_side_is_not_a_zero_division():
    s = _score(5, set(), set())
    assert s.precision == 0.0
    assert s.f1 == 0.0
    assert s.accuracy == 1.0  # all five agree they are affirmed


def test_zero_aligned_accuracy_is_zero_not_error():
    assert _score(0, set(), set()).accuracy == 0.0


def test_aggregate_pools_counts_and_carries_occurrence_totals():
    scores = [_score(2, {0}, {0}, unaligned=1), _score(2, {0}, {0, 1})]
    agg = aggregate(scores)
    assert agg.n_docs == 2
    assert agg.n_aligned == 4
    assert agg.n_unaligned == 1
    # tp=2, predicted=3, gold=2
    assert round(agg.micro_precision, 4) == round(2 / 3, 4)
    assert agg.micro_recall == 1.0
    assert round(agg.micro_accuracy, 4) == round(3 / 4, 4)


def test_aggregate_of_empty_list_does_not_raise():
    agg = aggregate([])
    assert agg.n_docs == 0
    assert agg.n_aligned == 0
    assert agg.micro_f1 == 0.0
    assert agg.micro_accuracy == 0.0

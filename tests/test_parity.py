"""Tests for CUI-level parity scoring math (DocScore/AggregateScore).

These exercise the scoring dataclasses directly with synthetic CUI sets, so
they run without a built dictionary or spaCy model -- see test_dictionary.py
and test_spacy_tokenizer.py for the tests that need those.
"""

from __future__ import annotations

from umlsmatch.eval.parity import DocScore, aggregate, silver_cuis


def test_perfect_match():
    s = DocScore("a.txt", frozenset({"C1", "C2"}), frozenset({"C1", "C2"}))
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_partial_overlap():
    # silver={C1,C2,C3}, python={C1,C2,C4}: tp={C1,C2}
    s = DocScore("a.txt", frozenset({"C1", "C2", "C3"}), frozenset({"C1", "C2", "C4"}))
    assert s.precision == 2 / 3
    assert s.recall == 2 / 3
    assert abs(s.f1 - 2 / 3) < 1e-9


def test_empty_python_output_scores_zero_not_error():
    s = DocScore("a.txt", frozenset({"C1"}), frozenset())
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_empty_silver_and_empty_python_is_not_a_zero_division():
    s = DocScore("a.txt", frozenset(), frozenset())
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_silver_cuis_extracts_from_record_shape():
    record = {
        "mentions": [
            {"concepts": [{"cui": "C1"}, {"cui": "C2"}]},
            {"concepts": []},
            {"concepts": [{"cui": "C1"}]},  # duplicate CUI across mentions
        ]
    }
    assert silver_cuis(record) == frozenset({"C1", "C2"})


def test_aggregate_micro_pools_counts_macro_averages_per_doc():
    scores = [
        DocScore("a.txt", frozenset({"C1", "C2"}), frozenset({"C1"})),  # P=1, R=.5
        DocScore("b.txt", frozenset({"C3"}), frozenset({"C3", "C4"})),  # P=.5, R=1
    ]
    agg = aggregate(scores)
    assert agg.n_docs == 2
    # micro: tp=2, python=3, silver=3
    assert round(agg.micro_precision, 4) == round(2 / 3, 4)
    assert round(agg.micro_recall, 4) == round(2 / 3, 4)
    # macro: mean of per-doc precision/recall
    assert round(agg.macro_precision, 4) == round((1.0 + 0.5) / 2, 4)
    assert round(agg.macro_recall, 4) == round((0.5 + 1.0) / 2, 4)


def test_aggregate_of_empty_list_does_not_raise():
    agg = aggregate([])
    assert agg.n_docs == 0
    assert agg.micro_f1 == 0.0
    assert agg.macro_f1 == 0.0

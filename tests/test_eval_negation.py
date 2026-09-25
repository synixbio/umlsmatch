"""Tests for negation-agreement scoring math (NegationScore/aggregate).

Pure dataclass logic with synthetic CUI sets -- runs without a built
dictionary or spaCy model. See test_negation.py for the trigger/scope logic
itself, and score_negation.py's manual runs for score_document() integration
coverage (it calls umlsmatch.pipeline.tokenizer and the matcher directly).
"""

from __future__ import annotations

from umlsmatch.eval.negation import NegationScore, aggregate, gold_cui_polarity


def test_perfect_agreement():
    s = NegationScore("a.txt", frozenset({"C1", "C2", "C3"}), frozenset({"C1"}), frozenset({"C1"}))
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0
    # All three commonly-found CUIs agree: C1 negated, C2/C3 affirmed on both sides.
    assert s.accuracy == 1.0


def test_disagreement_on_one_cui():
    # gold says C1 negated; python says C2 negated. Neither overlaps.
    s = NegationScore("a.txt", frozenset({"C1", "C2"}), frozenset({"C1"}), frozenset({"C2"}))
    assert s.true_positives == frozenset()
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.accuracy == 0.0


def test_no_negation_anywhere_is_perfect_agreement_not_zero_division():
    s = NegationScore("a.txt", frozenset({"C1", "C2"}), frozenset(), frozenset())
    assert s.precision == 0.0  # 0/0 -- no python-negated predictions to be precise about
    assert s.recall == 0.0
    assert s.accuracy == 1.0  # both sides agree: neither marked anything negated


def test_empty_common_cuis_accuracy_is_zero_not_error():
    s = NegationScore("a.txt", frozenset(), frozenset(), frozenset())
    assert s.accuracy == 0.0
    assert s.f1 == 0.0


def test_gold_cui_polarity_collapses_across_mentions():
    record = {
        "mentions": [
            {"negated": False, "concepts": [{"cui": "C1"}]},
            {"negated": True, "concepts": [{"cui": "C1"}]},  # same CUI, negated elsewhere
            {"negated": False, "concepts": [{"cui": "C2"}]},
        ]
    }
    polarity = gold_cui_polarity(record)
    assert polarity == {"C1": True, "C2": False}


def test_gold_cui_polarity_treats_a_missing_negated_key_as_affirmed():
    """cTAKES' default polarity is 1 (asserted); a legacy export omitting the
    field must not abort a whole corpus run with a KeyError."""
    record = {"mentions": [{"concepts": [{"cui": "C1"}]}]}
    assert gold_cui_polarity(record) == {"C1": False}


def test_gold_cui_polarity_skips_concepts_without_a_cui():
    record = {"mentions": [{"negated": True, "concepts": [{"cui": None}, {"cui": "C1"}]}]}
    assert gold_cui_polarity(record) == {"C1": True}


def test_aggregate_micro_pools_counts_macro_averages_per_doc():
    scores = [
        NegationScore("a.txt", frozenset({"C1", "C2"}), frozenset({"C1"}), frozenset({"C1"})),
        NegationScore("b.txt", frozenset({"C3"}), frozenset({"C3"}), frozenset()),
    ]
    agg = aggregate(scores)
    assert agg.n_docs == 2
    assert agg.n_common_cuis == 3
    # micro: tp=1, python_negated=1, gold_negated=2
    assert agg.micro_precision == 1.0
    assert agg.micro_recall == 0.5
    # doc b: no python-negated predictions -> precision 0.0 by convention
    assert round(agg.macro_precision, 4) == round((1.0 + 0.0) / 2, 4)


def test_aggregate_of_empty_list_does_not_raise():
    agg = aggregate([])
    assert agg.n_docs == 0
    assert agg.micro_f1 == 0.0
    assert agg.micro_accuracy == 0.0

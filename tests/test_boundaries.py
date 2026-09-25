"""Tests for exact-span-match boundary scoring (BoundaryScore/aggregate).

Pure dataclass logic with synthetic spans -- runs without a built dictionary
or spaCy model. See test_spacy_tokenizer.py for the tests that need spaCy,
and score_boundaries.py's manual runs for score_document()/score_jsonl()
integration coverage (they call umlsmatch.pipeline.tokenizer directly).
"""

from __future__ import annotations

from umlsmatch.eval.boundaries import BoundaryScore, aggregate, gold_token_spans


def test_perfect_match():
    s = BoundaryScore("a.txt", frozenset({(0, 5), (6, 10)}), frozenset({(0, 5), (6, 10)}))
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_one_character_offset_is_a_full_miss():
    # Exact-match by design: an off-by-one span does not partially count.
    s = BoundaryScore("a.txt", frozenset({(0, 5)}), frozenset({(0, 6)}))
    assert s.true_positives == frozenset()
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_partial_overlap():
    gold = frozenset({(0, 5), (6, 10), (11, 15)})
    pred = frozenset({(0, 5), (6, 10), (99, 100)})
    s = BoundaryScore("a.txt", gold, pred)
    assert s.true_positives == frozenset({(0, 5), (6, 10)})
    assert round(s.precision, 4) == round(2 / 3, 4)
    assert round(s.recall, 4) == round(2 / 3, 4)


def test_empty_predicted_scores_zero_not_error():
    s = BoundaryScore("a.txt", frozenset({(0, 5)}), frozenset())
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_empty_gold_and_empty_predicted_is_not_a_zero_division():
    s = BoundaryScore("a.txt", frozenset(), frozenset())
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_aggregate_micro_pools_counts_macro_averages_per_doc():
    scores = [
        BoundaryScore("a.txt", frozenset({(0, 5), (6, 10)}), frozenset({(0, 5)})),  # P=1, R=.5
        BoundaryScore("b.txt", frozenset({(0, 5)}), frozenset({(0, 5), (6, 10)})),  # P=.5, R=1
    ]
    agg = aggregate(scores)
    assert agg.n_docs == 2
    # micro: tp=2, predicted=3, gold=3
    assert round(agg.micro_precision, 4) == round(2 / 3, 4)
    assert round(agg.micro_recall, 4) == round(2 / 3, 4)
    assert round(agg.macro_precision, 4) == round((1.0 + 0.5) / 2, 4)
    assert round(agg.macro_recall, 4) == round((0.5 + 1.0) / 2, 4)


def test_aggregate_of_empty_list_does_not_raise():
    agg = aggregate([])
    assert agg.n_docs == 0
    assert agg.micro_f1 == 0.0
    assert agg.macro_f1 == 0.0


# --- gold-side newline filtering --------------------------------------------
# cTAKES emits a NewlineToken but drops it from a lookup window, and the Python
# tokenizer emits no whitespace token at all. Scoring them would count every
# line break in the corpus as a Python miss.


def test_gold_token_spans_drops_newline_tokens():
    record = {
        "pos_tokens": [
            [0, 5, "WordToken", "NN"],
            [5, 6, "NewlineToken", None],
            [6, 10, "WordToken", "NN"],
        ]
    }
    assert gold_token_spans(record) == frozenset({(0, 5), (6, 10)})


def test_gold_token_spans_falls_back_to_untyped_tokens():
    """Records exported before pos_tokens existed still score, unfiltered."""
    record = {"tokens": [[0, 5], [6, 10]]}
    assert gold_token_spans(record) == frozenset({(0, 5), (6, 10)})


def test_gold_token_spans_prefers_pos_tokens_over_tokens():
    record = {
        "pos_tokens": [[0, 5, "WordToken", "NN"], [5, 6, "NewlineToken", None]],
        "tokens": [[0, 5], [5, 6]],
    }
    assert gold_token_spans(record) == frozenset({(0, 5)})

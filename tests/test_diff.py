"""Tests for CUI disagreement aggregation (CuiEvidence/top).

Pure dataclass logic -- runs without a built dictionary or spaCy model. See
test_dictionary.py / test_spacy_tokenizer.py for the tests that need those,
and score_parity.py's manual runs for diff_corpus() integration coverage.
"""

from __future__ import annotations

from umlsmatch.eval.diff import CuiEvidence, top


def test_add_accumulates_count_and_docs():
    ev = CuiEvidence("C1")
    ev.add("aspirin", ("NN",), "doc1.txt")
    ev.add("Aspirin", ("NNP",), "doc2.txt")
    ev.add("aspirin", ("NN",), "doc1.txt")  # same doc again

    assert ev.count == 3
    assert ev.docs == {"doc1.txt", "doc2.txt"}


def test_add_caps_examples_but_not_count():
    ev = CuiEvidence("C1")
    for i in range(10):
        ev.add(f"text{i}", (), f"doc{i}.txt")

    assert ev.count == 10
    assert len(ev.example_texts) == 3
    assert ev.example_texts == ["text0", "text1", "text2"]


def test_preferred_text_set_once_first_wins():
    ev = CuiEvidence("C1")
    ev.add("x", (), "d1", preferred_text="First Name")
    ev.add("y", (), "d2", preferred_text="Second Name")

    assert ev.preferred_text == "First Name"


def test_preferred_text_only_overwritten_from_none():
    ev = CuiEvidence("C1", preferred_text=None)
    ev.add("x", (), "d1", preferred_text=None)
    assert ev.preferred_text is None
    ev.add("y", (), "d2", preferred_text="Later Name")
    assert ev.preferred_text == "Later Name"


def test_top_ranks_by_count_descending_ties_by_cui():
    evidence = {
        "C2": CuiEvidence("C2", count=5),
        "C1": CuiEvidence("C1", count=5),
        "C3": CuiEvidence("C3", count=9),
    }
    ranked = top(evidence, n=10)
    assert [e.cui for e in ranked] == ["C3", "C1", "C2"]


def test_top_respects_n():
    evidence = {f"C{i}": CuiEvidence(f"C{i}", count=i) for i in range(10)}
    ranked = top(evidence, n=3)
    assert len(ranked) == 3
    assert [e.cui for e in ranked] == ["C9", "C8", "C7"]

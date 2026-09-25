"""Tests for ``ClinicalPipeline.assess``: assertion rules over caller-supplied spans.

Require both the built dictionary and the optional `nlp` extra; skipped
otherwise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"

pytestmark = [
    pytest.mark.skipif(not DB.is_file(), reason=f"dictionary not built: {DB}"),
    pytest.mark.skipif(
        importlib.util.find_spec("spacy") is None,
        reason="optional 'nlp' extra (spaCy) not installed",
    ),
]

NOTE = (
    "Patient denies chest pain and shortness of breath.\n"
    "Mother had breast cancer.\n"
    "History of type 2 diabetes mellitus.\n"
    "Started on metformin 500 mg PO BID.\n"
)


@pytest.fixture(scope="module")
def nlp():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB) as p:
        yield p


def _span(phrase: str, cui: str) -> tuple[int, int, str]:
    start = NOTE.index(phrase)
    return start, start + len(phrase), cui


def test_agrees_with_analyze_on_its_own_matches(nlp):
    """The same span gets the same attributes from either entry point."""
    found = nlp.analyze(NOTE)
    assessed = nlp.assess(NOTE, [(a.start, a.end, a.cui) for a in found])
    assert len(assessed) == len(found)
    for a, b in zip(found, assessed, strict=True):
        assert b is not None
        assert (b.start, b.end, b.cui, b.group) == (a.start, a.end, a.cui, a.group)
        assert (b.negated, b.subject, b.history_of, b.uncertain, b.conditional) == (
            a.negated,
            a.subject,
            a.history_of,
            a.uncertain,
            a.conditional,
        )


def test_external_spans_get_the_rules(nlp):
    chest, cancer, metformin = nlp.assess(
        NOTE,
        [
            _span("chest pain", "C0008031"),
            _span("breast cancer", "C0006142"),
            _span("metformin", "C0025598"),
        ],
    )
    assert chest.negated is True, "chest pain follows 'denies'"
    assert cancer.subject == "family_member", "breast cancer is the mother's"
    assert metformin.negated is False
    assert metformin.term == "", "no dictionary string matched an external span"
    assert chest.preferred_text, "preferred text comes from this dictionary"


def test_results_keep_input_order_and_offsets(nlp):
    spans = [_span("metformin", "C0025598"), _span("chest pain", "C0008031")]
    out = nlp.assess(NOTE, spans)
    assert [(a.start, a.end) for a in out] == [(s, e) for s, e, _ in spans]
    for a in out:
        assert NOTE[a.start : a.end] == a.text


def test_span_crossing_a_sentence_is_not_assessed(nlp):
    start = NOTE.index("breath")
    end = NOTE.index("Mother") + len("Mother")
    (result,) = nlp.assess(NOTE, [(start, end, "C0000000")])
    assert result is None, "a guess would read as an assessment"


def test_whitespace_only_span_is_not_assessed(nlp):
    gap = NOTE.index("\n")
    (result,) = nlp.assess(NOTE, [(gap, gap + 1, "C0000000")])
    assert result is None


def test_mid_token_span_keeps_callers_offsets(nlp):
    start = NOTE.index("chest pain") + 1  # "hest pain"
    (result,) = nlp.assess(NOTE, [(start, start + len("hest pain"), "C0008031")])
    assert result is not None and result.negated is True
    assert result.start == start, "widened for scope, not in the output"


def test_unknown_cui_is_assessed_with_empty_labels(nlp):
    (result,) = nlp.assess(NOTE, [_span("chest pain", "C9999999")])
    assert result is not None and result.negated is True
    assert result.preferred_text == "" and result.group == ""


def test_invalid_spans_rejected(nlp):
    with pytest.raises(ValueError):
        nlp.assess(NOTE, [(5, 5, "C0008031")])
    with pytest.raises(ValueError):
        nlp.assess(NOTE, [(0, len(NOTE) + 1, "C0008031")])


def test_empty_input(nlp):
    assert nlp.assess(NOTE, []) == []
    assert nlp.assess("   ", [(0, 1, "C0008031")]) == [None]

"""Tests for the negspaCy second-opinion diff.

The overlap-filtering logic is pure and always runs -- it is the part that
decides *which* mentions get compared, so a bug there silently changes what
the diff means. The end-to-end test needs the optional `compare` extra.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from umlsmatch.analyze import Annotation
from umlsmatch.eval.negspacy_diff import DiffReport, _non_overlapping

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"


def _ann(start: int, end: int, cui: str = "C1", negated: bool = False) -> Annotation:
    return Annotation(
        cui=cui, text="x" * (end - start), start=start, end=end,
        group="FINDING", negated=negated,
    )


def test_non_overlapping_prefers_the_longest():
    """"chest pain" must win over "chest" and "pain".

    negspaCy can only be asked about non-overlapping entities, so this choice
    decides what the comparison is *about*. Keeping the short ones instead
    would compare the pipeline's nested noise rather than its concepts.
    """
    kept = _non_overlapping([_ann(0, 5), _ann(0, 10), _ann(6, 10)])
    assert [(a.start, a.end) for a in kept] == [(0, 10)]


def test_non_overlapping_keeps_disjoint_mentions():
    kept = _non_overlapping([_ann(0, 5), _ann(10, 20), _ann(30, 33)])
    assert len(kept) == 3


def test_non_overlapping_returns_document_order():
    """Order is load-bearing: results are zipped against doc.ents, which spaCy
    returns in document order. A mismatch would silently compare each mention
    against a different one's polarity."""
    kept = _non_overlapping([_ann(30, 33), _ann(0, 5), _ann(10, 20)])
    assert [a.start for a in kept] == [0, 10, 30]


def test_non_overlapping_handles_an_empty_document():
    assert _non_overlapping([]) == []


def test_report_partitions_disagreements_by_direction():
    from umlsmatch.eval.negspacy_diff import Disagreement

    report = DiffReport(compared=2)
    report.disagreements = [
        Disagreement("d", "C1", 0, 1, "a", ours=True, theirs=False),
        Disagreement("d", "C2", 2, 3, "b", ours=False, theirs=True),
    ]
    assert len(report.we_negate_they_do_not) == 1
    assert len(report.they_negate_we_do_not) == 1


def test_summary_reports_zero_agreement_without_dividing_by_zero():
    """An empty run must summarize, not raise -- this is a diagnostic tool and
    pointing it at an empty folder should say so plainly."""
    assert "0 documents" in DiffReport().summary()


@pytest.mark.skipif(
    importlib.util.find_spec("negspacy") is None or not DB.is_file(),
    reason="needs the optional 'compare' extra and the built dictionary",
)
def test_end_to_end_agrees_on_an_unambiguous_case():
    from umlsmatch.eval.negspacy_diff import compare_documents

    report = compare_documents([("t", "Patient denies chest pain.")], db_path=DB)
    assert report.documents == 1
    assert report.compared >= 1
    assert report.agree_negated >= 1, report.summary()

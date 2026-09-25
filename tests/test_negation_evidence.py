"""Tests for negation evidence -- the "why" behind each negated match.

The error analysis in ``eval.negation_diff`` attributes wrong flags to the
trigger that caused them, so evidence must agree with the polarity decision
exactly. If the two ever diverge, the attribution is blaming phrases that did
not fire.
"""

from __future__ import annotations

from umlsmatch.assertion.negation import (
    negated_matches,
    negation_evidence,
    trigger_text,
)
from umlsmatch.dictionary.matcher import Match, tokenize


def _match(tokens, token_start, token_end, cui="C1"):
    return Match(
        cui=cui,
        term=" ".join(t.norm for t in tokens[token_start:token_end]),
        text="",
        start=tokens[token_start].start,
        end=tokens[token_end - 1].end,
        token_start=token_start,
        token_end=token_end,
    )


def _index_of(tokens, word):
    return next(i for i, t in enumerate(tokens) if t.norm == word)


def test_evidence_keys_equal_negated_matches():
    """The two entry points must never disagree about what is negated."""
    tokens = tokenize("Patient denies chest pain but reports fever today")
    matches = [
        _match(tokens, _index_of(tokens, "chest"), _index_of(tokens, "pain") + 1, "C1"),
        _match(tokens, _index_of(tokens, "fever"), _index_of(tokens, "fever") + 1, "C2"),
    ]
    assert set(negation_evidence(tokens, matches)) == negated_matches(tokens, matches)


def test_evidence_names_the_trigger_that_fired():
    tokens = tokenize("Patient denies chest pain")
    m = _match(tokens, _index_of(tokens, "chest"), _index_of(tokens, "pain") + 1)
    evidence = negation_evidence(tokens, [m])
    assert [trigger_text(tokens, t) for t in evidence[m]] == ["denies"]


def test_affirmed_match_has_no_entry():
    tokens = tokenize("Patient reports chest pain")
    m = _match(tokens, _index_of(tokens, "chest"), _index_of(tokens, "pain") + 1)
    assert negation_evidence(tokens, [m]) == {}


def test_no_triggers_yields_empty_evidence():
    tokens = tokenize("Patient has diabetes")
    m = _match(tokens, _index_of(tokens, "diabetes"), _index_of(tokens, "diabetes") + 1)
    assert negation_evidence(tokens, [m]) == {}
    assert negated_matches(tokens, [m]) == frozenset()


def test_overlapping_triggers_are_all_reported():
    """"no" and "no evidence of" both match here; evidence must show both.

    This is an inert-lexicon wart: the bare "no" has the wider scope and always
    dominates, so the longer phrase never changes an outcome. Evidence makes
    that visible
    instead of leaving it to be inferred.
    """
    tokens = tokenize("There is no evidence of pneumonia here")
    idx = _index_of(tokens, "pneumonia")
    m = _match(tokens, idx, idx + 1)
    fired = [trigger_text(tokens, t) for t in negation_evidence(tokens, [m])[m]]
    assert "no" in fired
    assert "no evidence of" in fired


def test_scope_cap_is_honoured_by_evidence():
    """max_scope=0 negates nothing, so nothing has evidence either."""
    tokens = tokenize("Patient denies chest pain")
    m = _match(tokens, _index_of(tokens, "chest"), _index_of(tokens, "pain") + 1)
    assert negation_evidence(tokens, [m], max_scope=0) == {}


def test_trigger_text_is_normalized():
    tokens = tokenize("Patient DENIES chest pain")
    m = _match(tokens, _index_of(tokens, "chest"), _index_of(tokens, "pain") + 1)
    (trigger,) = negation_evidence(tokens, [m])[m]
    assert trigger_text(tokens, trigger) == "denies"

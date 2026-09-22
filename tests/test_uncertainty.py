"""Tests for the hedging rules.

These carry more weight than usual, because the corpus cannot. 120 positives
cannot distinguish a working lexicon from a nearly-working one, so a unit test
that pins "possible pneumonia is hedged and confirmed pneumonia is not" is the
only check on this module that means anything. The corpus score checks only
that the rules fire at a sane volume -- see the module docstring.
"""

from __future__ import annotations

from umlsmatch.assertion.negex_triggers import NEGATION_PHRASE_TYPES
from umlsmatch.assertion.scope import normalize_phrase
from umlsmatch.assertion.uncertainty import (
    HEDGE_FORWARD_CUES,
    uncertain_matches,
    uncertainty_evidence,
)
from umlsmatch.dictionary.matcher import Match, tokenize


def _match(tokens, phrase: str, cui: str = "C0032285") -> Match:
    words = phrase.casefold().split()
    norms = [t.norm for t in tokens]
    for i in range(len(norms) - len(words) + 1):
        if norms[i : i + len(words)] == words:
            return Match(
                cui=cui,
                term=phrase.casefold(),
                text=phrase,
                start=tokens[i].start,
                end=tokens[i + len(words) - 1].end,
                token_start=i,
                token_end=i + len(words),
                group="DISORDER",
            )
    raise AssertionError(f"{phrase!r} not in {norms}")


def _is_hedged(text: str, phrase: str = "pneumonia") -> bool:
    tokens = tokenize(text)
    m = _match(tokens, phrase)
    return m in uncertain_matches(tokens, [m])


# --- forward hedges -----------------------------------------------------------


def test_common_hedges_fire():
    for hedge in (
        "possible", "probable", "likely", "suspected", "questionable",
        "presumed", "equivocal",
    ):
        assert _is_hedged(f"{hedge.capitalize()} pneumonia."), hedge


def test_multi_word_hedges_fire():
    for hedge in (
        "concerning for", "suspicion for", "cannot rule out", "may represent",
        "suggestive of", "differential includes",
    ):
        assert _is_hedged(f"Findings {hedge} pneumonia."), hedge


def test_an_unhedged_assertion_is_not_uncertain():
    assert not _is_hedged("Patient has pneumonia.")
    assert not _is_hedged("Chest x-ray shows pneumonia.")


def test_a_certainty_cue_terminates_a_hedge():
    assert not _is_hedged("Possible sepsis, but confirmed pneumonia on culture.")


# --- backward hedges ----------------------------------------------------------


def test_a_trailing_hedge_claims_what_precedes_it():
    # "can not", not "cannot": `_is_hedged` tokenizes with the matcher's
    # tokenizer, which keeps "cannot" whole, while the lexicon is stored in the
    # *pipeline* tokenizer's spelling, where spaCy has split it. The two
    # tokenizers genuinely disagree on this word; writing the sentence in
    # spaCy's spelling is what makes this test exercise the same lexicon entry
    # the pipeline will. The closed spelling is covered end to end in
    # tests/test_pipeline_cues.py, which is the test that would have caught
    # this family being unreachable.
    assert _is_hedged("Pneumonia can not be excluded.")
    assert _is_hedged("Pneumonia is possible.")


# --- the NegEx pseudo-negation class ------------------------------------------


def test_the_negex_rule_out_family_is_adopted():
    """NegEx's `pnega` class is the hedging lexicon; negation only suppresses
    with it. See the module docstring."""
    assert _is_hedged("Rule out pneumonia.")
    assert _is_hedged("Pneumonia to be ruled out for further workup.", "pneumonia")


def test_every_negex_pseudo_negation_phrase_is_a_hedge_cue():
    """The adoption is wholesale, so nothing in that class is silently dropped.

    Compared in the tokenizer's spelling: NegEx writes ``r/o`` and the lexicon
    stores ``r | / | o``, because that is what spaCy emits. See
    ``scope.normalize_phrase``.
    """
    pnega = {
        normalize_phrase(p)
        for p, kind in NEGATION_PHRASE_TYPES.items()
        if kind == "pnega"
    }
    missing = pnega - set(HEDGE_FORWARD_CUES)
    assert not missing, f"NegEx pseudo-negation phrases not adopted: {sorted(missing)}"


def test_rule_out_never_fired_as_negation_anyway():
    """The phrase this module claims was inert in negation, not taken from it.

    "rule out" sits in negation's forward list *and* in its pseudo list (via
    NegEx), and pseudo suppression drops any trigger overlapping a pseudo span
    -- so it has never negated anything. Claiming it here takes nothing away.
    """
    from umlsmatch.assertion.negation import negated_matches

    tokens = tokenize("Rule out pneumonia.")
    m = _match(tokens, "pneumonia")
    assert m not in negated_matches(tokens, [m])


# --- pseudo-hedges ------------------------------------------------------------


def test_reporting_idioms_are_not_hedges():
    assert not _is_hedged("Findings consistent with prior pneumonia.")
    assert not _is_hedged("Discharge as soon as possible after pneumonia resolves.")


# --- evidence -----------------------------------------------------------------


def test_evidence_keys_match_the_decision():
    tokens = tokenize("Possible pneumonia and confirmed sepsis.")
    matches = [_match(tokens, "pneumonia"), _match(tokens, "sepsis", "C0243026")]
    assert set(uncertainty_evidence(tokens, matches)) == uncertain_matches(
        tokens, matches
    )


def test_evidence_names_the_hedge_that_fired():
    tokens = tokenize("Possible pneumonia.")
    m = _match(tokens, "pneumonia")
    (cue,) = uncertainty_evidence(tokens, [m])[m]
    assert [t.norm for t in tokens[cue.start : cue.end]] == ["possible"]


def test_no_hedge_means_no_uncertainty():
    tokens = tokenize("Pneumonia treated with levofloxacin.")
    assert uncertain_matches(tokens, [_match(tokens, "pneumonia")]) == frozenset()

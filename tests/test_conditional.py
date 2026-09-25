"""Tests for the conditional rules.

These carry more weight than the ones for any shipped attribute, because for
this attribute they are the *only* evidence. ``uncertain`` at least has a
corpus number that can show its lexicon is dead or runaway; 27 ``conditional``
positives cannot do even that. Until an adjudication runs, "if you develop
chest pain is conditional and the patient has chest pain is not" is pinned
here or it is pinned nowhere.

The pseudo-cue cases are the point of the file. ``if`` as a complementizer --
"asked if", "unclear if" -- is the failure mode that would make this rule set
unusable, and it is ordinary history-taking prose rather than an edge case.
"""

from __future__ import annotations

from umlsmatch.assertion.conditional import (
    CLAUSE_CUES,
    CLAUSE_SCOPE_TOKENS,
    conditional_evidence,
    conditional_matches,
)
from umlsmatch.dictionary.matcher import Match, tokenize


def _match(tokens, phrase: str, cui: str = "C0008031") -> Match:
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


def _is_conditional(text: str, phrase: str = "chest pain") -> bool:
    tokens = tokenize(text)
    m = _match(tokens, phrase)
    return m in conditional_matches(tokens, [m])


# --- the cue families ---------------------------------------------------------


def test_if_is_the_core_cue():
    assert _is_conditional("Call the office if you develop chest pain.")
    assert _is_conditional("If chest pain recurs, go to the ED.")
    assert _is_conditional("If the patient develops chest pain, notify the team.")


def test_the_non_if_hypothetical_cues_fire():
    assert _is_conditional("Return to the ED in case of chest pain.")
    assert _is_conditional("In the event of chest pain, call 911.")
    assert _is_conditional("Nitroglycerin prn chest pain.")
    assert _is_conditional("Take nitroglycerin as needed for chest pain.")
    assert _is_conditional("Do not exercise unless chest pain has resolved.")


def test_pronoun_forms_of_should_fire():
    assert _is_conditional("Should you develop chest pain, call the office.")
    assert _is_conditional("Should the patient report chest pain, page cardiology.")


def test_bare_should_is_not_a_cue():
    """"The patient should take aspirin" is a recommendation, not a condition.

    ConText lists the pronoun forms rather than bare ``should`` for exactly
    this reason, and the deontic use is far commoner in these notes.
    """
    assert not _is_conditional("The patient should be evaluated for chest pain.")


def test_an_unconditional_assertion_is_not_conditional():
    assert not _is_conditional("Patient has chest pain.")
    assert not _is_conditional("Chest pain resolved after nitroglycerin.")


# --- the complementizer, which is what this rule set is built around ----------


def test_if_meaning_whether_is_not_a_condition():
    for stem in (
        "We asked if he had",
        "It is unclear if he has",
        "Unable to tell if there is",
        "Will check if there is",
        "Follow up to see if there is",
        "Unknown if there is",
        "Not sure if there is",
    ):
        assert not _is_conditional(f"{stem} chest pain."), stem


def test_concessive_and_comparative_if_are_not_conditions():
    assert not _is_conditional("Even if chest pain returns, continue the statin.")
    assert not _is_conditional("Describes it as if chest pain were pressure.")


def test_a_pseudo_cue_only_suppresses_its_own_if():
    """Suppression is per-cue, not per-sentence.

    A sentence can carry both uses of ``if``, and dropping every cue because
    one was a complementizer would make the whole lexicon hostage to the
    commonest word in it.
    """
    text = "We asked if he smoked; if chest pain develops, call the office."
    assert _is_conditional(text)


# --- scope --------------------------------------------------------------------


def test_a_conditionally_indicated_treatment_is_conditional():
    """The apodosis is in scope, and for these sentences that is correct.

    "If you develop a fever, take Tylenol" prescribes the Tylenol
    conditionally. The module declines to terminate at the imperative for this
    reason among others; see its docstring.
    """
    assert _is_conditional("If you develop chest pain, take aspirin.", "aspirin")


def test_a_context_terminator_ends_the_clause():
    assert not _is_conditional("If asked, but chest pain is present, treat it.")


def test_the_scope_cap_bounds_a_long_clause():
    far = "If " + "really " * CLAUSE_SCOPE_TOKENS + "bad chest pain occurs, call."
    assert not _is_conditional(far)


# --- the indication family ----------------------------------------------------


def test_a_prn_indication_is_conditional():
    assert _is_conditional("Oxycodone 5 mg PO q4h PRN chest pain.")
    assert _is_conditional("Nitroglycerin 0.4 mg SL as needed for chest pain.")


def test_a_trailing_prn_governs_nothing():
    """The structural defect the first corpus run exposed.

    A medication row ending in PRN has no indication, and the next row is a
    different drug. Before ``_indication_scopes`` stopped at the first non-word
    token, the window walked straight into it -- "bisacodyl 10 mg Rectal Daily
    PRN . dextrose in water" made the dextrose conditional.
    """
    row = "Bisacodyl 10 mg Rectal Daily PRN . Dextrose in water 50% Intravenous."
    assert not _is_conditional(row, "dextrose")
    assert not _is_conditional(row, "water")


def test_an_indication_does_not_run_past_its_own_sentence():
    """"...as needed for pain. 40 tablet" -- the count is a different field."""
    text = "Ibuprofen 800 mg tablet twice daily as needed for pain. 40 tablet aspirin."
    assert _is_conditional(text, "pain")
    assert not _is_conditional(text, "aspirin")


def test_the_indication_rule_does_not_bound_a_clause():
    """Only the PRN family stops at punctuation.

    A conditional clause legitimately contains commas -- "if you develop
    nausea, or chest pain" is one protasis, and a comma rule would cut it after
    the first item.

    The token cap still does cut a long enough list, and nothing here fixes
    that: coordination propagation is a parse-driven rule that lives in
    ``negation`` and was measured on polarity. See ``scope``'s docstring on why
    it did not move.
    """
    assert _is_conditional("If you develop nausea, or chest pain, call the office.")


def test_a_cue_does_not_reach_backwards():
    """Every cue here is a left marker; nothing before one is conditional.

    The module declines backward cues deliberately, so this pins the absence
    rather than leaving it as an accident of the lexicon.
    """
    assert not _is_conditional("Chest pain improved if resting helps.")


# --- evidence -----------------------------------------------------------------


def test_evidence_keys_match_the_decision():
    tokens = tokenize("If chest pain develops, treat the pneumonia.")
    matches = [_match(tokens, "chest pain"), _match(tokens, "pneumonia", "C0032285")]
    assert set(conditional_evidence(tokens, matches)) == conditional_matches(
        tokens, matches
    )


def test_evidence_names_the_cue_that_fired():
    tokens = tokenize("If chest pain develops, call us.")
    m = _match(tokens, "chest pain")
    (cue,) = conditional_evidence(tokens, [m])[m]
    assert [t.norm for t in tokens[cue.start : cue.end]] == ["if"]


def test_no_cue_means_no_conditional():
    tokens = tokenize("Chest pain treated with nitroglycerin.")
    assert conditional_matches(tokens, [_match(tokens, "chest pain")]) == frozenset()


# --- the lexicon itself -------------------------------------------------------


def test_longer_if_phrases_are_not_listed_redundantly():
    """``if`` subsumes ``if you``/``if he``/``if the patient``.

    Phrase matching is per-offset, so a one-token cue fires inside every longer
    phrase containing it. A redundant entry is not wrong, it is invisible -- it
    can never be the span reported as the cause, which makes the evidence
    output lie about which rule fired.
    """
    redundant = [p for p in CLAUSE_CUES if len(p) > 1 and p[0] == "if"]
    assert not redundant, f"subsumed by the bare 'if' cue: {redundant}"

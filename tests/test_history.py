"""Tests for the history-taking-context rules.

Hand-built tokens throughout -- no dictionary, no spaCy. What these pin is
mostly what the module does *not* do: two of the three rules that carry
``subject`` were measured to hurt here and are off, and the value of pinning
that is that the next person to switch one on has to explain the measurement
rather than discover it.
"""

from __future__ import annotations

import pytest

from umlsmatch.assertion.history import (
    CANDIDATE_HISTORY_SECTIONS,
    HISTORY_SECTIONS,
    history_evidence,
    history_matches,
)
from umlsmatch.assertion.negation import NEGEX_EXCLUSIONS
from umlsmatch.dictionary.matcher import Match, tokenize


def _match(tokens, phrase: str, cui: str = "C0011849") -> Match:
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


def _is_history(text: str, phrase: str, *, cui: str = "C0011849", **kwargs) -> bool:
    tokens = tokenize(text)
    m = _match(tokens, phrase, cui)
    return m in history_matches(tokens, [m], **kwargs)


# --- the cue rule, which is the whole attribute -------------------------------


def test_history_of_claims_what_follows():
    assert _is_history("Patient with history of diabetes mellitus.", "diabetes")


def test_abbreviated_spellings_work():
    for cue in ("h/o", "hx of"):
        assert _is_history(f"Patient with {cue} diabetes mellitus.", "diabetes")


def test_status_post_claims_a_procedure():
    assert _is_history("Status post diabetes mellitus screening.", "diabetes")


def test_a_present_tense_sentence_is_not_history():
    assert not _is_history("Patient has diabetes mellitus.", "diabetes")


def test_a_trailing_temporal_cue_claims_what_precedes_it():
    assert _is_history("Diabetes mellitus, resolved.", "diabetes")
    assert _is_history("Diabetes mellitus three years ago.", "diabetes")


def test_a_present_cue_terminates_a_history_scope():
    """"History of COPD, now presents with diabetes" -- the diabetes is current."""
    assert not _is_history(
        "History of COPD but now diabetes mellitus is new.", "diabetes"
    )


def test_the_forward_reach_is_short():
    far = "History of chronic obstructive pulmonary disease and also the diabetes mellitus."
    assert not _is_history(far, "diabetes")


# --- the NegEx cross-reference ------------------------------------------------


def test_negex_temporality_exclusions_are_adopted_here():
    """The other half of the decision in negation.NEGEX_EXCLUSIONS.

    Those phrases were refused as *negation* because they encode temporality.
    If they were not picked up as temporality, refusing them would have thrown
    the signal away rather than routing it.
    """
    for phrase in NEGEX_EXCLUSIONS:
        text = f"Diabetes mellitus {' '.join(phrase)}."
        assert _is_history(text, "diabetes"), f"{phrase} is not a history cue"


def test_resolved_is_history_not_negation():
    from umlsmatch.assertion.negation import negated_matches

    tokens = tokenize("Bowel obstruction resolved.")
    m = _match(tokens, "bowel obstruction", "C0021843")
    assert m not in negated_matches(tokens, [m]), "'resolved' is not polarity"
    assert m in history_matches(tokens, [m]), "'resolved' is temporality"


# --- the two rules that were measured and dropped -----------------------------


def test_the_section_rule_is_off_and_that_is_a_measurement():
    """Precision 0.116; enabling it took corpus F1 from 0.564 to 0.430."""
    assert not HISTORY_SECTIONS
    assert not _is_history(
        "Diabetes mellitus, hypertension.", "diabetes", section="past_medical_history"
    )


def test_self_reference_is_off_by_default():
    """cTAKES marks 54 of 450 bare "history" mentions; enabling this took
    corpus F1 from 0.564 to 0.477."""
    text = "Past medical history reviewed."
    assert not _is_history(text, "past medical history", cui="C0262926")
    assert _is_history(
        text, "past medical history", cui="C0262926", cue_self_reference=True
    )


def test_history_of_present_illness_is_not_suppressed():
    """The opposite of the obvious call, and it was measured: suppressing it
    cost F1 0.508 -> 0.473 (before the tokenization fix). A history taken in
    the HPI is still history."""
    assert _is_history(
        "History of present illness: history of diabetes mellitus.", "diabetes"
    )


# --- evidence -----------------------------------------------------------------


#: Sentences the invariant is checked over, as ``(text, [(phrase, cui), ...])``.
#:
#: Two, because the section rule's guard is what distinguishes them. The first
#: carries present-tense cues ("current", "today"), so
#: ``_section_claims_everything`` is false and the section branch cannot fire
#: even with the rule on. The second carries no cue in either direction, so with
#: the rule on the section is the *only* thing that can claim anything -- which
#: is the case where a divergence between the two functions is visible at all.
_INVARIANT_SENTENCES = [
    (
        "History of diabetes mellitus and current asthma today.",
        [("diabetes", "C0011849"), ("asthma", "C0004096")],
    ),
    (
        "Insomnia chronic, asthma stable.",
        [("insomnia", "C0917801"), ("asthma", "C0004096")],
    ),
]

#: Rule configurations the invariant must survive.
#:
#: The shipped default leaves :data:`HISTORY_SECTIONS` empty, so the section
#: branch in both functions is dead and a matrix of default-only cases passes
#: whether or not ``history_evidence`` honours the override at all. It did not,
#: for a while: it read the module constant directly while ``history_matches``
#: resolved the parameter, and the docstring promise that the evidence keys are
#: exactly the decision held only by accident of the default. The rows carrying
#: ``history_sections`` are the ones that pin it.
_INVARIANT_OPTIONS = [
    {},
    {"cue_self_reference": True},
    {"section": "past_medical_history"},
    {"section": "past_medical_history", "history_sections": CANDIDATE_HISTORY_SECTIONS},
    {"section": "review_of_systems", "history_sections": CANDIDATE_HISTORY_SECTIONS},
    {
        "section": "past_medical_history",
        "history_sections": CANDIDATE_HISTORY_SECTIONS,
        "cue_self_reference": True,
    },
]


@pytest.mark.parametrize("text,phrases", _INVARIANT_SENTENCES)
@pytest.mark.parametrize("options", _INVARIANT_OPTIONS)
def test_evidence_keys_match_the_decision(text, phrases, options):
    """``history_evidence`` keys are exactly what ``history_matches`` returns.

    This is the module's stated contract and what
    :mod:`umlsmatch.eval.explain` relies on to attribute an error to a rule.
    """
    tokens = tokenize(text)
    matches = [_match(tokens, phrase, cui) for phrase, cui in phrases]
    assert set(history_evidence(tokens, matches, **options)) == history_matches(
        tokens, matches, **options
    )


def test_evidence_attributes_a_section_claim_to_no_cue():
    """A section-claimed match maps to an empty tuple, not to a borrowed cue.

    ``eval.explain`` reads that empty tuple as "the section did this" and labels
    the rule ``SECTION``. If the section rule ever put a cue here, every mention
    in a PMH list would be attributed to whichever cue happened to be nearby.
    """
    tokens = tokenize("Insomnia chronic, asthma stable.")
    matches = [_match(tokens, "insomnia", "C0917801")]
    evidence = history_evidence(
        tokens, matches,
        section="past_medical_history",
        history_sections=CANDIDATE_HISTORY_SECTIONS,
    )
    assert evidence == {matches[0]: ()}


def test_evidence_ignores_the_override_for_an_unclaimed_section():
    """The override is not a blanket switch: the section still has to be in it."""
    tokens = tokenize("Insomnia chronic, asthma stable.")
    matches = [_match(tokens, "insomnia", "C0917801")]
    assert not history_evidence(
        tokens, matches,
        section="review_of_systems",
        history_sections=CANDIDATE_HISTORY_SECTIONS,
    )


def test_evidence_names_the_cue_that_fired():
    tokens = tokenize("Patient with history of diabetes mellitus.")
    m = _match(tokens, "diabetes")
    (cue,) = history_evidence(tokens, [m])[m]
    assert [t.norm for t in tokens[cue.start : cue.end]] == ["history", "of"]


def test_no_cue_means_no_history():
    tokens = tokenize("Diabetes mellitus is well controlled.")
    assert history_matches(tokens, [_match(tokens, "diabetes")]) == frozenset()


# --- the opt-in section rule ---------------------------------------------------


def test_history_sections_defaults_to_off():
    """The shipped default claims nothing on a section header alone."""
    from umlsmatch.assertion.history import HISTORY_SECTIONS, history_matches
    from umlsmatch.dictionary.matcher import tokenize

    assert not HISTORY_SECTIONS
    tokens = tokenize("Insomnia chronic")
    m = _match(tokens, "insomnia")
    assert m not in history_matches(tokens, [m], section="past_medical_history")


def test_history_sections_can_be_turned_on():
    """Passing the candidate set claims the section body.

    Off by default because the measurement that justifies it is against model
    verdicts, not a clinician's -- see the HISTORY_SECTIONS docstring.
    """
    from umlsmatch.assertion.history import (
        CANDIDATE_HISTORY_SECTIONS,
        history_matches,
    )
    from umlsmatch.dictionary.matcher import tokenize

    tokens = tokenize("Insomnia chronic")
    m = _match(tokens, "insomnia")
    claimed = history_matches(
        tokens, [m],
        section="past_medical_history",
        history_sections=CANDIDATE_HISTORY_SECTIONS,
    )
    assert m in claimed


def test_history_sections_still_needs_the_section_to_match():
    """A section not in the set is untouched even with the rule on."""
    from umlsmatch.assertion.history import (
        CANDIDATE_HISTORY_SECTIONS,
        history_matches,
    )
    from umlsmatch.dictionary.matcher import tokenize

    tokens = tokenize("Insomnia chronic")
    m = _match(tokens, "insomnia")
    assert m not in history_matches(
        tokens, [m], section="review_of_systems",
        history_sections=CANDIDATE_HISTORY_SECTIONS,
    )

"""Tests for the patient-vs-family-member rules.

Hand-built tokens throughout, so these need neither a dictionary nor spaCy.
What they pin is the behaviour the corpus measurement cannot see: that the two
directions have different reaches, that a spouse is not a relative, that the
section rule is not a blanket, and that the self-reference rule is a switch.
"""

from __future__ import annotations

from umlsmatch.assertion.subject import (
    FAMILY_CUES,
    KINSHIP_BACKWARD_SCOPE_TOKENS,
    KINSHIP_CUES,
    KINSHIP_FORWARD_SCOPE_TOKENS,
    family_member_matches,
    subject_evidence,
)
from umlsmatch.dictionary.matcher import Match, tokenize


def _match(tokens, phrase: str, cui: str = "C0011849") -> Match:
    """Build a Match over the first occurrence of `phrase` in `tokens`."""
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


def _is_family(text: str, phrase: str, *, cui: str = "C0011849", **kwargs) -> bool:
    tokens = tokenize(text)
    m = _match(tokens, phrase, cui)
    return m in family_member_matches(tokens, [m], **kwargs)


# --- the cue rule, forward ----------------------------------------------------


def test_kinship_term_claims_what_follows_it():
    assert _is_family("Mother with diabetes mellitus.", "diabetes")


def test_family_history_header_claims_a_list():
    assert _is_family("Family history of diabetes mellitus.", "diabetes")


def test_a_plain_sentence_is_the_patients():
    assert not _is_family("Patient has diabetes mellitus.", "diabetes")


# --- the cue rule, backward ---------------------------------------------------


def test_a_trailing_kinship_term_claims_what_precedes_it():
    """"MI in her father" -- the relation comes last as often as first."""
    assert _is_family("Myocardial infarction in her father.", "myocardial infarction")


def test_the_ehr_table_layout_is_carried_by_the_backward_rule():
    """Epic-style family history tables run `Problem Relation Age`.

    A forward window from "Brother" reaches into the *next* row, which is what
    the wide forward cap got wrong (precision 0.41 on this corpus). The
    backward rule is what makes this layout work.
    """
    text = "Arthritis Brother RA Cancer Sister"
    assert _is_family(text, "arthritis")


def test_family_history_does_not_scope_backward():
    """Measured at 7 right against 33 wrong; the text before it is the previous
    section, not the section's contents."""
    assert not _is_family(
        "Past medical history includes diabetes mellitus. Family history:", "diabetes"
    )


def test_the_backward_reach_is_tighter_than_a_sentence():
    """A wide backward window walks into the previous clause, which is the
    patient's."""
    far = "Diabetes mellitus was diagnosed in the clinic last year by the team and father."
    assert not _is_family(far, "diabetes")


# --- what is not a relative ---------------------------------------------------


def test_a_spouse_is_not_a_family_member():
    """0 right against 11 wrong on this corpus. cTAKES does not count them, and
    a spouse in these notes is usually the historian, not the subject."""
    assert not _is_family("Wife found patient with hypoglycemia.", "hypoglycemia")
    for phrase in ("wife", "husband", "spouse"):
        assert (phrase,) not in KINSHIP_CUES
        assert (phrase,) not in FAMILY_CUES


def test_pseudo_cues_suppress_a_family_reading():
    assert not _is_family("Family planning counseling for diabetes mellitus.", "diabetes")
    assert not _is_family("Seen in family practice for diabetes mellitus.", "diabetes")


def test_relative_as_a_comparative_is_not_a_person():
    assert not _is_family("Relative to baseline, diabetes mellitus is stable.", "diabetes")


# --- the section rule ---------------------------------------------------------


def test_a_family_history_section_claims_its_contents():
    assert _is_family("Diabetes mellitus, hypertension.", "diabetes", section="family_history")


def test_the_section_rule_is_not_a_blanket():
    """A patient cue in the sentence pulls it back -- the same restraint
    umlsmatch.assertion.sections makes about negation."""
    assert not _is_family(
        "Patient has never been screened for diabetes mellitus.",
        "diabetes",
        section="family_history",
    )


def test_a_kinship_cue_still_wins_inside_a_section_with_a_patient_cue():
    """Falling back to the cue rule must not lose the obvious cases."""
    assert _is_family(
        "The patient's mother had diabetes mellitus.",
        "diabetes",
        section="family_history",
    )


def test_another_section_does_not_claim_anything():
    assert not _is_family(
        "Diabetes mellitus, hypertension.", "diabetes", section="past_medical_history"
    )


def test_no_section_leaves_the_cue_rule_alone():
    """A caller that does not track sections gets fewer calls, not wrong ones."""
    assert not _is_family("Diabetes mellitus, hypertension.", "diabetes", section=None)


# --- the self-reference rule --------------------------------------------------


def test_the_cue_phrase_itself_is_a_family_member_concept():
    assert _is_family("No family history of note.", "family history", cui="C0241889")


def test_self_reference_can_be_switched_off():
    """It is half the reference positives; a score that hides that misleads."""
    tokens = tokenize("No family history of note.")
    m = _match(tokens, "family history", cui="C0241889")
    assert m in family_member_matches(tokens, [m], cue_self_reference=True)
    assert m not in family_member_matches(tokens, [m], cue_self_reference=False)


# --- evidence -----------------------------------------------------------------


def test_evidence_keys_match_the_decision():
    tokens = tokenize("Mother with diabetes mellitus and the patient has asthma.")
    matches = [_match(tokens, "diabetes"), _match(tokens, "asthma", "C0004096")]
    family = family_member_matches(tokens, matches)
    assert set(subject_evidence(tokens, matches)) == family


def test_evidence_names_the_cue_that_fired():
    tokens = tokenize("Mother with diabetes mellitus.")
    m = _match(tokens, "diabetes")
    (cue,) = subject_evidence(tokens, [m])[m]
    assert [t.norm for t in tokens[cue.start : cue.end]] == ["mother"]


def test_section_claimed_matches_have_no_cue_to_name():
    """An empty tuple is the honest answer, not a missing key."""
    tokens = tokenize("Diabetes mellitus, hypertension.")
    m = _match(tokens, "diabetes")
    assert subject_evidence(tokens, [m], section="family_history")[m] == ()


# --- the caps -----------------------------------------------------------------


def test_the_two_kinship_caps_are_documented_constants():
    """Both were measured; neither should be edited without re-measuring."""
    assert KINSHIP_FORWARD_SCOPE_TOKENS > 0
    assert KINSHIP_BACKWARD_SCOPE_TOKENS > 0


def test_no_cue_means_no_family_member():
    tokens = tokenize("Diabetes mellitus is well controlled.")
    assert family_member_matches(tokens, [_match(tokens, "diabetes")]) == frozenset()

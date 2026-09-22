"""Tests for negative-findings section detection.

Header recognition itself is parse-free and uses ``matcher.tokenize``, so those
tests always run. The end-to-end ones need the dictionary and spaCy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from umlsmatch.assertion.sections import (
    NEGATIVE_FINDINGS_SECTIONS,
    OTHER_SECTIONS,
    is_header_only,
    negative_findings_section,
    section_header,
)
from umlsmatch.dictionary.matcher import tokenize

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"


@pytest.mark.parametrize(
    "text",
    [
        "Review of Systems:",
        "review of systems:",
        "ROS:",
        "Systems Review:",
        "Allergies:",
        "Drug Allergies:",
    ],
)
def test_recognizes_header_spellings(text):
    assert negative_findings_section(tokenize(text)) in NEGATIVE_FINDINGS_SECTIONS


def test_spellings_collapse_to_one_canonical_name():
    """Several headers, one section name -- callers should not have to know them all."""
    names = {
        negative_findings_section(tokenize(t))
        for t in ("Review of Systems:", "ROS:", "Systems Review:")
    }
    assert names == {"review_of_systems"}


def test_header_must_be_followed_by_a_colon():
    """Prose that merely mentions the words is not a section header.

    Without the colon test this fires inside ordinary narrative and uncaps
    every trigger for the rest of the document.
    """
    assert negative_findings_section(tokenize("Allergies to penicillin were discussed.")) is None
    assert negative_findings_section(tokenize("We did a review of systems today.")) is None


def test_header_must_be_at_the_start():
    assert negative_findings_section(tokenize("Documented in ROS: nothing")) is None


def test_unknown_section_is_not_a_negative_findings_section():
    """Physical Exam is full of negatives and deliberately excluded -- it is also
    full of affirmed findings, and its sentences are short enough that the
    window already handles them."""
    assert negative_findings_section(tokenize("Physical Exam:")) is None
    assert negative_findings_section(tokenize("Past Medical History:")) is None


def test_empty_and_bare_token_lists():
    assert negative_findings_section([]) is None
    assert negative_findings_section(tokenize("ROS")) is None


def test_is_header_only_distinguishes_inline_content():
    assert is_header_only(tokenize("Review of Systems:")) is True
    assert is_header_only(tokenize("Review of Systems: Negative for fever")) is False
    assert is_header_only([]) is False


# --- Headers that close a section --------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "IMPRESSION Patient is alert",       # no colon at all
        "Impression Patient is alert",
        "# Impression Patient is alert",     # Markdown
        "## Plan Discharge tomorrow",
        "physical exam: unremarkable",       # colon, lowercase content
        "HOSPITAL COURSE The patient was admitted",
        "Past Medical History: hypertension",
        "Family History: mother with diabetes",
    ],
)
def test_other_headers_are_recognized_without_a_colon(text):
    """A header need not end in ':' to be recognized, and none of these do.

    Closing only on a colon would leave ``in_negative_section`` stuck on for
    the rest of the note.
    """
    assert section_header(tokenize(text)) in OTHER_SECTIONS
    # ... but they must never *open* a negative-findings section.
    assert negative_findings_section(tokenize(text)) is None


@pytest.mark.parametrize(
    "text",
    [
        "Plan to discharge tomorrow.",
        "Allergies to penicillin were discussed.",
        "Findings were reviewed with the patient.",
        "Impression of the wound was favorable.",
    ],
)
def test_a_header_word_in_prose_is_not_a_header(text):
    """Lowercase content after the word is what separates prose from a heading.

    Without this guard the general header list closes sections on ordinary
    narrative, which re-caps scope and costs the long-list negations the
    section rule exists to catch.
    """
    assert section_header(tokenize(text)) is None


# --- Enumerated and marked-up headers ----------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1. Review of Systems: Negative for fever",
        "1) Review of Systems: Negative for fever",
        "(1) Review of Systems: Negative for fever",
        "A. Allergies: none",
        "- Allergies: none",
        "* Review of Systems: Negative for fever",
        "# Review of Systems: Negative for fever",
    ],
)
def test_list_markers_do_not_hide_a_header(text):
    """The phrase lookup is anchored at index 0, so any marker would defeat it.

    Widening the token window does not help -- ``norms[:length]`` still starts
    at the marker. The marker has to be skipped.
    """
    assert negative_findings_section(tokenize(text)) in NEGATIVE_FINDINGS_SECTIONS


def test_marker_skipping_does_not_invent_headers():
    assert negative_findings_section(tokenize("1. Patient denies fever")) is None
    assert section_header(tokenize("2. The wound was clean")) is None


# --- End to end --------------------------------------------------------------

needs_pipeline = pytest.mark.skipif(
    not DB.is_file() or importlib.util.find_spec("spacy") is None,
    reason="needs the built dictionary and the optional 'nlp' extra",
)


def _negated(text: str, **kwargs) -> dict[str, bool]:
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, **kwargs) as nlp:
        return {a.text: a.negated for a in nlp.analyze(text)}


@needs_pipeline
def test_section_uncaps_scope_for_a_long_review_of_systems_list():
    """A list this long outruns any fixed window; the header is what survives."""
    text = (
        "Review of Systems: Negative for chills, fever, night sweats, "
        "weight loss, fatigue, headache, cough.\n"
    )
    assert _negated(text)["cough"] is True


@needs_pipeline
def test_section_still_requires_a_trigger():
    """The rule uncaps a scope; it never negates a section wholesale.

    This is the failure mode worth guarding: Review-of-Systems sections
    routinely carry positives, and blanket-negating the section would flip
    them. "Positive for cough" has no trigger, so nothing fires.
    """
    text = "Review of Systems: Positive for cough and wheezing.\n"
    negated = _negated(text)
    assert negated.get("cough") is False
    assert not any(negated.values()), negated


@needs_pipeline
def test_a_later_header_closes_the_section():
    """Section state must not leak into the rest of the note.

    Without a close, one Review of Systems uncaps every trigger to the end of
    the document -- the error would be invisible in a short test and large in
    a real note.
    """
    text = (
        "Review of Systems: Negative for fever.\n"
        "Physical Exam:\n"
        "No acute distress, patient is alert and oriented and conversing easily.\n"
    )
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB) as nlp:
        anns = nlp.analyze(text)
    late = [a for a in anns if a.start > text.index("Physical Exam")]
    assert late, "expected annotations after the section change"
    assert not all(a.negated for a in late), "section state leaked past its header"


@needs_pipeline
@pytest.mark.parametrize(
    "header",
    ["IMPRESSION", "# Impression", "HOSPITAL COURSE", "Assessment and Plan"],
)
def test_a_header_without_a_colon_closes_the_section(header):
    """A recognized header closes the section even with no ':' on it.

    None of these headers carry one. Were the close test "this window ends in
    ':'", the flag would stay on and every later trigger in the note would run
    with an uncapped scope. The finding below is reported, not denied, and the
    ';' does not stop an uncapped scope -- so with a leak in
    place it flipped to negated.
    """
    tail = (
        "No acute distress was noted on examination of the abdomen today; "
        "the chest showed a large pleural effusion.\n"
    )
    leaked = _negated(f"Review of Systems: Negative for chills.\n{header}\n{tail}")
    clean = _negated(f"{header}\n{tail}")

    assert clean["pleural effusion"] is False, "control: nothing should negate this"
    assert leaked["pleural effusion"] is False, "section state leaked past the header"
    # The trigger that *is* in the sentence still works, capped as usual.
    assert leaked["distress"] is True


@needs_pipeline
@pytest.mark.parametrize(
    "label",
    [
        "Course: Patient tolerated the procedure well.",  # capitalized: splits
        "Service: Cardiology",  # unrecognized-but-now-listed, colon-terminated
        "Wound Care:",  # unrecognized, header-only
    ],
)
def test_unrecognized_headers_still_close_the_section(label):
    """Three different mechanisms close a section, and only one is this
    module's table.

    A capitalized continuation is split by the tokenizer's label-colon rule, so
    the header arrives alone and `is_header_only` closes it. A colon-terminated
    window closes the same way whether or not the label is recognized. The
    table is the third path and the least load-bearing -- which is why
    extending it measured as a no-op.
    """
    tail = "The chest showed a large pleural effusion.\n"
    leaked = _negated(f"Review of Systems: Negative for chills.\n{label}\n{tail}")
    assert leaked["pleural effusion"] is False, "section state leaked past the label"


@needs_pipeline
def test_the_known_leak_shape_is_an_unrecognized_lowercase_continuation():
    """The residual, stated as a test rather than only as prose.

    An unrecognized label whose colon is followed by *lowercase* text is
    neither recognized by the table nor split by the tokenizer nor caught by
    `is_header_only`, so the section stays open. 49 of 686 unrecognized
    label-colon windows in the corpus have this shape. Adding the label to
    OTHER_SECTIONS is the fix when it matters; this test exists so the gap is
    recorded rather than rediscovered.
    """
    # The tail needs its own trigger and no terminator before the finding:
    # uncapping only widens a scope a trigger already opened, so a leaked
    # section is invisible until a later sentence has one.
    tail = (
        "No acute distress was noted on examination of the abdomen today; "
        "the chest showed a large pleural effusion.\n"
    )
    leaked = _negated(
        f"Review of Systems: Negative for chills.\nWound care: wound is clean.\n{tail}"
    )
    # Documenting current behaviour, not endorsing it. If a future change makes
    # this False, that is an improvement -- update the test and the module
    # docstring's measurement table together.
    assert leaked["pleural effusion"] is True, (
        "the leak did not reproduce; if it is genuinely closed now, update the "
        "measurement table in sections.py rather than only this assertion"
    )


@needs_pipeline
def test_an_enumerated_header_still_uncaps_scope():
    """A leading "1." must not hide the header."""
    text = (
        "1. Review of Systems: Negative for chills, fever, night sweats, "
        "weight loss, fatigue, headache, cough.\n"
    )
    assert _negated(text)["cough"] is True


# --- header spans, for the chart-furniture filter ------------------------------


def test_section_header_span_reports_where_the_header_ends():
    """The offset covers the heading and its colon, and nothing after."""
    from umlsmatch.assertion.sections import section_header_span
    from umlsmatch.dictionary.matcher import tokenize

    text = "Past Medical History: Insomnia chronic"
    found = section_header_span(tokenize(text))
    assert found is not None
    section, end = found
    assert section == "past_medical_history"
    assert text[:end] == "Past Medical History:"


def test_section_header_span_handles_an_attached_colon():
    """A colon spaCy leaves attached must not shift the offset.

    `_detached` splits it into its own element, making the norms list longer
    than the token list -- the reason the span is computed from owning tokens
    rather than by indexing the two in parallel.
    """
    from umlsmatch.assertion.sections import section_header_span
    from umlsmatch.dictionary.matcher import Token, tokenize

    attached = [Token(text="Allergies:", start=0, end=10, pos="NOUN"),
                *[t for t in tokenize("Penicillin")]]
    attached[1] = Token(text="Penicillin", start=11, end=21, pos="NOUN")
    found = section_header_span(attached)
    assert found is not None
    assert found[1] == 10


def test_section_header_span_is_none_for_ordinary_prose():
    from umlsmatch.assertion.sections import section_header_span
    from umlsmatch.dictionary.matcher import tokenize

    assert section_header_span(tokenize("allergies to penicillin were discussed")) is None

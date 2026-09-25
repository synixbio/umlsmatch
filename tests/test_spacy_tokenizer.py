"""Tests for the spaCy-backed tokenizer/POS bridge feeding the matcher.

Skips entirely if spaCy or its model isn't installed, so the base suite
doesn't require the optional `nlp` extra (`pip install -e .[nlp]`).
"""

from __future__ import annotations

import pytest

spacy_util = pytest.importorskip("spacy.util")

from umlsmatch.dictionary.matcher import DEFAULT_EXCLUSION_TAGS  # noqa: E402
from umlsmatch.pipeline.tokenizer import (  # noqa: E402
    DEFAULT_MODEL,
    annotate_sentences,
    load_model,
    new_model,
)

needs_model = pytest.mark.skipif(
    not spacy_util.is_package(DEFAULT_MODEL),
    reason=f"spaCy model not installed: {DEFAULT_MODEL} (pip install -e .[nlp])",
)


# --- sharing ----------------------------------------------------------------


@needs_model
def test_load_model_shares_one_instance():
    """The cache is the point for single-threaded callers, and is pinned here
    so that removing it cannot pass unnoticed -- scripts that build several
    pipelines in a loop rely on it."""
    assert load_model(DEFAULT_MODEL) is load_model(DEFAULT_MODEL)


@needs_model
def test_new_model_returns_an_unshared_instance():
    """A ``Language`` is single-thread-at-a-time; concurrent callers need their
    own. See ``tokenizer.new_model`` for the mechanism."""
    a = new_model(DEFAULT_MODEL)
    b = new_model(DEFAULT_MODEL)
    assert a is not b
    assert a is not load_model(DEFAULT_MODEL)
    assert a.vocab is not b.vocab  # the interning store is the shared state


@needs_model
def test_unshared_models_tokenize_identically():
    """Isolation must not change results, only who owns them."""
    text = "Patient denies chest x-ray findings. Past Medical History: COPD."
    shared = [
        [t.text for t in w] for w in annotate_sentences(text, model=DEFAULT_MODEL)
    ]
    own = [[t.text for t in w] for w in annotate_sentences(text, model=new_model())]
    assert shared == own


@needs_model
def test_splits_into_sentences():
    text = "History of type 2 diabetes mellitus and atrial fibrillation. Patient denies chest pain."
    sentences = annotate_sentences(text)
    assert len(sentences) == 2


@needs_model
def test_token_offsets_round_trip_to_source_text():
    text = "History of type 2 diabetes mellitus and atrial fibrillation. Patient denies chest pain."
    for sentence in annotate_sentences(text):
        for tok in sentence:
            assert text[tok.start : tok.end] == tok.text


@needs_model
def test_offsets_are_document_absolute_across_sentences():
    text = "First sentence here. Second one starts later."
    sentences = annotate_sentences(text)
    assert len(sentences) == 2
    second_first_token = sentences[1][0]
    assert second_first_token.start > 0
    assert text[second_first_token.start : second_first_token.end] == second_first_token.text


@needs_model
def test_conjunction_and_cardinal_are_excluded_as_anchors():
    # The finding this module exists to fix: without
    # POS tags, "and" matches a genuine but spurious UMLS concept (C1706368).
    text = "History of type 2 diabetes mellitus and atrial fibrillation."
    [tokens] = annotate_sentences(text)
    by_text = {t.text: t for t in tokens}

    assert by_text["and"].pos in DEFAULT_EXCLUSION_TAGS
    assert by_text["2"].is_word is False
    # A genuine anchor candidate should not be excluded.
    assert by_text["diabetes"].pos not in DEFAULT_EXCLUSION_TAGS
    assert by_text["diabetes"].is_word is True


@needs_model
def test_label_colon_splits_into_its_own_sentence():
    # The boundary-agreement finding: cTAKES treats inline
    # "Label:" pseudo-headers as sentence-final even with no other separator.
    # Synthetic narrative -- never paste real note text into a tracked file.
    text = "History of Present Illness: Spouse reported worsening cough at home."
    sentences = annotate_sentences(text)
    assert len(sentences) == 2
    assert sentences[0][-1].text == ":"
    assert sentences[1][0].text == "Spouse"


@needs_model
def test_colon_before_lowercase_does_not_split():
    text = "The plan is: continue monitoring closely for now."
    sentences = annotate_sentences(text)
    assert len(sentences) == 1


@needs_model
def test_colon_before_number_splits():
    text = "Vitals: 98.6 T, 72 HR, 120/80 BP."
    sentences = annotate_sentences(text)
    assert len(sentences) == 2
    assert sentences[1][0].text == "98.6"


# --- whitespace never reaches the lookup window -----------------------------
# cTAKES drops NewlineToken when assembling a window
# (AbstractJCasTermAnnotator.getAnnotationsInWindow); spaCy emits a token for
# any unattached whitespace run. Leaving those in broke two things at once:
# multi-word terms spanning a line wrap, and the label-colon rule below.


@needs_model
def test_whitespace_tokens_are_dropped():
    text = "Acute congestive\nheart failure.\n\nMeds:  aspirin."
    for sentence in annotate_sentences(text):
        assert all(tok.text.strip() for tok in sentence), [t.text for t in sentence]


@needs_model
def test_line_wrap_does_not_interrupt_a_multiword_term():
    """A hard-wrapped phrase must join to the same norm as the unwrapped one.

    The matcher verifies a candidate by joining the window's token norms with
    single spaces, so a retained "\\n" token makes any term crossing the wrap
    unmatchable.
    """
    wrapped = "He has congestive\nheart failure now."
    flat = "He has congestive heart failure now."

    def norms(text):
        return [" ".join(t.norm for t in sentence) for sentence in annotate_sentences(text)]

    assert norms(wrapped) == norms(flat)


@needs_model
def test_label_colon_splits_when_the_label_ends_the_line():
    """The commoner spelling of the pseudo-header: label, newline, body.

    With the whitespace token retained, the token after ":" was "\\n" -- not
    capitalized, not a digit -- so this did not split, while the
    space-separated form did.
    """
    sentences = annotate_sentences("History of Present Illness:\nSpouse reported a fall.")
    assert len(sentences) == 2
    assert sentences[0][-1].text == ":"
    assert sentences[1][0].text == "Spouse"


@needs_model
def test_whitespace_only_input_yields_no_windows():
    assert annotate_sentences("   \n\n  \t ") == []


def test_dependency_fields_are_populated():
    tokens = annotate_sentences("Patient denies chest pain and fever.")[0]
    assert any(t.dep for t in tokens), "no dependency labels reached the window"
    assert any(t.head >= 0 for t in tokens), "no head indices reached the window"


def test_head_indices_are_local_to_the_window():
    """`head` must index the window, not the document.

    Whitespace removal and label-colon splitting both renumber tokens, so a
    head carried over from spaCy's document numbering would point into the
    wrong sentence -- and would do so silently, because any index in range
    still looks valid to a consumer.
    """
    text = "Medications: Aspirin daily.\nPatient denies chest pain and fever.\n"
    for window in annotate_sentences(text):
        for tok in window:
            assert -1 <= tok.head < len(window), f"{tok.text} head={tok.head}"


def test_head_outside_the_window_becomes_minus_one():
    """Splitting separates a label from its dependents; that edge must be dropped."""
    windows = annotate_sentences("Medications: Aspirin, Lisinopril.\n")
    assert len(windows) >= 2, windows
    # The root of each window has no head inside it.
    assert any(t.head == -1 for t in windows[0])


def test_a_root_reports_head_minus_one():
    """spaCy makes a root its own head; this module spells that -1.

    Left as-is, a root would look like an ordinary edge pointing at itself and
    any graph walk over heads would loop.
    """
    tokens = annotate_sentences("Patient denies chest pain.")[0]
    assert any(t.head == -1 for t in tokens)
    assert all(t.head != i for i, t in enumerate(tokens)), "a token is its own head"

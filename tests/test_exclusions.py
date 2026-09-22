"""Regression tests pinning the hand-transcribed cTAKES exclusion lists.

See umlsmatch.dictionary.exclusions for provenance and the investigation
that produced these -- verified against a clone of apache/ctakes @ main.
"""

from __future__ import annotations

from umlsmatch.dictionary.exclusions import (
    BAD_POS_TERM_SET,
    EXCLUDED_TERM_TEXTS,
    RARE_WORD_BAD_POS_TERMS,
    UNWANTED_TEXTS,
)


def test_bad_pos_term_set_size():
    # RareWordUtil.java lines 32-75 (org.apache.ctakes.gui.dictionary.util).
    assert len(BAD_POS_TERM_SET) == 181


def test_unwanted_texts_size():
    # ctakes-gui/.../data/tiny/UnwantedTexts.txt, comments/blanks stripped.
    assert len(UNWANTED_TEXTS) == 30


def test_no_overlap_between_the_two_lists():
    assert frozenset() == BAD_POS_TERM_SET & UNWANTED_TEXTS


def test_excluded_term_texts_is_the_union():
    assert EXCLUDED_TERM_TEXTS == BAD_POS_TERM_SET | UNWANTED_TEXTS


def test_known_members_found_by_the_parity_diff():
    # tools/diff_parity.py's top false positives against real cTAKES output --
    # these are exactly the two entries the ported lists were found to explain.
    assert "past" in BAD_POS_TERM_SET
    assert "date" in UNWANTED_TEXTS


def test_multi_word_unwanted_text_preserved():
    # UnwantedTexts.txt has one multi-word entry; confirms it wasn't
    # accidentally split into single tokens during transcription.
    assert "at 10" in UNWANTED_TEXTS


def test_rare_word_bad_pos_terms_size():
    # RareWordTermMapCreator.java lines 51-84
    # (org.apache.ctakes.dictionary.lookup2.dictionary). 117 string literals,
    # 114 distinct -- "that", "to" and "which" are each listed twice.
    assert len(RARE_WORD_BAD_POS_TERMS) == 114


def test_the_two_bad_pos_lists_are_not_interchangeable():
    """The runtime list is a strict subset of the GUI build tool's list.

    They look alike enough to swap by accident, and swapping would change
    which token anchors a term's rare-word index entry -- silently altering
    what the dictionary can match rather than raising anything.
    """
    assert RARE_WORD_BAD_POS_TERMS < BAD_POS_TERM_SET
    # Forms of be/have/do, teens and tens, honorifics and "no"/"not" are in the
    # build-tool list only.
    for word in ("be", "have", "is", "not", "no", "thirty", "mrs", "a", "an"):
        assert word in BAD_POS_TERM_SET
        assert word not in RARE_WORD_BAD_POS_TERMS


def test_rare_word_bad_pos_terms_are_normalized():
    for text in RARE_WORD_BAD_POS_TERMS:
        assert text == " ".join(text.casefold().split()), text


def test_all_entries_are_already_normalized():
    # Entries are matched against build_dictionary.normalize()'s output
    # (casefold + whitespace-collapse), so they must already be in that form.
    for text in EXCLUDED_TERM_TEXTS:
        assert text == " ".join(text.casefold().split()), text

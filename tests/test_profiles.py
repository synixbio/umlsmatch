"""Tests for the named configuration profiles.

Profile resolution is pure and needs neither the dictionary nor spaCy, so it is
tested here rather than in `test_analyze.py`, which skips without both. The
pipeline-level tests at the bottom do need them and say so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from umlsmatch.analyze import (
    DEFAULT_PROFILE,
    PROFILES,
    UNSET,
    resolve_profile,
)

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"

needs_pipeline = pytest.mark.skipif(
    not DB.is_file() or importlib.util.find_spec("spacy") is None,
    reason="needs the built dictionary and the optional 'nlp' extra",
)


# --- the profile table ------------------------------------------------------


def test_default_profile_exists():
    assert DEFAULT_PROFILE in PROFILES


def test_every_profile_sets_the_same_keys():
    """A profile that omits a key would inherit it from whichever profile the
    caller last looked at, which is exactly the ambiguity naming them removes."""
    keys = [frozenset(settings) for settings in PROFILES.values()]
    assert len(set(keys)) == 1, f"profiles disagree about which keys they set: {keys}"


def test_the_two_profiles_actually_differ():
    assert PROFILES["strict"] != PROFILES["clinical_recall"]


def test_strict_profile_is_the_conservative_one():
    """Both switches change what is extracted or how history is assigned, and
    every published validation figure is measured with them off. If this flips,
    the numbers in README.md stop describing the default build."""
    assert PROFILES["strict"] == {
        "history_sections": False,
        "drop_header_mentions": False,
    }


def test_clinical_recall_enables_section_aware_history():
    """The switch worth the whole profile mechanism: adjudicated `history_of`
    recall .198 -> .670. See docs/ADJUDICATION_RESULTS.md."""
    assert PROFILES["clinical_recall"]["history_sections"] is True


# --- resolution -------------------------------------------------------------


def test_resolve_none_gives_the_default_profile():
    assert resolve_profile(None) == PROFILES[DEFAULT_PROFILE]


def test_resolve_returns_a_copy():
    """A caller mutating its settings must not edit the table for everyone."""
    settings = resolve_profile("clinical_recall")
    settings["history_sections"] = False
    assert PROFILES["clinical_recall"]["history_sections"] is True


def test_unknown_profile_raises_value_error_naming_the_valid_ones():
    """Reached from a CLI flag and an env var, so the input is a typo far more
    often than a bug -- the error has to say what to type instead."""
    with pytest.raises(ValueError) as exc:
        resolve_profile("clinical-recall")  # hyphen, not underscore
    message = str(exc.value)
    assert "clinical_recall" in message and "strict" in message


# --- the sentinel -----------------------------------------------------------


def test_unset_is_not_none_and_not_falsey_by_accident():
    """`None` already means "not assessed" on `Annotation`. If the constructor
    reused it for "take the profile's value", `history_sections=None` would be
    two different questions with one spelling."""
    assert UNSET is not None
    assert UNSET is not False


# --- the pipeline honours it ------------------------------------------------


@needs_pipeline
def test_pipeline_defaults_to_the_strict_profile():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB) as p:
        assert p.profile == DEFAULT_PROFILE
        assert p.history_sections is False
        assert p.drop_header_mentions is False


@needs_pipeline
def test_named_profile_sets_both_switches():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, profile="clinical_recall") as p:
        assert p.history_sections is True
        assert p.drop_header_mentions is True


@needs_pipeline
def test_explicit_keyword_overrides_the_profile():
    """The profile supplies defaults; it does not overrule the caller."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(
        DB, profile="clinical_recall", drop_header_mentions=False
    ) as p:
        assert p.history_sections is True  # from the profile
        assert p.drop_header_mentions is False  # from the caller
        # `profile` names the starting point, not the final state.
        assert p.profile == "clinical_recall"


@needs_pipeline
def test_explicit_false_is_distinguishable_from_unsupplied():
    """The sentinel earning its keep: `history_sections=False` against a
    profile that sets it True must stick."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, profile="clinical_recall", history_sections=False) as p:
        assert p.history_sections is False


@needs_pipeline
def test_unknown_profile_fails_before_opening_the_dictionary():
    """Opening a 589 MB dictionary and loading a model to then reject a typo is
    a slow way to report a one-word mistake."""
    from umlsmatch import ClinicalPipeline

    with pytest.raises(ValueError, match="unknown profile"):
        ClinicalPipeline(DB, profile="nonsense")


@needs_pipeline
def test_clinical_recall_finds_more_history_than_parity():
    """The profiles differ in output, not just in configuration."""
    from umlsmatch import ClinicalPipeline

    # Bare entries under a PMH heading carrying no per-item cue -- the exact
    # shape `history_sections` exists for, and the one both this pipeline and
    # cTAKES miss without it. Invented, not corpus text.
    note = (
        "PAST MEDICAL HISTORY:\n"
        "Hypertension.\n"
        "Type 2 diabetes mellitus.\n"
        "ASSESSMENT:\n"
        "Patient reports chest pain today.\n"
    )
    with ClinicalPipeline(DB, profile="strict") as p:
        strict = sum(1 for a in p.analyze(note) if a.history_of)
    with ClinicalPipeline(DB, profile="clinical_recall") as p:
        recall = sum(1 for a in p.analyze(note) if a.history_of)
    assert recall > strict, (strict, recall)

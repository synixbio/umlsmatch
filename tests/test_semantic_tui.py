"""Regression tests for the generated UMLS semantic-type tables.

These pin values read directly from the Java source so that a bad regeneration
fails loudly instead of silently degrading entity typing.
"""

from __future__ import annotations

import pytest

from umlsmatch.umls.semantic_tui import (
    MAJOR_GROUPS,
    TUI_NAMES,
    TUI_TO_GROUP,
    SemanticGroup,
    best_group,
    group_for_tui,
    groups_for_tuis,
    normalize_tui,
)

# TUI, expected group member name, expected UMLS semantic type name.
# Transcribed by hand from SemanticTui.java precisely because the generator is
# what is under test -- do not regenerate these.
KNOWN_TUIS = [
    ("T047", "DISORDER", "Disease or Syndrome"),
    ("T116", "DRUG", "Amino Acid, Peptide, or Protein"),
    ("T023", "ANATOMY", "Body Part, Organ, or Organ Component"),
    ("T061", "PROCEDURE", "Therapeutic or Preventive Procedure"),
    ("T201", "CLINICAL_ATTRIBUTE", "Clinical Attribute"),
    ("T203", "DEVICE", "Drug Delivery Device"),
    ("T033", "FINDING", "Finding"),
    ("T005", "DISORDER", "Virus"),
    ("T204", "ENTITY", "Eukaryote"),
    ("T200", "DRUG", "Clinical Drug"),
    ("T060", "PROCEDURE", "Diagnostic Procedure"),
    ("T019", "DISORDER", "Congenital Abnormality"),
]

# NE_TYPE_ID_* values from CONST.java. Note the gap at 4 and the 1001+ range.
EXPECTED_GROUP_CODES = {
    "UNKNOWN": 0,
    "DRUG": 1,
    "DISORDER": 2,
    "FINDING": 3,
    "PROCEDURE": 5,
    "ANATOMY": 6,
    "CLINICAL_ATTRIBUTE": 7,
    "DEVICE": 8,
    "LAB": 9,
    "PHENOMENON": 10,
    "SUBJECT": 1001,
    "TITLE": 1002,
    "EVENT": 1003,
    "ENTITY": 1004,
    "TIME": 1005,
    "MODIFIER": 1006,
    "LAB_MODIFIER": 1007,
}


def test_table_sizes():
    assert len(TUI_TO_GROUP) == 136
    assert len(TUI_NAMES) == 136
    assert len(list(SemanticGroup)) == 17


def test_group_codes_match_java_constants():
    actual = {g.name: g.code for g in SemanticGroup}
    assert actual == EXPECTED_GROUP_CODES


def test_major_groups():
    assert [g.name for g in MAJOR_GROUPS] == [
        "DRUG",
        "DISORDER",
        "FINDING",
        "PROCEDURE",
        "ANATOMY",
    ]


@pytest.mark.parametrize("tui,group,name", KNOWN_TUIS)
def test_known_tui_mappings(tui, group, name):
    assert group_for_tui(tui).name == group
    assert TUI_NAMES[tui] == name


def test_tui_label_matches_numeric_code():
    for tui in TUI_TO_GROUP:
        assert tui.startswith("T")
        assert tui[1:].isdigit()
        assert len(tui) == 4


def test_no_tui_maps_to_unknown():
    """Every listed TUI has a real group; UNKNOWN is only a fallback."""
    assert SemanticGroup.UNKNOWN not in set(TUI_TO_GROUP.values())


@pytest.mark.parametrize("value", ["T047", "t047", "T47", "t47", 47, "47", " T047 "])
def test_normalize_tui_accepts_common_spellings(value):
    assert normalize_tui(value) == "T047"


@pytest.mark.parametrize("value", ["", "X047", "abc", "T-1"])
def test_normalize_tui_rejects_garbage(value):
    with pytest.raises(ValueError):
        normalize_tui(value)


def test_unmapped_tui_falls_back_to_unknown():
    assert group_for_tui(999) is SemanticGroup.UNKNOWN


class TestBestGroup:
    """cTAKES resolves multi-TUI concepts by highest group code."""

    def test_higher_code_wins(self):
        assert best_group([SemanticGroup.DRUG, SemanticGroup.DISORDER]) is SemanticGroup.DISORDER

    def test_minor_group_outranks_major(self):
        # Follows the Java comparator, not the contradictory javadoc. See the
        # docstring on best_group().
        assert (
            best_group([SemanticGroup.PROCEDURE, SemanticGroup.PHENOMENON])
            is SemanticGroup.PHENOMENON
        )

    def test_unknown_always_loses(self):
        assert best_group([SemanticGroup.UNKNOWN, SemanticGroup.DRUG]) is SemanticGroup.DRUG

    def test_empty_is_unknown(self):
        assert best_group([]) is SemanticGroup.UNKNOWN

    def test_only_unknown_is_unknown(self):
        assert best_group([SemanticGroup.UNKNOWN]) is SemanticGroup.UNKNOWN

    def test_order_independent(self):
        groups = [SemanticGroup.DRUG, SemanticGroup.ANATOMY, SemanticGroup.FINDING]
        assert best_group(groups) is best_group(list(reversed(groups)))


def test_groups_for_tuis():
    # A concept carrying both a disorder and an anatomy TUI.
    assert groups_for_tuis(["T047", "T023"]) == {
        SemanticGroup.DISORDER,
        SemanticGroup.ANATOMY,
    }

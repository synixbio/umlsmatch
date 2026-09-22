"""Tests for the shared corpus helpers.

These back the CLI and every batch example, so a regression here breaks all of
them at once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umlsmatch.corpus import (
    TEXT_SUFFIXES,
    document_label,
    duplicate_label_warning,
    duplicate_labels,
    iter_text_files,
    parse_groups,
    read_text,
)


def test_finds_text_files_recursively(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "b.note").write_text("b", encoding="utf-8")

    assert [p.name for p in iter_text_files(tmp_path)] == ["a.txt", "b.note"]


def test_ignores_non_text_suffixes(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("x", encoding="utf-8")
    (tmp_path / "skip.json").write_text("x", encoding="utf-8")

    assert [p.name for p in iter_text_files(tmp_path)] == ["keep.txt"]


def test_results_are_sorted(tmp_path):
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    # Stable ordering matters: two runs over one corpus should be diffable.
    assert [p.name for p in iter_text_files(tmp_path)] == ["a.txt", "b.txt", "c.txt"]


def test_a_plain_file_yields_itself(tmp_path):
    """Callers accept 'a file or a directory' without branching."""
    f = tmp_path / "note.txt"
    f.write_text("x", encoding="utf-8")
    assert list(iter_text_files(f)) == [f]


def test_a_single_file_is_yielded_regardless_of_suffix(tmp_path):
    """Explicitly naming a file overrides the suffix filter -- the CLI relies on
    this so `umlsmatch report.md` works."""
    f = tmp_path / "report.md"
    f.write_text("x", encoding="utf-8")
    assert list(iter_text_files(f)) == [f]


def test_empty_directory_yields_nothing(tmp_path):
    assert list(iter_text_files(tmp_path)) == []


def test_a_directory_named_like_a_note_is_not_yielded(tmp_path):
    """Suffix alone doesn't make it a note -- every caller then tries to read it."""
    (tmp_path / "archive.txt").mkdir()
    (tmp_path / "real.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_text_files(tmp_path)] == ["real.txt"]


def test_custom_suffixes(tmp_path):
    (tmp_path / "a.dat").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_text_files(tmp_path, suffixes={".dat"})] == ["a.dat"]


def test_suffix_matching_is_case_insensitive(tmp_path):
    (tmp_path / "SHOUT.TXT").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_text_files(tmp_path)] == ["SHOUT.TXT"]


def test_text_suffixes_is_immutable():
    with pytest.raises(AttributeError):
        TEXT_SUFFIXES.add(".pdf")  # type: ignore[attr-defined]


def test_read_text_survives_bad_bytes(tmp_path):
    """Clinical exports carry stray encodings; losing a character beats aborting."""
    f = tmp_path / "bad.txt"
    f.write_bytes(b"caf\xe9 au lait")
    assert "au lait" in read_text(f)


def test_read_text_roundtrips_utf8(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("naïve café", encoding="utf-8")
    assert read_text(f) == "naïve café"


def test_document_label_is_the_file_name():
    """Every exporter writes this, so they all have to agree on it."""
    assert document_label("free_texts/synthetic/doc_01.txt") == "doc_01.txt"
    assert document_label(Path("a") / "b" / "note.txt") == "note.txt"


def test_document_label_of_a_bare_name_is_itself():
    assert document_label("note.txt") == "note.txt"


def test_no_duplicate_labels_in_a_flat_corpus():
    files = ["notes/a.txt", "notes/b.txt"]
    assert duplicate_labels(files) == {}
    assert duplicate_label_warning(files) is None


def test_duplicate_labels_across_subdirectories():
    """The cost of keying documents by name: iter_text_files recurses."""
    files = ["2024/a.txt", "2025/a.txt", "2025/b.txt"]
    dupes = duplicate_labels(files)
    assert set(dupes) == {"a.txt"}
    assert [str(p) for p in dupes["a.txt"]] == [str(Path("2024/a.txt")),
                                                str(Path("2025/a.txt"))]


def test_duplicate_label_warning_names_the_offenders():
    warning = duplicate_label_warning(["x/a.txt", "y/a.txt", "z/a.txt"])
    assert warning is not None
    # The count that matters is how many notes merge, not how many names clash.
    assert "a.txt" in warning and "3 notes" in warning


def test_duplicate_label_warning_truncates_a_long_list():
    files = [f"{d}/{n}.txt" for d in ("x", "y") for n in "abcde"]
    warning = duplicate_label_warning(files)
    assert warning is not None and "..." in warning


@pytest.mark.parametrize(
    "value,expected",
    [
        ("DISORDER", {"DISORDER"}),
        ("disorder,drug", {"DISORDER", "DRUG"}),
        (" DISORDER , DRUG ", {"DISORDER", "DRUG"}),
        ("DISORDER,,DRUG", {"DISORDER", "DRUG"}),
    ],
)
def test_parse_groups(value, expected):
    assert parse_groups(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", ",", " , "])
def test_parse_groups_returns_none_for_empty(value):
    """None means 'no filtering'; an empty set would mean 'match nothing'."""
    assert parse_groups(value) is None

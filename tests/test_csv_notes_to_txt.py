"""Tests for the CSV-to-one-note-per-file converter.

This script decides output *filenames* from a column of an upstream export, so
its input is data, not a trusted path component. The two failure modes worth
pinning: an id that escapes --out-dir, and an id that silently overwrites an
earlier note (which shrinks the corpus without saying so).
"""

from __future__ import annotations

from csv_notes_to_txt import convert, safe_stem


def _csv(tmp_path, name: str, rows: str) -> None:
    (tmp_path / name).write_text(rows, encoding="utf-8")


def test_safe_stem_keeps_ordinary_ids():
    # Synthetic id in the shape the real export uses (<pat>_<note>). Never use a
    # real note id here: this file is version-controlled, the corpus is not.
    assert safe_stem("12345678_987654321", "fb") == "12345678_987654321"


def test_safe_stem_neutralizes_path_separators():
    assert "/" not in safe_stem("a/b", "fb")
    assert "\\" not in safe_stem("a\\b", "fb")


def test_safe_stem_neutralizes_parent_traversal():
    stem = safe_stem("../../etc/passwd", "fb")
    assert ".." not in stem
    assert "/" not in stem


def test_safe_stem_falls_back_when_nothing_survives():
    assert safe_stem("...", "fallback") == "fallback"
    assert safe_stem("", "fallback") == "fallback"


def test_convert_writes_one_file_per_row(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _csv(src, "a.csv", "note_id,note_text\n1,first note\n2,second note\n")

    assert convert(src, out, "note_text", "note_id") == 2
    assert (out / "1.txt").read_text(encoding="utf-8") == "first note"
    assert (out / "2.txt").read_text(encoding="utf-8") == "second note"


def test_convert_keeps_a_traversing_id_inside_out_dir(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _csv(src, "a.csv", "note_id,note_text\n../escaped,body\n")

    assert convert(src, out, "note_text", "note_id") == 1
    written = list(out.glob("*.txt"))
    assert len(written) == 1
    assert written[0].parent == out


def test_convert_does_not_lose_a_duplicate_id(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _csv(src, "a.csv", "note_id,note_text\ndup,first\ndup,second\n")

    assert convert(src, out, "note_text", "note_id") == 2
    assert len(list(out.glob("*.txt"))) == 2, "a note was silently overwritten"
    assert "duplicate note id" in capsys.readouterr().err


def test_convert_skips_a_csv_without_the_text_column(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _csv(src, "bad.csv", "id,body\n1,x\n")
    _csv(src, "good.csv", "note_id,note_text\n1,x\n")

    assert convert(src, out, "note_text", "note_id") == 1
    assert "skip bad.csv" in capsys.readouterr().err


def test_convert_falls_back_when_the_id_column_is_missing(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _csv(src, "a.csv", "note_text\nbody one\nbody two\n")

    assert convert(src, out, "note_text", "note_id") == 2
    assert {p.name for p in out.glob("*.txt")} == {"a_0.txt", "a_1.txt"}

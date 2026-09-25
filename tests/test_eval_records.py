"""Tests for the shared silver-standard JSONL reader.

Every scorer in umlsmatch.eval reads its input through this, so "what counts
as a record" and "what a malformed line looks like" are answered once here.
"""

from __future__ import annotations

import json

import pytest

from umlsmatch.eval.records import iter_records


def _write(tmp_path, lines: str):
    path = tmp_path / "silver.jsonl"
    path.write_text(lines, encoding="utf-8")
    return path


def test_reads_one_record_per_line(tmp_path):
    path = _write(tmp_path, '{"source_file": "a"}\n{"source_file": "b"}\n')
    assert [r["source_file"] for r in iter_records(path)] == ["a", "b"]


def test_skips_blank_lines(tmp_path):
    """A trailing newline is normal and must not raise."""
    path = _write(tmp_path, '{"a": 1}\n\n   \n{"a": 2}\n')
    assert [r["a"] for r in iter_records(path)] == [1, 2]


def test_empty_file_yields_nothing(tmp_path):
    assert list(iter_records(_write(tmp_path, ""))) == []


def test_malformed_line_names_the_line_number(tmp_path):
    path = _write(tmp_path, '{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match=r":2: not valid JSON"):
        list(iter_records(path))


def test_error_names_the_file(tmp_path):
    path = _write(tmp_path, "{oops\n")
    # Escaped: `match=` is a regex, and an unescaped '.' would also accept
    # "silverXjsonl" -- weakening the assertion this test exists to make.
    with pytest.raises(ValueError, match=r"silver\.jsonl"):
        list(iter_records(path))


def test_is_lazy(tmp_path):
    """Streaming, not slurping -- corpora run to hundreds of MB."""
    path = _write(tmp_path, '{"a": 1}\nnot json\n')
    stream = iter_records(path)
    assert next(stream)["a"] == 1  # the bad line hasn't been reached yet


def test_reads_unicode_as_utf8(tmp_path):
    path = _write(tmp_path, json.dumps({"text": "café naïve"}) + "\n")
    assert next(iter(iter_records(path)))["text"] == "café naïve"

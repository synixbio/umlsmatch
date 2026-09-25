"""Per-run output directories: naming, collisions, the manifest, and the CLI.

The unit tests here are deliberately *not* gated on the dictionary or the
optional `nlp` extra. :mod:`umlsmatch.runs` is standard library only, and a
module that can be imported on a bare checkout should have tests that run
there too -- the same reasoning as ``tests/test_zero_dependency_core.py``. The
CLI tests further down need a real pipeline and carry their own skips.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from umlsmatch.runs import MANIFEST_NAME, RunWriter, new_run_id, safe_name

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "umls_sno_rx.sqlite"


# --- run identifiers ----------------------------------------------------------


def test_run_id_is_sortable_and_utc():
    stamp = datetime(2026, 9, 21, 7, 14, 55, tzinfo=timezone.utc)
    assert new_run_id(stamp).startswith("20260921T071455Z-")


def test_run_ids_from_the_same_second_differ():
    """Scripted fan-out starts several runs inside one second."""
    stamp = datetime(2026, 9, 21, 7, 14, 55, tzinfo=timezone.utc)
    assert len({new_run_id(stamp) for _ in range(50)}) == 50


def test_run_ids_sort_chronologically():
    early = new_run_id(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    late = new_run_id(datetime(2026, 11, 2, 3, 4, 5, tzinfo=timezone.utc))
    assert early < late


# --- filename derivation ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        # Underscores are preserved; only the directory and extension go.
        ("free_texts/synthetic/doc_01_addendum.txt", "doc_01_addendum"),
        # The CLI's label for piped text. The angle brackets are illegal in a
        # Windows filename, so this one is load-bearing rather than cosmetic.
        ("<stdin>", "stdin"),
        ("a/b/note.text", "note"),
        ("weird name (1).txt", "weird_name_1"),
        # Nothing survives sanitizing; a generic name beats an empty one.
        ("???.txt", "document"),
    ],
)
def test_safe_name(label, expected):
    assert safe_name(label) == expected


def test_safe_name_never_returns_a_path_separator():
    assert "/" not in safe_name("a/b/c.txt")
    assert "\\" not in safe_name("a\\b\\c.txt")


# --- RunWriter ----------------------------------------------------------------


def test_writer_creates_its_own_directory(tmp_path):
    w = RunWriter(tmp_path, run_id="r1")
    assert w.directory == tmp_path / "r1"
    assert w.directory.is_dir()


def test_writer_refuses_an_existing_directory(tmp_path):
    """Reusing a run id would mix two runs under one manifest."""
    RunWriter(tmp_path, run_id="r1")
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path, run_id="r1")


def test_two_runs_coexist(tmp_path):
    a = RunWriter(tmp_path)
    b = RunWriter(tmp_path)
    assert a.directory != b.directory
    assert a.directory.is_dir() and b.directory.is_dir()


def test_path_for_disambiguates_collisions(tmp_path):
    """`iter_text_files` recurses, so equal stems are expected, not exotic."""
    w = RunWriter(tmp_path, run_id="r1", suffix=".jsonl")
    names = [w.path_for(f"{d}/note.txt").name for d in ("a", "b", "c")]
    assert names == ["note.jsonl", "note-2.jsonl", "note-3.jsonl"]


def test_path_for_uses_the_suffix(tmp_path):
    w = RunWriter(tmp_path, run_id="r1", suffix=".jsonl")
    assert w.path_for("x/note.txt").name == "note.jsonl"


def test_manifest_totals_and_mapping(tmp_path):
    w = RunWriter(tmp_path, run_id="r1", suffix=".jsonl")
    for src, n in (("a/note.txt", 3), ("b/note.txt", 4)):
        w.record(src, w.path_for(src), n)
    path = w.write_manifest(tool="umlsmatch")

    assert path.name == MANIFEST_NAME
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m["run_id"] == "r1"
    assert m["tool"] == "umlsmatch"
    assert m["totals"] == {"documents": 2, "annotations": 7}
    # The mapping is what keeps disambiguated names traceable to their source.
    assert [d["source"] for d in m["documents"]] == ["a/note.txt", "b/note.txt"]
    assert [d["file"] for d in m["documents"]] == ["note.jsonl", "note-2.jsonl"]


def test_artifact_reserves_the_name_verbatim(tmp_path):
    """Merged outputs are named by the caller, not derived from a note path."""
    w = RunWriter(tmp_path, run_id="r1", suffix=".txt")
    assert w.artifact("annotations.jsonl") == w.directory / "annotations.jsonl"


@pytest.mark.parametrize("name", ["../escape.json", "a/b.json", ".", ".."])
def test_artifact_refuses_to_leave_the_run_directory(tmp_path, name):
    w = RunWriter(tmp_path, run_id="r1")
    with pytest.raises(ValueError):
        w.artifact(name)


def test_artifact_run_records_its_own_totals(tmp_path):
    """A merged-artifact run has no per-document files to derive totals from.

    Deriving them anyway would report two zeroes and describe a run that
    processed a whole corpus as having done nothing.
    """
    w = RunWriter(tmp_path, run_id="r1")
    path = w.artifact("annotations.jsonl")
    w.record_artifact(path, documents=20, annotations=2188, failed=0)
    m = json.loads(
        w.write_manifest(totals={"documents": 20, "annotations": 2188}).read_text(
            encoding="utf-8"
        )
    )

    assert m["totals"] == {"documents": 20, "annotations": 2188}
    assert m["artifacts"] == [
        {"file": "annotations.jsonl", "documents": 20, "annotations": 2188, "failed": 0}
    ]
    # Nothing per-document happened, so the key is absent rather than empty.
    assert "documents" not in m


def test_a_per_document_run_records_no_artifacts(tmp_path):
    w = RunWriter(tmp_path, run_id="r1", suffix=".jsonl")
    w.record("a/note.txt", w.path_for("a/note.txt"), 3)
    m = json.loads(w.write_manifest().read_text(encoding="utf-8"))
    assert "artifacts" not in m
    assert m["totals"]["documents"] == 1


# --- finding the newest artifact ----------------------------------------------


def _touch(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


def test_latest_artifact_picks_the_newest(tmp_path):
    from umlsmatch.runs import latest_artifact

    _touch(tmp_path / "r1" / "annotations.db", 1_000)
    newest = _touch(tmp_path / "r2" / "annotations.db", 2_000)
    _touch(tmp_path / "r3" / "annotations.db", 1_500)

    assert latest_artifact([".db"], root=tmp_path) == newest


def test_latest_artifact_goes_by_mtime_not_run_id(tmp_path):
    """A long job finishes after a short one that started later.

    Run ids are creation timestamps, so ordering by name would hand back the
    directory that was *created* last rather than the one that finished.
    """
    from umlsmatch.runs import latest_artifact

    finished_last = _touch(tmp_path / "20260921T070000Z-aaaaaa" / "a.db", 9_000)
    _touch(tmp_path / "20260921T080000Z-bbbbbb" / "a.db", 1_000)

    assert latest_artifact([".db"], root=tmp_path) == finished_last


def test_latest_artifact_filters_by_suffix(tmp_path):
    from umlsmatch.runs import latest_artifact

    _touch(tmp_path / "r1" / "annotations.jsonl", 5_000)
    db = _touch(tmp_path / "r1" / "annotations.db", 1_000)

    assert latest_artifact([".db"], root=tmp_path) == db


def test_latest_artifact_is_case_insensitive(tmp_path):
    from umlsmatch.runs import latest_artifact

    db = _touch(tmp_path / "r1" / "ANNOTATIONS.DB", 1_000)
    assert latest_artifact([".db"], root=tmp_path) == db


def test_latest_artifact_returns_none_when_empty(tmp_path):
    from umlsmatch.runs import latest_artifact

    _touch(tmp_path / "r1" / "annotations.jsonl", 1_000)
    assert latest_artifact([".db"], root=tmp_path) is None


def test_latest_artifact_returns_none_for_a_missing_root(tmp_path):
    """Nothing has been run yet -- a caller's problem to report, not a crash."""
    from umlsmatch.runs import latest_artifact

    assert latest_artifact([".db"], root=tmp_path / "nope") is None


def test_latest_artifact_is_deterministic_on_equal_mtimes(tmp_path):
    from umlsmatch.runs import latest_artifact

    _touch(tmp_path / "r1" / "a.db", 1_000)
    _touch(tmp_path / "r2" / "a.db", 1_000)
    picks = {latest_artifact([".db"], root=tmp_path) for _ in range(10)}
    assert len(picks) == 1


def test_find_artifacts_is_newest_first(tmp_path):
    """The order a caller walks when the newest is not one it can use."""
    from umlsmatch.runs import find_artifacts

    _touch(tmp_path / "r1" / "a.db", 1_000)
    mid = _touch(tmp_path / "r2" / "b.db", 2_000)
    newest = _touch(tmp_path / "r3" / "c.db", 3_000)

    found = find_artifacts([".db"], root=tmp_path)
    assert found[0] == newest
    assert found[1] == mid
    assert len(found) == 3


def test_find_artifacts_empty_for_a_missing_root(tmp_path):
    from umlsmatch.runs import find_artifacts

    assert find_artifacts([".db"], root=tmp_path / "nope") == []


def test_latest_artifact_agrees_with_find_artifacts(tmp_path):
    """`latest_artifact` is the head of the list; they must not drift."""
    from umlsmatch.runs import find_artifacts, latest_artifact

    _touch(tmp_path / "r1" / "a.db", 1_000)
    _touch(tmp_path / "r2" / "b.db", 3_000)
    _touch(tmp_path / "r3" / "c.db", 2_000)

    assert latest_artifact([".db"], root=tmp_path) == find_artifacts(
        [".db"], root=tmp_path
    )[0]


def test_sqlite_suffixes_cover_both_spellings(tmp_path):
    """Both SQLite tools resolve "the newest database" through this one set.

    `load_to_sqlite.py` writes whatever name it is given (`.db` in the docs),
    `notes_of_interest.py` writes `.sqlite`, and the built dictionaries are
    `.sqlite` too. A copy of this list in either script is one that gains a
    suffix while the other does not, and then the two disagree about which
    database is newest.
    """
    from umlsmatch.runs import SQLITE_SUFFIXES, latest_artifact

    assert {".db", ".sqlite"} <= set(SQLITE_SUFFIXES)

    _touch(tmp_path / "r1" / "annotations.db", 1_000)
    newest = _touch(tmp_path / "r2" / "notes_of_interest.sqlite", 2_000)
    assert latest_artifact(SQLITE_SUFFIXES, root=tmp_path) == newest


def test_manifest_stamp_carries_shared_provenance():
    from umlsmatch import __version__
    from umlsmatch.runs import manifest_stamp

    stamp = manifest_stamp(script="parse_to_jsonl.py")
    assert stamp["tool"] == "umlsmatch"
    assert stamp["version"] == __version__
    assert stamp["script"] == "parse_to_jsonl.py"
    # Parses as the UTC instant it claims to be.
    datetime.strptime(stamp["created"], "%Y-%m-%dT%H:%M:%SZ")


def test_manifest_is_utf8_json(tmp_path):
    w = RunWriter(tmp_path, run_id="r1")
    w.record("nóte.txt", w.path_for("nóte.txt"), 1)
    m = json.loads(w.write_manifest().read_text(encoding="utf-8"))
    assert m["documents"][0]["source"] == "nóte.txt"


# --- the PHI gate -------------------------------------------------------------


def test_the_manifest_name_is_covered_by_the_phi_pattern():
    """A run manifest lists source paths, which here are patient identifiers.

    The annotation files beside it are caught by the .jsonl/.txt suffix rules;
    the manifest is JSON, which the pattern matches by name rather than by
    suffix. If it stops being listed, a run directory becomes committable.
    """
    pattern_file = REPO / ".githooks" / "phi-paths.pattern"
    pattern = next(
        line
        for line in pattern_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert re.search(pattern, f"runs/20260921T071455Z-3f9a1c/{MANIFEST_NAME}")


def test_the_manifest_name_is_gitignored():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert MANIFEST_NAME in text


# --- CLI ----------------------------------------------------------------------

cli = pytest.mark.skipif(
    not DB.is_file() or importlib.util.find_spec("spacy") is None,
    reason="needs the built dictionary and the optional 'nlp' extra",
)

NOTE = "Patient denies chest pain. Started on metformin 500 mg.\n"


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "one.txt").write_text(NOTE, encoding="utf-8")
    (d / "two.txt").write_text("Mother had breast cancer.\n", encoding="utf-8")
    return d


@cli
def test_out_dir_writes_one_file_per_document(corpus, tmp_path):
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB), "--json",
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0
    run = tmp_path / "runs" / "r1"
    assert sorted(p.name for p in run.iterdir()) == sorted(
        [MANIFEST_NAME, "one.jsonl", "two.jsonl"]
    )


@cli
def test_out_dir_concatenates_to_the_merged_output(corpus, tmp_path):
    """`cat run/*.jsonl` has to reproduce what `-o` would have written."""
    from umlsmatch.__main__ import main

    merged = tmp_path / "merged.jsonl"
    assert main([str(corpus), "--db", str(DB), "--json", "-o", str(merged)]) == 0
    assert main([str(corpus), "--db", str(DB), "--json",
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0

    run = tmp_path / "runs" / "r1"
    concat = "".join(
        p.read_text(encoding="utf-8") for p in sorted(run.glob("*.jsonl"))
    )
    assert concat == merged.read_text(encoding="utf-8")


@cli
def test_a_document_with_no_annotations_still_gets_a_file(tmp_path):
    """A missing file would not distinguish "none found" from "not processed"."""
    from umlsmatch.__main__ import main

    d = tmp_path / "notes"
    d.mkdir()
    (d / "blank.txt").write_text("", encoding="utf-8")
    assert main([str(d), "--db", str(DB), "--json",
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0

    rec = json.loads((tmp_path / "runs" / "r1" / "blank.jsonl").read_text("utf-8"))
    assert rec["annotations"] == []


@cli
def test_two_runs_do_not_overwrite_each_other(corpus, tmp_path):
    from umlsmatch.__main__ import main

    root = tmp_path / "runs"
    for _ in range(2):
        assert main([str(corpus), "--db", str(DB), "--json", "--out-dir", str(root)]) == 0
    assert len(list(root.iterdir())) == 2


@cli
def test_manifest_records_the_effective_settings_not_the_flags(corpus, tmp_path):
    """--profile sets history_sections without it ever appearing on argv."""
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB), "--json", "--profile", "clinical_recall",
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0

    m = json.loads(
        (tmp_path / "runs" / "r1" / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert m["settings"]["profile"] == "clinical_recall"
    assert m["settings"]["history_sections"] is True
    assert m["settings"]["drop_header_mentions"] is True
    assert m["format"] == "jsonl"
    assert m["totals"]["documents"] == 2


@cli
def test_manifest_totals_match_the_files(corpus, tmp_path):
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB), "--json",
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0
    run = tmp_path / "runs" / "r1"
    m = json.loads((run / MANIFEST_NAME).read_text(encoding="utf-8"))

    for entry in m["documents"]:
        rec = json.loads((run / entry["file"]).read_text(encoding="utf-8"))
        assert len(rec["annotations"]) == entry["annotations"]
    assert m["totals"]["annotations"] == sum(
        d["annotations"] for d in m["documents"]
    )


@cli
def test_out_dir_text_mode_omits_the_document_banner(corpus, tmp_path):
    """The filename already names the source; the `=== label ===` line is noise."""
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB),
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0
    body = (tmp_path / "runs" / "r1" / "one.txt").read_text(encoding="utf-8")
    assert "===" not in body
    assert "NEGATED" in body


@cli
def test_out_dir_reports_its_directory_even_under_json(corpus, tmp_path, capsys):
    """Nothing reaches stdout, so a silent run would hide a generated name."""
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB), "--json",
                 "--out-dir", str(tmp_path / "runs")]) == 0
    assert str(tmp_path / "runs") in capsys.readouterr().err


@cli
def test_duplicate_run_id_exits_2(corpus, tmp_path, capsys):
    from umlsmatch.__main__ import main

    root = tmp_path / "runs"
    argv = [str(corpus), "--db", str(DB), "--out-dir", str(root), "--run-id", "r1"]
    assert main(argv) == 0
    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("conflict", [["--no-save"], ["-o", "merged.jsonl"]])
def test_cli_rejects_run_id_with_nothing_to_name(conflict):
    """--run-id names a run directory, so it needs one to be written."""
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), "--run-id", "r1", *conflict])


@pytest.mark.parametrize(
    "flags",
    [
        ["-o", "a.jsonl", "--out-dir", "runs"],
        ["-o", "a.jsonl", "--no-save"],
        ["--out-dir", "runs", "--no-save"],
    ],
)
def test_cli_rejects_two_destinations(flags):
    """Each of the three answers "where does output go?"; two would ignore one."""
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), *flags])


# --- saving is the default ----------------------------------------------------


@cli
def test_a_plain_run_saves_without_being_asked(corpus, tmp_path, monkeypatch):
    """The headline behaviour: no flags, and the run is still on disk."""
    from umlsmatch.__main__ import main
    from umlsmatch.runs import DEFAULT_RUN_ROOT

    monkeypatch.chdir(tmp_path)
    assert main([str(corpus), "--db", str(DB), "--json"]) == 0

    runs = list((tmp_path / DEFAULT_RUN_ROOT).iterdir())
    assert len(runs) == 1
    assert sorted(p.name for p in runs[0].iterdir()) == sorted(
        [MANIFEST_NAME, "one.jsonl", "two.jsonl"]
    )


def test_the_default_root_is_under_out():
    """`out/` is already gitignored and matched by the PHI pattern.

    Saving by default writes note text to disk unasked, so the destination has
    to be somewhere the existing PHI rules already cover. A new top-level
    directory would have been protected only by the suffix rules, and the
    manifest is not a covered suffix.
    """
    from umlsmatch.runs import DEFAULT_RUN_ROOT

    assert DEFAULT_RUN_ROOT.parts[0] == "out"
    pattern_file = REPO / ".githooks" / "phi-paths.pattern"
    pattern = next(
        line
        for line in pattern_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    default = DEFAULT_RUN_ROOT.as_posix()
    assert re.search(pattern, f"{default}/20260921T071455Z-3f9a1c/{MANIFEST_NAME}")
    assert re.search(pattern, f"{default}/20260921T071455Z-3f9a1c/note.jsonl")


@cli
def test_a_plain_run_still_prints_to_stdout(corpus, tmp_path, monkeypatch, capsys):
    """Saving is additive: it must not have quietly taken stdout away."""
    from umlsmatch.__main__ import main

    monkeypatch.chdir(tmp_path)
    assert main([str(corpus), "--db", str(DB)]) == 0
    out = capsys.readouterr()
    assert "NEGATED" in out.out
    assert "-> " in out.err  # and it says where the copy went


@cli
def test_no_save_writes_nothing(corpus, tmp_path, monkeypatch, capsys):
    from umlsmatch.__main__ import main

    monkeypatch.chdir(tmp_path)
    assert main([str(corpus), "--db", str(DB), "--no-save"]) == 0
    assert not (tmp_path / "out").exists()
    assert "NEGATED" in capsys.readouterr().out


@cli
def test_out_saves_no_run_directory(corpus, tmp_path, monkeypatch):
    """`-o` is an explicit destination; a second copy was not asked for."""
    from umlsmatch.__main__ import main

    monkeypatch.chdir(tmp_path)
    merged = tmp_path / "merged.jsonl"
    assert main([str(corpus), "--db", str(DB), "--json", "-o", str(merged)]) == 0
    assert merged.is_file()
    assert not (tmp_path / "out").exists()


@cli
def test_out_dir_sends_nothing_to_stdout(corpus, tmp_path, capsys):
    from umlsmatch.__main__ import main

    assert main([str(corpus), "--db", str(DB),
                 "--out-dir", str(tmp_path / "runs"), "--run-id", "r1"]) == 0
    assert capsys.readouterr().out == ""


@cli
def test_run_id_alone_names_the_default_run(corpus, tmp_path, monkeypatch):
    """No --out-dir needed: there is always a run directory to name now."""
    from umlsmatch.__main__ import main
    from umlsmatch.runs import DEFAULT_RUN_ROOT

    monkeypatch.chdir(tmp_path)
    assert main([str(corpus), "--db", str(DB), "--run-id", "mine"]) == 0
    assert (tmp_path / DEFAULT_RUN_ROOT / "mine" / MANIFEST_NAME).is_file()

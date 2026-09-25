"""Tests for the unified dictionary build entry point.

What matters here is the *ordering* and which options reach which stage -- the
stages themselves are three existing scripts with their own tests. `--dry-run`
makes that testable without a UMLS release and without minutes of work per
assertion.
"""

from __future__ import annotations

import pytest

from umlsmatch.build import STAGES, main


def _commands(capsys, argv):
    assert main([*argv, "--dry-run"]) == 0
    printed = capsys.readouterr().out.splitlines()
    return [line for line in printed if not line.startswith("#")]


# --- ordering ---------------------------------------------------------------


def test_stage_order_is_the_dependency_order():
    """Retokenize before indexing is not a preference. Rare-word statistics
    computed over pre-retokenization spellings key the index on strings the
    tokenizer will never produce."""
    assert [s.name for s in STAGES] == ["dictionary", "retokenize", "rare-index"]


def test_runs_all_three_stages_in_order(capsys):
    commands = _commands(capsys, ["--umls-dir", "META"])
    assert len(commands) == 3
    for stage, command in zip(STAGES, commands, strict=True):
        assert stage.script in command


# --- option routing ---------------------------------------------------------


def test_stage_one_writes_to_out_and_the_rest_read_db(capsys):
    """The scripts share `--db` in name only: stage 1 spells its output
    `--out`, so forwarding every flag to every stage would fail mid-build."""
    build, retokenize, index = _commands(capsys, ["--umls-dir", "META", "--db", "d.sqlite"])
    assert "--out" in build and "--db" not in build
    assert "--db" in retokenize
    assert "--db" in index


def test_stage_specific_flags_go_only_to_their_stage(capsys):
    build, retokenize, index = _commands(
        capsys,
        [
            "--umls-dir", "META",
            "--tui-set", "all",
            "--model", "en_core_sci_sm",
            "--ctakes-root", "../ctakes-java",
        ],
    )
    assert "--tui-set" in build and "--tui-set" not in retokenize
    assert "en_core_sci_sm" in retokenize and "en_core_sci_sm" not in build
    assert "--ctakes-root" in index and "--ctakes-root" not in build


def test_unset_options_are_omitted_rather_than_passed_empty(capsys):
    """Passing `--sources ''` is not the same as not passing it; the scripts
    have their own defaults and those must survive."""
    build, _, _ = _commands(capsys, ["--umls-dir", "META"])
    for absent in ("--sources", "--code-sources", "--limit", "--keep-semantic-tags"):
        assert absent not in build


# --- resuming ---------------------------------------------------------------


def test_from_skips_earlier_stages(capsys, tmp_path):
    db = tmp_path / "d.sqlite"
    db.touch()
    commands = _commands(capsys, ["--from", "retokenize", "--db", str(db)])
    assert len(commands) == 2
    assert "retokenize_terms.py" in commands[0]


def test_only_runs_exactly_one_stage(capsys, tmp_path):
    db = tmp_path / "d.sqlite"
    db.touch()
    commands = _commands(capsys, ["--only", "rare-index", "--db", str(db)])
    assert len(commands) == 1
    assert "build_rare_word_index.py" in commands[0]


def test_from_and_only_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["--from", "retokenize", "--only", "rare-index", "--dry-run"])


# --- refusing to start a build that cannot work -----------------------------


def test_umls_dir_is_required_when_stage_one_runs():
    with pytest.raises(SystemExit):
        main(["--dry-run"])


def test_umls_dir_is_not_required_when_stage_one_is_skipped(capsys, tmp_path):
    db = tmp_path / "d.sqlite"
    db.touch()
    assert main(["--from", "retokenize", "--db", str(db), "--dry-run"]) == 0


def test_resuming_onto_a_missing_dictionary_is_refused(tmp_path):
    """`--from retokenize` against a database that was never built would run
    two stages against nothing and report success on an empty artifact."""
    with pytest.raises(SystemExit):
        main(["--from", "retokenize", "--db", str(tmp_path / "absent.sqlite"), "--dry-run"])

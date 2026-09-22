"""The PHI path pattern, and the two gates that are supposed to share it.

This is the only test here whose failure cannot be fixed by a later commit.
Every other rule in this repository is recoverable: a bug ships and is fixed, a
document goes stale and is corrected. Note text reaching a remote is removed
only by rewriting history, and only if someone notices.

**Why this file exists.** The pattern would otherwise be written out three
times -- in ``.githooks/pre-commit`` and twice in ``.github/workflows/ci.yml``
-- and copies drift. A corpus path that moves in the hook but not in CI lets the
raw clinical notes past the job whose own comment calls it "what makes the PHI
gate non-optional", and nothing fails, because nothing compares them.

So the pattern lives in ``.githooks/phi-paths.pattern`` and both read it.
That fixes the drift and introduces a new way to fail silently -- a caller that
stops reading the file, or reverts to an inline regex, is invisible until it
matters. Hence :func:`test_the_hook_reads_the_shared_pattern` and
:func:`test_ci_reads_the_shared_pattern`, which are the load-bearing tests in
this module even though the behavioural ones below look more interesting.

**The carve-out.** ``.githooks/phi-allow.pattern`` takes paths back out of the
rule above -- currently the synthetic corpus, which is committed. It inverts
the risk of everything described so far: phi-paths.pattern fails closed when it
is wrong and announces itself by refusing something, while a carve-out that is
wrong fails open and announces nothing, because a gate that has stopped
refusing is indistinguishable from a clean repository. The tests that matter
for it are therefore the negative ones --
:func:`test_the_carve_out_exempts_no_phi` and
:func:`test_both_gates_refuse_to_treat_an_empty_carve_out_as_no_filter`.

**These tests do not need git, a corpus, or a dictionary.** They read the three
files as text and run the pattern with :mod:`re`. That keeps them running on a
fresh clone and in CI, which is where they have to run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PATTERN_FILE = REPO / ".githooks" / "phi-paths.pattern"
ALLOW_FILE = REPO / ".githooks" / "phi-allow.pattern"
HOOK = REPO / ".githooks" / "pre-commit"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def _first_pattern_line(path: Path) -> str | None:
    """Parse a pattern file the same way the shell callers do.

    Kept deliberately identical to ``grep -Ev '^[[:space:]]*(#|$)' ... | head -1``:
    a parser here that is more permissive than the shell's would pass a file the
    real gates read differently, which is the failure this module exists to
    prevent rather than to introduce.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line
    return None


def _pattern() -> str:
    line = _first_pattern_line(PATTERN_FILE)
    if line is None:
        raise AssertionError(f"no pattern line in {PATTERN_FILE}")
    return line


def _allow() -> str | None:
    """The carve-out from phi-allow.pattern, or None when there is none.

    "No exceptions" and "a pattern matching nothing" are deliberately not the
    same value here, because they are not the same thing in the shell: the hook
    and CI skip the filter entirely when this file is absent or empty, since
    ``grep -Eiv ""`` matches every line and would invert the gate into refusing
    nothing. Returning None models that skip.
    """
    if not ALLOW_FILE.is_file():
        return None
    return _first_pattern_line(ALLOW_FILE)


def _blocked(path: str) -> bool:
    """The gate as the two callers actually apply it: refuse, then carve out.

    Every behavioural test below goes through this rather than through
    :func:`_pattern` alone -- the raw pattern still matches the synthetic
    corpus, and it is the carve-out that makes those notes committable.
    """
    if not re.search(_pattern(), path, re.I):
        return False
    allow = _allow()
    return not (allow and re.search(allow, path, re.I))


# --- the wiring ---------------------------------------------------------------


def test_the_shared_pattern_file_exists_and_parses():
    assert PATTERN_FILE.is_file(), f"{PATTERN_FILE} is missing; both gates read it"
    pattern = _pattern()
    assert pattern.strip() == pattern, "leading/trailing space would reach grep verbatim"
    re.compile(pattern)  # raises if it is not a valid regex


def test_the_pattern_file_holds_exactly_one_pattern():
    """A second uncommented line would be silently ignored by ``head -1``.

    The trap this guards is specific: someone adds a pattern on a new line,
    sees it in the file, and assumes it is enforced. It is not -- both callers
    take the first line only, which is why the file's own header says so.
    """
    live = [
        line
        for line in PATTERN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(live) == 1, f"expected one pattern line, found {len(live)}: {live}"


def test_the_hook_reads_the_shared_pattern():
    """The hook must source the file, not carry its own copy."""
    text = HOOK.read_text(encoding="utf-8")
    assert "phi-paths.pattern" in text, "the hook stopped reading the shared pattern"


def test_ci_reads_the_shared_pattern():
    """Both of CI's checks -- tree and history -- must use the shared pattern."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "phi-paths.pattern" in text, "CI stopped reading the shared pattern"
    assert text.count('grep -Ei "$pat"') == 2, (
        "CI runs a tree check and a history check and both must use the shared "
        "pattern; found a different number of uses"
    )


def test_the_carve_out_file_exists_and_parses():
    assert ALLOW_FILE.is_file(), f"{ALLOW_FILE} is missing; both gates read it"
    allow = _allow()
    assert allow, f"{ALLOW_FILE} holds no pattern line"
    assert allow.strip() == allow, "leading/trailing space would reach grep verbatim"
    re.compile(allow)


def test_the_carve_out_holds_exactly_one_pattern():
    """Same ``head -1`` trap as the block pattern, with the stakes reversed.

    A second line here is not merely ignored -- it is an exception someone
    believes they have written and has not, which they will discover when a
    commit they expected to pass is refused, or not at all.
    """
    live = [
        line
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(live) == 1, f"expected one pattern line, found {len(live)}: {live}"


def test_both_gates_read_the_carve_out():
    """A gate that reads only the block pattern refuses the synthetic corpus.

    That direction fails safe, so nothing leaks -- but it also makes the
    committed corpus uncommittable, which is how an exception file quietly
    stops being applied and then stops being maintained.
    """
    assert "phi-allow.pattern" in HOOK.read_text(encoding="utf-8"), (
        "the hook stopped reading the carve-out"
    )
    ci = WORKFLOW.read_text(encoding="utf-8")
    assert "phi-allow.pattern" in ci, "CI stopped reading the carve-out"
    assert ci.count("| carve") == 2, (
        "CI's tree and history checks must both apply the carve-out; found a "
        "different number of uses"
    )


def test_both_gates_refuse_to_treat_an_empty_carve_out_as_no_filter():
    """The fail-open hazard, asserted on the text of the gates themselves.

    ``grep -Eiv ""`` matches every line and inverts to nothing: without the
    emptiness guard, deleting phi-allow.pattern would not restore the strict
    gate, it would disable it entirely. Checked as text because the alternative
    is running the shell, which these tests deliberately do not do.
    """
    assert '-n "$phi_allow"' in HOOK.read_text(encoding="utf-8"), (
        "the hook applies the carve-out without checking it is non-empty; an "
        "empty pattern would invert into refusing nothing"
    )
    assert '-n "$allow"' in WORKFLOW.read_text(encoding="utf-8"), (
        "CI applies the carve-out without checking it is non-empty; an empty "
        "pattern would invert into refusing nothing"
    )


@pytest.mark.parametrize("path", [HOOK, WORKFLOW])
def test_neither_gate_carries_an_inlined_copy(path):
    """The specific regression: a hard-coded path list reappearing in a gate.

    Matched on ``free_texts`` next to a suffix alternation rather than on the
    exact old string, so a *reformatted* inline copy is caught too. The pattern
    file itself is exempt -- it is the one place this belongs.
    """
    text = path.read_text(encoding="utf-8")
    inlined = re.findall(r"\^\(data\|out[^']*\)/", text)
    assert not inlined, (
        f"{path.name} carries an inlined PHI pattern {inlined}; it must read "
        ".githooks/phi-paths.pattern instead"
    )


# --- the behaviour ------------------------------------------------------------


#: Paths that must be refused. Written in the corpus's real *layout* rather
#: than as minimal examples: the bug being guarded against was a pattern that
#: looked right and did not match the layout on disk.
#:
#: The note identifiers are placeholder zeros on purpose. Real notes are named
#: with patient and note ids, which is exactly what this gate exists to keep out
#: of the repository -- so writing a genuine one here, in a file that *is*
#: committed, would leak the identifier the test is about.
MUST_BLOCK = [
    "free_texts/notes/00000000_000000000_0000000000.csv",
    "free_texts/json/silver.jsonl",
    "free_texts/xmi_out/00000000_000000000_0000000000.txt.xmi",
    "free_texts/adjudication/subject/review.csv",
    "free_texts/adjudication/subject/design.json",
    "free_texts/adjudication/subject/_verdicts.json",
    "free_texts/adjudication/subject/_pairs.json",
    "out/annotations.jsonl",
    "out/annotations.db",
    "data/umls_sno_rx.sqlite",
    # The same artifacts written somewhere new -- a reviewer saving a review
    # file to the desktop, a script defaulting its output to the repo root.
    "review.csv",
    "notes.txt",
    "annotations.parquet",
    "scratch/annotations.db",
    "somewhere/design.json",
    # The carve-out covers free_texts/synthetic/*.txt and must reach no
    # further. These are what a too-broad exception would let through: derived
    # artifacts written into the synthetic directory by the same pipeline that
    # runs over the real corpus, and a nested subtree created beneath it.
    "free_texts/synthetic/annotations.jsonl",
    "free_texts/synthetic/review.csv",
    "free_texts/synthetic/annotations.parquet",
    "free_texts/synthetic/run_manifest.json",
    "free_texts/synthetic/nested/leaked.txt",
]

#: Paths that must be allowed. Every one is a file this repository tracks or
#: plausibly will; a gate that blocks these gets disabled, and a disabled gate
#: protects nothing.
MUST_ALLOW = [
    "src/umlsmatch/analyze.py",
    "src/umlsmatch/py.typed",
    "tests/test_phi_guard.py",
    "tools/score_attributes.py",
    "docs/USER_GUIDE.md",
    "README.md",
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    ".githooks/pre-commit",
    ".githooks/phi-paths.pattern",
    ".githooks/phi-allow.pattern",
    ".github/workflows/ci.yml",
    ".vscode/settings.json",
    "examples/README.md",
    # The synthetic corpus. These match phi-paths.pattern twice over -- the
    # free_texts/ prefix and the .txt suffix -- and reach this list only
    # because the carve-out takes them back out, which is exactly the path
    # through the gate that needs a test.
    "free_texts/synthetic/doc_01.txt",
    "free_texts/synthetic/doc_20.txt",
]


@pytest.mark.parametrize("path", MUST_BLOCK)
def test_phi_bearing_paths_are_refused(path):
    assert _blocked(path), f"{path} would reach a remote"


@pytest.mark.parametrize("path", MUST_ALLOW)
def test_ordinary_project_files_are_allowed(path):
    assert not _blocked(path), (
        f"{path} is blocked; a gate that blocks ordinary files gets bypassed"
    )


@pytest.mark.parametrize("path", MUST_BLOCK)
def test_the_carve_out_exempts_no_phi(path):
    """The carve-out must never be the reason a PHI path gets through.

    This is the failure mode the block pattern does not have: widening this
    file is silent, because a gate that has stopped refusing things looks
    exactly like a repository with nothing to refuse. Asserted against the
    carve-out alone, so it still holds if the block pattern is later narrowed.
    """
    allow = _allow()
    assert not (allow and re.search(allow, path, re.I)), (
        f"the carve-out in {ALLOW_FILE.name} exempts {path}; it is too broad"
    )


def test_the_synthetic_corpus_on_disk_is_committable():
    """The carve-out against the real directory, not a list of strings.

    MUST_ALLOW names two notes; this covers whatever is actually there, so a
    file whose name the pattern does not anticipate -- a subdirectory, a
    rename away from doc_NN.txt -- is caught here rather than by someone
    wondering why their commit is refused.
    """
    corpus = REPO / "free_texts" / "synthetic"
    if not corpus.is_dir():
        pytest.skip("synthetic corpus not present")
    notes = sorted(corpus.glob("*.txt"))
    assert notes, f"{corpus} holds no .txt notes; the carve-out covers nothing"
    refused = [
        rel
        for p in notes
        if _blocked(rel := str(p.relative_to(REPO)).replace("\\", "/"))
    ]
    assert not refused, (
        "these synthetic notes are refused by the gate and cannot be "
        "committed:\n  " + "\n  ".join(refused)
    )


def test_nothing_tracked_today_would_be_blocked():
    """Run the gate over the repository as it actually is.

    The parametrized cases above are a fixed list and will age. This walks the
    working tree, so a file added later that the pattern happens to catch fails
    here rather than at someone's commit.

    Ignored directories are skipped by name rather than by asking git, so this
    needs no subprocess and works from a tarball. ``free_texts`` is among them
    and now contains one directory that is *not* ignored, the synthetic corpus;
    that is covered by
    :func:`test_the_synthetic_corpus_on_disk_is_committable` instead, which
    asserts the opposite of this test and would be hidden by the skip here.
    """
    # `dist`/`build` are here for the same reason as `data` and `out`: gitignored,
    # so nothing inside them can be committed and the gate has no say over them.
    # Not hypothetical -- `python -m build` writes
    # `dist/*.dist-info/entry_points.txt`, which the pattern's `\.txt$` suffix
    # clause matches, so without these two entries building a release and then
    # running the suite fails it.
    skip = {".git", ".venv", "__pycache__", "data", "out", "free_texts",
            "dist", "build", ".pytest_cache", ".ruff_cache", ".obsidian",
            "node_modules"}
    blocked = [
        rel
        for p in REPO.rglob("*")
        if p.is_file()
        and not skip & set(p.relative_to(REPO).parts)
        and _blocked(rel := str(p.relative_to(REPO)).replace("\\", "/"))
    ]
    assert not blocked, (
        "these committable files match the PHI pattern:\n  " + "\n  ".join(blocked)
    )

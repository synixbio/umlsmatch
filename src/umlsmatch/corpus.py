"""Locating and reading a corpus of plain-text notes.

Small helpers shared by the CLI (:mod:`umlsmatch.__main__`) and the scripts in
``examples/``. They live in the package rather than in an examples-only module
because the shipped CLI needs exactly the same behaviour -- keeping one
definition means "which files count as notes?" has a single answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = [
    "TEXT_SUFFIXES",
    "document_label",
    "duplicate_label_warning",
    "duplicate_labels",
    "iter_text_files",
    "parse_groups",
    "read_text",
]

#: Extensions treated as clinical notes when scanning a directory.
TEXT_SUFFIXES = frozenset({".txt", ".text", ".note"})


def iter_text_files(
    root: str | Path, *, suffixes: Iterable[str] = TEXT_SUFFIXES
) -> Iterator[Path]:
    """Yield note files under `root`, recursively, in sorted order.

    Sorted so that repeated runs over the same corpus produce output in the same
    order -- diffing two runs is otherwise needlessly painful. A plain file path
    is yielded as itself, so callers can accept "a file or a directory" without
    branching.

    Only regular files are yielded: a directory named ``archive.txt`` matches
    the suffix test but is not a note, and every caller immediately tries to
    read what it gets back.
    """
    root = Path(root)
    wanted = {s.lower() for s in suffixes}
    if root.is_file():
        yield root
        return
    yield from sorted(
        p for p in root.rglob("*") if p.suffix.lower() in wanted and p.is_file()
    )


def read_text(path: str | Path) -> str:
    """Read a note as UTF-8, replacing undecodable bytes rather than raising.

    Clinical exports routinely carry stray encodings; losing one character beats
    aborting a corpus run.
    """
    return Path(path).read_text(encoding="utf-8", errors="replace")


def document_label(path: str | Path) -> str:
    """How a document is named in an export: its file name.

    One definition because every exporter in ``examples/`` writes this field --
    as ``document`` in the CSV and Parquet tables, as ``source`` in JSONL and in
    ``documents.source`` in SQLite -- and a join across two of those formats
    only works if they agree. They did not: the tabular exporters wrote the file
    name while the JSONL ones wrote the full path, so loading both into one
    database produced two rows per note and double-counted everything.

    The file name rather than the path because it is the part that identifies
    the note: this corpus encodes patient and note ids in the name, while the
    directory above it records only where the corpus happened to be mounted,
    which differs between machines and would make two runs of the same corpus
    fail to join. The cost is :func:`duplicate_labels` -- two notes in different
    subdirectories can share a name, and this cannot tell them apart.
    """
    return Path(path).name


def duplicate_labels(files: Iterable[str | Path]) -> dict[str, list[Path]]:
    """File names claimed by more than one path, as ``label -> paths``.

    :func:`iter_text_files` recurses, so this is a real case rather than a
    hypothetical one: ``2024/notes/a.txt`` and ``2025/notes/a.txt`` are two
    documents that :func:`document_label` calls the same thing, and an export
    keyed by label silently merges them.
    """
    seen: dict[str, list[Path]] = {}
    for f in files:
        seen.setdefault(document_label(f), []).append(Path(f))
    return {label: paths for label, paths in seen.items() if len(paths) > 1}


def duplicate_label_warning(files: Iterable[str | Path]) -> str | None:
    """A ready-to-print warning for :func:`duplicate_labels`, or None.

    Formatted here rather than in each exporter so the five of them cannot
    describe the same hazard five different ways -- and returned rather than
    printed so this module stays free of output policy.
    """
    dupes = duplicate_labels(files)
    if not dupes:
        return None
    worst = max(dupes.values(), key=len)
    shown = ", ".join(sorted(dupes)[:3]) + (" ..." if len(dupes) > 3 else "")
    return (
        f"warning: {len(dupes)} file name(s) are shared by more than one note "
        f"({shown}); up to {len(worst)} notes will be merged into one document "
        f"in the output, because documents are keyed by file name.\n"
        f"         Flatten the corpus or rename the duplicates to keep them apart."
    )


def parse_groups(value: str | None) -> set[str] | None:
    """Parse a ``--groups DISORDER,DRUG`` argument into a set, or None.

    None means "no filtering", which is distinct from an empty set ("match
    nothing") -- so a blank or omitted value must not collapse to ``set()``.
    """
    if not value:
        return None
    groups = {g.strip().upper() for g in value.split(",") if g.strip()}
    return groups or None

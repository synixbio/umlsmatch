"""Per-run output directories: one annotation file per document, plus a manifest.

The CLI's ``-o`` writes every document into a single file, which answers "what
did this corpus produce?" but not "what did *this run* produce, and with which
settings?". Re-running overwrites the answer, so two configurations cannot be
compared without remembering to rename the output by hand -- and the settings
that produced a file live only in the shell history that invoked it.

``--out-dir`` writes each run into its own directory instead::

    runs/
    |-- 20260921T071455Z-3f9a1c/
    |   |-- run_manifest.json
    |   |-- doc_01.jsonl
    |   |-- doc_02.jsonl
    |   `-- ...
    `-- 20260921T072310Z-b1e0d4/
        `-- ...

Nothing is ever overwritten, one file per note stays diffable against the next
run's, and :data:`MANIFEST_NAME` records the configuration alongside the output
rather than in the operator's memory.

**Everything written here is PHI.** The annotation files quote note text
verbatim and the manifest records source paths, which in this corpus carry
patient and note identifiers. Both are covered by ``.gitignore`` and
``.githooks/phi-paths.pattern`` -- by suffix for the annotations, by name for
the manifest, which is why that name is distinctive rather than ``manifest``.

This module is deliberately standard library only, like the rest of the core
package: a run directory is written on the batch path, where the optional web
and NLP extras may not be installed.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_RUN_ROOT",
    "MANIFEST_NAME",
    "SQLITE_SUFFIXES",
    "RunWriter",
    "find_artifacts",
    "latest_artifact",
    "manifest_stamp",
    "new_run_id",
    "safe_name",
]

#: Extensions treated as a SQLite database when scanning a run root.
#:
#: Here rather than in one of the example scripts for the same reason
#: :data:`umlsmatch.corpus.TEXT_SUFFIXES` is: two tools now ask "which files
#: count as a database?" -- ``sqlite_reader.py`` and ``sqlite_browser.py`` --
#: and a second copy is one that will eventually gain ``.sqlite3`` while the
#: other does not, so the two disagree about what "the newest database" means.
SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})

#: Where the CLI puts run directories when ``--out-dir`` is not given.
#:
#: Under ``out/`` deliberately, and not a new top-level directory. Saving is on
#: by default, which means the CLI writes note text to disk without being asked
#: -- so the default destination has to be one the PHI rules already cover.
#: ``out/`` is listed in ``.gitignore`` and matched by
#: ``.githooks/phi-paths.pattern``; a fresh ``runs/`` would have been protected
#: only by the suffix rules, and the manifest is not a covered suffix.
DEFAULT_RUN_ROOT = Path("out") / "runs"

#: Filename of the per-run provenance record, inside the run directory.
#:
#: Not plain ``manifest.json``: the PHI pattern blocks JSON by name rather than
#: by suffix (a blanket ``\.json$`` would also block ``.vscode/settings.json``),
#: so the name has to be distinctive enough to add to that list without
#: catching unrelated project files that happen to be called "manifest".
MANIFEST_NAME = "run_manifest.json"

#: Characters kept in a generated filename. Everything else collapses to "_".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def new_run_id(now: datetime | None = None) -> str:
    """A sortable, collision-resistant identifier for one run.

    UTC and fixed-width so a directory listing sorts chronologically -- local
    time would reorder itself twice a year, and the point of these directories
    is comparing a run against the one before it.

    The random suffix is not decoration. Scripted fan-out (a shell loop over
    profiles, a CI matrix) starts several runs inside the same second, and a
    second-resolution timestamp alone would hand two of them the same directory.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def safe_name(label: str) -> str:
    """A filesystem-safe stem for the document labelled `label`.

    Derived from the input filename so output is recognisable next to input.
    ``<stdin>`` -- the CLI's label for piped text -- would otherwise produce a
    name Windows refuses outright, so the sanitizing pass is not merely
    cosmetic: the angle brackets collapse to underscores and then strip to
    ``stdin``.
    """
    stem = Path(label).stem or label
    cleaned = _UNSAFE.sub("_", stem).strip("._-")
    # Every character was punctuation. Rare, but a silent empty filename is a
    # worse outcome than a generic one.
    return cleaned or "document"


def find_artifacts(
    suffixes: Iterable[str], *, root: str | Path = DEFAULT_RUN_ROOT
) -> list[Path]:
    """Matching files under `root`, newest first.

    Ordered by modification time rather than by run id: a run directory is
    named when it is *created*, and a long corpus job finishes after a short
    one that started later. What a caller means by "the last one" is the one
    that finished with the file in it.

    The path is the tie-break, so the order is deterministic on filesystems
    with coarse timestamps rather than varying with directory order.

    The whole list rather than just the newest, because "newest" is not always
    what a caller can use: a tool that reads one particular schema needs to
    walk down the list until it finds one it understands, and to say how many
    it rejected when it does not.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    wanted = {s.lower() for s in suffixes}
    found = [p for p in root.rglob("*") if p.suffix.lower() in wanted and p.is_file()]
    return sorted(found, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def latest_artifact(
    suffixes: Iterable[str], *, root: str | Path = DEFAULT_RUN_ROOT
) -> Path | None:
    """The most recently modified matching file under `root`, or None.

    Lets a tool default to "whatever I produced last" instead of making the
    caller paste a generated run id back in -- the id is a timestamp precisely
    so that nothing overwrites anything, which is exactly what makes it
    tedious to retype.
    """
    found = find_artifacts(suffixes, root=root)
    return found[0] if found else None


def manifest_stamp(**extra: Any) -> dict[str, Any]:
    """The provenance every run manifest carries, plus whatever `extra` adds.

    Shared so the CLI and the example scripts cannot drift into stamping runs
    differently -- comparing two run directories is the whole point, and that
    gets harder the moment one of them spells ``created`` another way.
    """
    # Local import: this module is otherwise standalone, and importing the
    # package at module scope would make `umlsmatch.runs` unimportable from
    # inside `umlsmatch/__init__.py` should it ever want it.
    from umlsmatch import __version__

    return {
        "tool": "umlsmatch",
        "version": __version__,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }


class RunWriter:
    """Allocates one output file per document under a fresh run directory.

    Writes are streamed by the caller rather than buffered here: a corpus run
    holds one document's annotations at a time, and collecting every file's
    contents to write at the end would undo that for no gain.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str | None = None,
        suffix: str = ".txt",
    ) -> None:
        self.run_id = run_id or new_run_id()
        self.directory = Path(root) / self.run_id
        self.suffix = suffix
        # exist_ok=False, and the error is allowed to propagate. A directory
        # that is already there means a caller-supplied --run-id names an
        # earlier run: writing into it would mix two runs' files under one
        # manifest, with the earlier run's leftovers indistinguishable from
        # this one's output. That is precisely what per-run directories exist
        # to prevent, so it fails instead.
        self.directory.mkdir(parents=True, exist_ok=False)
        self._used: set[str] = set()
        self._documents: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, Any]] = []

    def path_for(self, label: str) -> Path:
        """Reserve and return the output path for the document `label`.

        Two notes in different input directories can share a filename, and
        :func:`~umlsmatch.corpus.iter_text_files` recurses, so collisions are
        expected rather than hypothetical. The second gets ``-2``, the third
        ``-3``; the manifest records which source produced which file, so the
        mapping stays unambiguous even once the names diverge from the inputs.
        """
        base = safe_name(label)
        name = base
        n = 2
        while name in self._used:
            name = f"{base}-{n}"
            n += 1
        self._used.add(name)
        return self.directory / f"{name}{self.suffix}"

    def artifact(self, name: str) -> Path:
        """Reserve `name` inside the run directory, used verbatim.

        The counterpart to :meth:`path_for`, for a run whose output is one
        merged file for the whole corpus -- ``annotations.jsonl``,
        ``annotations.csv`` -- rather than a file per note. The name is a
        caller's literal rather than a note path, so there is nothing to
        sanitize; it is rejected outright if it would leave the directory.
        """
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"artifact name must be a bare filename, got {name!r}")
        self._used.add(Path(name).stem)
        return self.directory / name

    def record(self, source: str, path: Path, n_annotations: int) -> None:
        """Note that `source` was written to `path`, for the manifest."""
        self._documents.append(
            {
                "source": source,
                "file": path.name,
                "annotations": n_annotations,
            }
        )

    def record_artifact(self, path: Path, **stats: Any) -> None:
        """Note that a merged output was written, with whatever counts fit it.

        `stats` is open rather than a fixed signature because the scripts that
        use this produce different things: rows for a CSV, documents and
        annotations for JSONL. Forcing one shape would have most of them
        recording a field that does not apply.
        """
        self._artifacts.append({"file": path.name, **stats})

    def write_manifest(self, **fields: Any) -> Path:
        """Write :data:`MANIFEST_NAME` and return its path.

        `fields` is merged in ahead of the per-document table, so the caller
        controls what provenance is recorded without this module having to
        import the pipeline to ask.
        """
        manifest: dict[str, Any] = {"run_id": self.run_id, **fields}
        # Derived only when there are per-document files to derive it from. A
        # run that produced one merged artifact has its own totals, passed in
        # `fields`, and overwriting those with two zeroes would report the run
        # as having done nothing.
        if self._documents:
            manifest["totals"] = {
                "documents": len(self._documents),
                "annotations": sum(d["annotations"] for d in self._documents),
            }
            manifest["documents"] = self._documents
        if self._artifacts:
            manifest["artifacts"] = self._artifacts
        path = self.directory / MANIFEST_NAME
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

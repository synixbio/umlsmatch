"""Reading silver-standard JSONL, one definition for every scorer.

``tools/run_java_ctakes.py`` writes one JSON object per line. The four scoring
modules in this package all consume that file. Inlining a read loop in each
invites them to disagree on edge cases such as a trailing blank line. One
definition means one answer to "what counts as a record", and a
parse error names the line it came from instead of surfacing as a bare
``JSONDecodeError`` from somewhere in a long corpus run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

__all__ = ["iter_records"]


def iter_records(jsonl_path: str | Path) -> Iterator[dict]:
    """Yield each record in a silver-standard JSONL file, skipping blank lines."""
    path = Path(jsonl_path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc

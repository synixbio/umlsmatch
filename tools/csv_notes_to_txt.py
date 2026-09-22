#!/usr/bin/env python3
"""Flatten CSV-exported clinical notes into one plain-text file per note.

cTAKES' ``AbstractFileTreeReader`` (used by ``runPiperFile -i``) treats each
file in the input directory as one whole document of plain text. It cannot
be pointed at CSV exports with columns like
``pat_id,note_id,recorded_datetime,note_type,note_text`` -- it would tokenize
the header row, commas, and quoting as if they were clinical narrative.

This script reads every ``*.csv`` in ``--csv-dir``, pulls the ``--text-column``
field out of each row, and writes it to ``<--id-column value>.txt`` under
``--out-dir`` -- one document per row, ready for
``tools/run_java_ctakes.py --input-dir``.

Usage::

    python tools/csv_notes_to_txt.py \\
        --csv-dir free_texts/notes \\
        --out-dir free_texts/synthetic

Note: the CSVs handled here contain PHI. This script only reads/writes local
files; it does not print, log, or transmit note text anywhere.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Note ids come from an upstream export, so they are data, not trusted path
# components: anything outside this set is replaced before the id becomes a
# filename. Without it a value like "../x" or "a/b" writes outside --out-dir.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_stem(note_id: str, fallback: str) -> str:
    """Reduce a note id to a filename-safe stem, or `fallback` if nothing remains."""
    stem = _UNSAFE_IN_FILENAME.sub("_", note_id).strip("._")
    return stem or fallback


def convert(csv_dir: Path, out_dir: Path, text_column: str, id_column: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: dict[str, str] = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if text_column not in (reader.fieldnames or []):
                print(
                    f"skip {csv_path.name}: no column {text_column!r} "
                    f"(found {reader.fieldnames})",
                    file=sys.stderr,
                )
                continue
            for i, row in enumerate(reader):
                fallback = f"{csv_path.stem}_{i}"
                stem = safe_stem(row.get(id_column) or fallback, fallback)
                # Two rows sharing an id would silently leave one note on disk
                # and quietly shrink the corpus, so say so and keep both.
                if stem in seen:
                    print(
                        f"warning: duplicate note id {stem!r} "
                        f"({csv_path.name} row {i}, first seen in {seen[stem]}); "
                        f"writing as {stem}_{i}",
                        file=sys.stderr,
                    )
                    stem = f"{stem}_{i}"
                seen[stem] = csv_path.name
                (out_dir / f"{stem}.txt").write_text(row.get(text_column, ""), encoding="utf-8")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv-dir", type=Path, required=True, help="Folder of note CSV files")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Folder to write one .txt per note"
    )
    parser.add_argument(
        "--text-column", default="note_text", help="CSV column holding the note body"
    )
    parser.add_argument(
        "--id-column", default="note_id", help="CSV column used as the output filename"
    )
    args = parser.parse_args()

    if not args.csv_dir.is_dir():
        sys.exit(f"error: not a directory: {args.csv_dir}")

    count = convert(args.csv_dir, args.out_dir, args.text_column, args.id_column)
    print(f"Wrote {count} note .txt files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

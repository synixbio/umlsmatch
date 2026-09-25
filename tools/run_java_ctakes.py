#!/usr/bin/env python3
"""Run the real Java cTAKES over a folder of notes to produce silver-standard output.

Wraps ``bin/runPiperFile`` from an installed cTAKES binary distribution and,
optionally, flattens each produced XMI into a JSONL record via
``umlsmatch.silver.xmi``. This is the "(2) Java cTAKES output as silver
standard" path -- run once, then diff the
Python pipeline's output against these records.

Usage::

    python tools/run_java_ctakes.py \\
        --input-dir corpus/notes \\
        --output-dir corpus/silver_xmi \\
        --jsonl corpus/silver.jsonl

Autodetects a cTAKES install under ``D:/apps/apache-ctakes-*`` (or set
``CTAKES_HOME`` / pass ``--ctakes-home`` explicitly). Defaults to
``DefaultFastPipeline.piper`` -- the pipeline documented in
the reference pipeline -- which requires the UMLS dictionary already be
installed at ``resources/org/apache/ctakes/dictionary/lookup/fast``
(``bin/getUmlsDictionary`` in the cTAKES install).

To re-flatten an existing run's XMI without re-invoking Java::

    python tools/run_java_ctakes.py --skip-run \\
        --output-dir corpus/silver_xmi --jsonl corpus/silver.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umlsmatch.silver.xmi import parse_ctakes_xmi

DEFAULT_PIPER = "org/apache/ctakes/clinical/pipeline/DefaultFastPipeline.piper"


def find_ctakes_home() -> Path | None:
    env = os.environ.get("CTAKES_HOME")
    if env:
        return Path(env)
    for candidate in sorted(Path("D:/apps").glob("apache-ctakes-*")):
        if (candidate / "bin").is_dir():
            return candidate
    return None


def run_piper(ctakes_home: Path, input_dir: Path, output_dir: Path, piper: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    script_name = "runPiperFile.bat" if sys.platform == "win32" else "runPiperFile.sh"
    script = ctakes_home / "bin" / script_name
    if not script.exists():
        raise FileNotFoundError(f"{script} not found -- is --ctakes-home correct?")
    # runPiperFile cd's into CTAKES_HOME before invoking Java, so relative
    # input/output paths must be resolved against the caller's cwd first.
    cmd = [
        str(script),
        "-p", piper,
        "-i", str(input_dir.resolve()),
        "-o", str(output_dir.resolve()),
        "--xmiOut", str(output_dir.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=ctakes_home)


def export_jsonl(output_dir: Path, jsonl_path: Path, expected_stems: set[str] | None = None) -> int:
    """Flatten `output_dir`'s XMI files into `jsonl_path`.

    If `expected_stems` is given (the filenames actually processed this run),
    XMI files whose stem doesn't match one are skipped. `output_dir` is a
    long-lived cache directory -- without this filter, a stale or unrelated
    *.xmi left over from a previous run (different corpus, or accidentally
    pointed at raw non-.txt input) gets silently folded into the silver
    standard. `expected_stems` is None only for --skip-run, where there is no
    current input list to check against.
    """
    count = 0
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for xmi_path in sorted(output_dir.glob("*.xmi")):
            if expected_stems is not None and xmi_path.stem not in expected_stems:
                print(f"skip {xmi_path.name}: not part of this run's --input-dir", file=sys.stderr)
                continue
            doc = parse_ctakes_xmi(xmi_path)
            record = {
                "source_file": xmi_path.stem,  # "<name>.txt" from "<name>.txt.xmi"
                "text": doc.text,
                "sentences": [list(s) for s in doc.sentences],
                "tokens": [list(t) for t in doc.tokens],
                # (begin, end, token_type, part_of_speech); POS is null for
                # non-word tokens. Consumed by tools/score_pos.py.
                "pos_tokens": [list(t) for t in doc.pos_tokens],
                "mentions": [
                    {
                        "type": m.type,
                        "text": m.text,
                        "begin": m.begin,
                        "end": m.end,
                        "negated": m.negated,
                        "uncertain": m.uncertain,
                        "conditional": m.conditional,
                        "generic": m.generic,
                        "subject": m.subject,
                        "history_of": m.history_of,
                        "concepts": [
                            {
                                "cui": c.cui,
                                "tui": c.tui,
                                "scheme": c.coding_scheme,
                                "preferred_text": c.preferred_text,
                            }
                            for c in m.concepts
                        ],
                    }
                    for m in doc.mentions
                ],
            }
            fh.write(json.dumps(record) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ctakes-home", type=Path, default=find_ctakes_home(),
        help="Path to an installed cTAKES binary distribution (default: autodetect under D:/apps)",
    )
    parser.add_argument("--input-dir", type=Path, help="Folder of plain-text clinical notes")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Folder to write/read cTAKES XMI output"
    )
    parser.add_argument(
        "--piper", default=DEFAULT_PIPER, help=f"Piper file to run (default: {DEFAULT_PIPER})"
    )
    parser.add_argument(
        "--jsonl", type=Path, default=None,
        help="If given, flatten the XMI output into this JSONL silver-standard file",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip invoking cTAKES; only convert an existing --output-dir of XMI files to --jsonl",
    )
    args = parser.parse_args()

    if not args.skip_run:
        if args.input_dir is None:
            parser.error("--input-dir is required unless --skip-run is given")
        if not args.input_dir.is_dir():
            parser.error(f"--input-dir is not a directory: {args.input_dir}")
        if args.ctakes_home is None:
            parser.error(
                "--ctakes-home not given and no apache-ctakes-* install found under D:/apps"
            )
        print(f"Running {args.piper} over {args.input_dir} with cTAKES at {args.ctakes_home} ...")
        run_piper(args.ctakes_home, args.input_dir, args.output_dir, args.piper)

    if args.jsonl:
        if not args.output_dir.is_dir():
            parser.error(f"--output-dir has no XMI to read: {args.output_dir}")
        expected_stems = (
            {p.name for p in args.input_dir.iterdir() if p.is_file()}
            if not args.skip_run
            else None
        )
        count = export_jsonl(args.output_dir, args.jsonl, expected_stems)
        print(f"Wrote {count} silver-standard records to {args.jsonl}")
        if not count:
            print("warning: no records written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

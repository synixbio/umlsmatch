"""One command for the three-stage dictionary build: ``python -m umlsmatch.build``.

    python -m umlsmatch.build --umls-dir "D:/umls/2026AA/META"

A usable dictionary is three scripts run in a strict order:

1. ``tools/build_dictionary.py``     -- MRCONSO/MRSTY -> SQLite
2. ``tools/retokenize_terms.py``     -- re-spell every term the way the
                                        tokenizer segments text
3. ``tools/build_rare_word_index.py`` -- rare-word statistics over the *final*
                                        spellings

**Stage 2 is the one that gets skipped**, and until
:meth:`~umlsmatch.dictionary.matcher.RareWordMatcher._verify_retokenized` it
cost nothing visible: the build succeeded, the pipeline started, and ~96k
hyphenated and clitic terms -- roughly 10% of the vocabulary -- silently never
matched. That check now makes a skipped stage a loud failure at load time; this
module makes it hard to do in the first place. Together they are belt and
braces on the same mistake, which is proportionate for one whose only symptom
is a slightly shorter annotation list.

Stage 3 after stage 2 is equally load-bearing and easier to overlook: rare-word
statistics computed over pre-retokenization spellings index the wrong keys, so
re-running stage 2 alone leaves the dictionary just as broken.

The individual scripts remain the supported way to resume a long build -- a
full 2026AA run reads ~2.1 GB and stage 1 alone takes many minutes, so a
failure in stage 3 should not mean redoing stage 1. ``--from`` does that here
too.

Run from a repository checkout: this drives the scripts under ``tools/``, which
are not part of the installed wheel. Building a dictionary needs a local UMLS
release anyway, so it is a checkout-side operation either way.

Standard library only -- no third-party dependencies. The stages themselves
have their own requirements (stage 2 needs the ``nlp`` extra), and each says so.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["STAGES", "Stage", "find_tools_dir", "main"]


@dataclass(frozen=True)
class Stage:
    """One step of the build, and the script that performs it."""

    #: ``--from`` / ``--only`` spelling.
    name: str
    script: str
    #: Shown while it runs, so a long build says what it is doing.
    description: str


#: In dependency order. Order is the whole point of this module, so it lives in
#: one list rather than in the control flow of :func:`main`.
STAGES: tuple[Stage, ...] = (
    Stage("dictionary", "build_dictionary.py", "parsing MRCONSO/MRSTY into SQLite"),
    Stage("retokenize", "retokenize_terms.py", "re-spelling terms for the tokenizer"),
    Stage("rare-index", "build_rare_word_index.py", "indexing by rarest token"),
)


def find_tools_dir() -> Path:
    """The repository's ``tools/`` directory.

    Resolved from this file rather than from the working directory, so the
    command works from anywhere inside a checkout.
    """
    tools = Path(__file__).resolve().parent.parent.parent / "tools"
    if not tools.is_dir():
        raise SystemExit(
            "error: cannot find the tools/ directory.\n"
            "`python -m umlsmatch.build` drives the build scripts, which ship "
            "with the repository rather than with the installed package. Run it "
            "from a checkout:\n"
            "  git clone <repo> && cd <repo> && pip install -e '.[nlp]'"
        )
    return tools


def _stage_argv(stage: Stage, tools: Path, args: argparse.Namespace) -> list[str]:
    """The full command for `stage`, with only the options that stage accepts.

    Passing every flag to every script would be simpler and wrong: they share
    ``--db`` in name only (stage 1 spells its output ``--out``), and an
    unrecognized flag is an argparse error two minutes into a long run.
    """
    argv = [sys.executable, str(tools / stage.script)]
    if stage.name == "dictionary":
        argv += ["--umls-dir", str(args.umls_dir), "--out", str(args.db)]
        if args.sources is not None:
            argv += ["--sources", args.sources]
        if args.code_sources is not None:
            argv += ["--code-sources", args.code_sources]
        if args.tui_set is not None:
            argv += ["--tui-set", args.tui_set]
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        if args.keep_semantic_tags:
            argv += ["--keep-semantic-tags"]
        if args.keep_excluded_texts:
            argv += ["--keep-excluded-texts"]
    elif stage.name == "retokenize":
        argv += ["--db", str(args.db)]
        if args.model is not None:
            argv += ["--model", args.model]
        if args.force:
            argv += ["--force"]
    else:
        argv += ["--db", str(args.db)]
        if args.ctakes_root is not None:
            argv += ["--ctakes-root", str(args.ctakes_root)]
    return argv


def _selected(args: argparse.Namespace) -> list[Stage]:
    """Stages to run, honouring ``--from`` and ``--only``."""
    if args.only:
        return [s for s in STAGES if s.name == args.only]
    names = [s.name for s in STAGES]
    return list(STAGES[names.index(args.start) :])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build. Returns the process exit status."""
    ap = argparse.ArgumentParser(
        prog="python -m umlsmatch.build",
        description="Build a UMLS lookup dictionary: all three stages, in order.",
        epilog=(
            "Each stage is also runnable on its own from tools/; this exists so "
            "that running them out of order, or skipping one, is not the default "
            "way to get it wrong."
        ),
    )
    ap.add_argument(
        "--umls-dir",
        type=Path,
        default=None,
        help="UMLS META directory (required unless --from skips stage 1)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("data/umls_sno_rx.sqlite"),
        help="dictionary to build (default: data/umls_sno_rx.sqlite)",
    )
    stage_names = [s.name for s in STAGES]
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument(
        "--from",
        dest="start",
        choices=stage_names,
        default=stage_names[0],
        help="resume from this stage, keeping what earlier stages already wrote",
    )
    resume.add_argument(
        "--only",
        choices=stage_names,
        default=None,
        help="run exactly one stage. You are then responsible for the ordering "
        "this command exists to guarantee.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run, and run none of them",
    )

    stage1 = ap.add_argument_group("stage 1 (build_dictionary.py)")
    stage1.add_argument("--sources", default=None, help="vocabularies to read synonyms from")
    stage1.add_argument(
        "--code-sources",
        default=None,
        help="vocabularies a concept must carry a code in",
    )
    stage1.add_argument("--tui-set", choices=("ctakes", "all"), default=None)
    stage1.add_argument(
        "--limit", type=int, default=None, help="read only N lines per file (smoke test)"
    )
    stage1.add_argument("--keep-semantic-tags", action="store_true")
    stage1.add_argument("--keep-excluded-texts", action="store_true")

    stage2 = ap.add_argument_group("stage 2 (retokenize_terms.py)")
    stage2.add_argument("--model", default=None, help="spaCy model (default: the tokenizer's)")
    stage2.add_argument("--force", action="store_true", help="re-tokenize even if already done")

    stage3 = ap.add_argument_group("stage 3 (build_rare_word_index.py)")
    stage3.add_argument(
        "--ctakes-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="clone of apache/ctakes, to re-check the vendored BAD_POS_TERMS",
    )

    args = ap.parse_args(argv)
    stages = _selected(args)

    needs_umls = any(s.name == "dictionary" for s in stages)
    if needs_umls and args.umls_dir is None:
        ap.error("--umls-dir is required for the 'dictionary' stage")
    if not needs_umls and not args.db.is_file():
        ap.error(
            f"--db {args.db} does not exist, and the stage that creates it is "
            "being skipped. Drop --from/--only, or point --db at a built dictionary."
        )

    tools = find_tools_dir()
    commands = [(s, _stage_argv(s, tools, args)) for s in stages]

    if args.dry_run:
        for stage, argv_ in commands:
            print(f"# {stage.description}")
            print(shlex.join(argv_))
        return 0

    for i, (stage, argv_) in enumerate(commands, start=1):
        print(f"\n=== [{i}/{len(commands)}] {stage.name}: {stage.description} ===", flush=True)
        # argv is built by _stage_argv from parsed options, never a shell string.
        result = subprocess.run(argv_)
        if result.returncode != 0:
            # Print the failing command rather than only the code: the stages
            # are individually runnable, and re-running one by hand after a fix
            # is the whole reason they stayed separate scripts.
            print(
                f"\nerror: stage '{stage.name}' failed with exit code "
                f"{result.returncode}:\n  {shlex.join(argv_)}\n"
                f"Fix it and resume with:\n"
                f"  python -m umlsmatch.build --from {stage.name} --db {args.db}"
                + (f" --umls-dir {args.umls_dir}" if args.umls_dir else ""),
                file=sys.stderr,
            )
            return result.returncode

    print(f"\ndictionary ready: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

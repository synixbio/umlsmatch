"""Command-line interface: ``python -m umlsmatch``.

Analyze clinical text from files, a directory, or stdin.

    python -m umlsmatch note.txt
    python -m umlsmatch free_texts/synthetic/ --json -o out.jsonl
    echo "Patient denies chest pain." | python -m umlsmatch -
    python -m umlsmatch note.txt --groups DISORDER,DRUG --negated-only

**Every run saves its own annotation files.** Alongside the listing on stdout,
each invocation writes a directory under ``out/runs/`` holding one annotation
file per input document and a manifest of the settings that produced them, so
two configurations can be compared without either overwriting the other::

    python -m umlsmatch free_texts/synthetic/          # -> out/runs/<run-id>/
    python -m umlsmatch free_texts/synthetic/ --profile clinical_recall

Three flags redirect that, and they are mutually exclusive because each is an
answer to the same question -- where does this run's output go?

    --out-dir DIR   run directory under DIR instead, and nothing on stdout
    -o FILE         one merged file, the pre-0.2 behaviour, no run directory
    --no-save       stdout only, nothing written

Saving by default means the CLI writes note text to disk without being asked,
which is why the default root sits under ``out/`` -- already covered by
``.gitignore`` and ``.githooks/phi-paths.pattern``. ``--no-save`` is the opt
out.

Flag values map straight onto :class:`~umlsmatch.analyze.ClinicalPipeline`'s
constructor arguments, so ``--max-scope 4`` and ``max_scope=4`` mean the same
thing; ``--no-max-scope`` is the CLI spelling of ``max_scope=None``. That
contract is the reason every constructor argument has a flag: five of them once
did not, including ``history_sections`` -- the single setting that most changes
what this pipeline reports about a chart -- so the documented mapping held for
everything except the arguments a user would most want to reach.

Start from a profile and adjust::

    python -m umlsmatch note.txt --profile clinical_recall
    python -m umlsmatch note.txt --profile clinical_recall --no-drop-header-mentions
    python -m umlsmatch note.txt --conditional --no-uncertainty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from umlsmatch.analyze import (
    DB_ENV_VAR,
    DEFAULT_PROFILE,
    PROFILES,
    UNSET,
    ClinicalPipeline,
    find_dictionary,
)
from umlsmatch.assertion.attributes import SUBJECT_PATIENT
from umlsmatch.assertion.negation import MAX_SCOPE_TOKENS
from umlsmatch.corpus import iter_text_files, parse_groups, read_text
from umlsmatch.runs import DEFAULT_RUN_ROOT, RunWriter, manifest_stamp


def _flags(a) -> str:
    """The assertion flags worth printing, for the human-readable listing.

    Only *positive* assessments are shown. An attribute that is False and one
    that was never assessed both print nothing here, which is a real loss of
    information -- ``--json`` is where the three-state value is preserved, and
    this column is a scan aid, not a record.

    Every attribute the pipeline can assert appears here. ``CONDITIONAL`` was
    missing while ``--conditional`` did not exist either, so turning the
    prototype on in Python and printing the result rendered it invisible --
    which reads as the rules never firing.
    """
    flags = ["NEGATED"] if a.negated else []
    if a.subject and a.subject != SUBJECT_PATIENT:
        flags.append(a.subject.upper())
    if a.history_of:
        flags.append("HISTORY")
    if a.uncertain:
        flags.append("UNCERTAIN")
    if a.conditional:
        flags.append("CONDITIONAL")
    return " ".join(flags)


def _write(anns, label: str, sink, *, as_json: bool, header: bool) -> None:
    """Render one document's annotations to `sink`.

    Shared by the single-stream sinks (stdout, ``-o``) and the per-document
    files ``--out-dir`` writes, so the two cannot drift into formatting the
    same annotations differently.

    In JSON mode this is one object on one line whatever the destination, which
    is what keeps a run directory concatenable: ``cat run/*.jsonl`` reproduces
    the merged ``-o`` output exactly.
    """
    if as_json:
        print(
            json.dumps(
                {"source": label, "annotations": [a.to_dict() for a in anns]},
                ensure_ascii=False,
            ),
            file=sink,
        )
        return
    # Only when several documents share one stream. A per-document file is
    # already named for its source, so the banner would be noise.
    if header:
        print(f"\n=== {label} ===", file=sink)
    for a in anns:
        print(
            f"{a.start:>6}:{a.end:<6} {a.cui:<10} {a.group:<10} "
            f"{a.text[:40]:<42}{_flags(a)}",
            file=sink,
        )


def _manifest_fields(args, nlp, db: Path) -> dict:
    """Provenance for a run directory's manifest.

    The settings are read back off the constructed pipeline rather than echoed
    from `args`, because those are not the same thing: ``--profile
    clinical_recall`` leaves ``history_sections`` unset on the command line and
    True on the pipeline. Recording the flags would document what was typed;
    recording the attributes documents what actually ran, which is what someone
    comparing two run directories needs.
    """
    return manifest_stamp(
        dictionary=str(db),
        format="jsonl" if args.json else "text",
        filters={
            "groups": sorted(nlp.groups) if nlp.groups else None,
            "negated_only": args.negated_only,
            "affirmed_only": args.affirmed_only,
        },
        settings={
            "profile": nlp.profile,
            "max_scope": nlp.max_scope,
            "coordination": nlp.coordination,
            "clause_bounding": nlp.clause_bounding,
            "sections": nlp.sections,
            "resolve_overlaps": nlp.resolve_overlaps,
            "subject": nlp.subject,
            "history": nlp.history,
            "history_sections": nlp.history_sections,
            "uncertainty": nlp.uncertainty,
            "conditional": nlp.conditional,
            "drop_header_mentions": nlp.drop_header_mentions,
        },
    )


def _gather(paths: list[str]) -> list[tuple[str, str]]:
    """Resolve inputs to (label, text) pairs. '-' means stdin."""
    out: list[tuple[str, str]] = []
    for raw in paths:
        if raw == "-":
            out.append(("<stdin>", sys.stdin.read()))
            continue
        p = Path(raw)
        if p.exists():
            files = list(iter_text_files(p))
            if not files:
                print(f"warning: no text files under {p}", file=sys.stderr)
            out.extend((str(f), read_text(f)) for f in files)
        else:
            print(f"error: no such file or directory: {raw}", file=sys.stderr)
            raise SystemExit(2)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit status.

    `argv` defaults to ``sys.argv[1:]``; it is a parameter so the tests can
    drive the CLI without a subprocess.
    """
    # There are two ways in -- `python -m umlsmatch` and the `umlsmatch`
    # console script the wheel installs -- and usage text naming the other one
    # is the kind of thing a reader copies verbatim and then has to debug.
    # argparse's own default is no good here either: it derives from
    # ``sys.argv[0]``, which is ``__main__.py`` under ``-m``.
    invoked_as = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    ap = argparse.ArgumentParser(
        prog="umlsmatch" if invoked_as == "umlsmatch" else "python -m umlsmatch",
        description="Extract UMLS concepts and negation from clinical text.",
        epilog=(
            f"Dictionary resolution: --db, then ${DB_ENV_VAR}, "
            "then data/umls_sno_rx.sqlite"
        ),
    )
    ap.add_argument("inputs", nargs="+", help="files, directories, or '-' for stdin")
    ap.add_argument("--db", default=None, help="dictionary SQLite path")
    # All three answer the same question -- "where does this run's output go?"
    # -- with incompatible answers, so taking two would have to silently ignore
    # one. Passing none of them is the default: the listing goes to stdout as
    # it always has, *and* a run directory is saved under DEFAULT_RUN_ROOT.
    sink_group = ap.add_mutually_exclusive_group()
    sink_group.add_argument(
        "-o",
        "--out",
        default=None,
        help="write one merged file here instead of stdout. Saves no run "
        "directory -- this flag says where the output goes.",
    )
    sink_group.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help=f"put this run's directory under DIR instead of {DEFAULT_RUN_ROOT}, "
        "and send nothing to stdout.",
    )
    sink_group.add_argument(
        "--no-save",
        action="store_true",
        help="print to stdout and write nothing to disk.",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="name this run's directory instead of generating a timestamped "
        "one. Fails if it already exists.",
    )
    ap.add_argument("--json", action="store_true", help="JSONL output, one object per document")
    ap.add_argument(
        "--groups",
        default=None,
        help="comma-separated semantic groups to keep, e.g. DISORDER,DRUG",
    )
    ap.add_argument("--negated-only", action="store_true", help="only negated concepts")
    ap.add_argument("--affirmed-only", action="store_true", help="only non-negated concepts")
    ap.add_argument(
        "--resolve-overlaps",
        action="store_true",
        help="keep only the longest of overlapping matches (lowers recall)",
    )
    ap.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help=(
            f"named configuration (default: {DEFAULT_PROFILE}). 'clinical_recall' "
            "turns on section-aware history and drops header mentions: adjudicated "
            "history_of F1 0.277 -> 0.704. Individual flags below still win."
        ),
    )

    attrs = ap.add_argument_group(
        "assertion attributes",
        "Turning one off leaves its field null -- 'not assessed', which is a "
        "different claim from 'assessed and absent'. See --json.",
    )
    attrs.add_argument(
        "--no-subject",
        action="store_true",
        help="do not assess whose problem a mention is (patient vs family member)",
    )
    attrs.add_argument(
        "--no-history",
        action="store_true",
        help="do not assess whether a mention sits in a history-taking context",
    )
    attrs.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="do not assess whether a mention is hedged",
    )
    attrs.add_argument(
        "--conditional",
        action="store_true",
        help="assess conditional mentions ('call if you develop chest pain'). "
        "A PROTOTYPE: ConText's HYPOTHETICAL cues, unmeasured -- 27 corpus "
        "positives cannot score them. Off leaves the field null.",
    )

    sect = ap.add_argument_group("section rules")
    sect.add_argument(
        "--history-sections",
        dest="history_sections",
        action="store_true",
        default=None,
        help="treat everything under a Past Medical History header as history, "
        "not only mentions with their own cue. Raises adjudicated recall "
        "0.198 -> 0.670 and lowers cTAKES agreement sharply; see "
        "docs/ADJUDICATION_RESULTS.md. Implied by --profile clinical_recall.",
    )
    sect.add_argument(
        "--no-history-sections",
        dest="history_sections",
        action="store_false",
        help="opt back out of the section rule after --profile clinical_recall",
    )
    sect.add_argument(
        "--drop-header-mentions",
        dest="drop_header_mentions",
        action="store_true",
        default=None,
        help="discard concepts lying inside a section heading (the 'History' in "
        "'Past Medical History:'). Removes 1.6%% of annotations, 88%% of them "
        "chart furniture. Implied by --profile clinical_recall.",
    )
    sect.add_argument(
        "--no-drop-header-mentions",
        dest="drop_header_mentions",
        action="store_false",
        help="keep header mentions after --profile clinical_recall",
    )
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument(
        "--max-scope",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"negation scope cap in tokens (default: {MAX_SCOPE_TOKENS}). "
            "0 means a zero-token scope, i.e. nothing is negated."
        ),
    )
    scope.add_argument(
        "--no-max-scope",
        action="store_true",
        help="remove the cap entirely; a trigger reaches to the terminator or "
        "sentence edge. Over-negates badly -- for diagnosis, not production.",
    )
    ap.add_argument(
        "--no-coordination",
        action="store_true",
        help="stop negation following a coordinate list past the scope cap. "
        "Reproduces the lexical-only numbers; see umlsmatch.assertion.negation.",
    )
    ap.add_argument(
        "--no-clause-bounding",
        action="store_true",
        help="stop negation halting at a coordinated clause ('denies X, and "
        "reports Y'). Lowers measured precision -- for reproduction only.",
    )
    ap.add_argument(
        "--no-sections",
        action="store_true",
        help="ignore Review-of-Systems style headers when scoping negation",
    )
    args = ap.parse_args(argv)

    if args.negated_only and args.affirmed_only:
        ap.error("--negated-only and --affirmed-only are mutually exclusive")
    if args.run_id is not None and (args.no_save or args.out):
        # Naming a run directory that will not be written is a typo, and the
        # alternative -- ignoring it -- loses output the caller expected to
        # find under that name. Without --no-save or -o there is always a run
        # directory to name, default root included, so --run-id stands alone.
        ap.error("--run-id cannot be combined with --no-save or -o/--out")
    if args.max_scope is not None and args.max_scope < 0:
        # A negative cap makes every scope empty, silently disabling negation
        # rather than erroring -- which reads as "negation is broken".
        ap.error("--max-scope must be >= 0 (use --no-max-scope to remove the cap)")

    db = find_dictionary(args.db)
    if not db.is_file():
        print(
            f"error: dictionary not found: {db}\n"
            "Build it first (see docs/UMLS_UPDATE_GUIDE.md):\n"
            "  python -m umlsmatch.build --umls-dir <UMLS META dir>",
            file=sys.stderr,
        )
        return 1

    documents = _gather(args.inputs)
    if not documents:
        print("error: no input documents", file=sys.stderr)
        return 2

    groups = parse_groups(args.groups)

    kwargs = {
        "profile": args.profile,
        "resolve_overlaps": args.resolve_overlaps,
        "groups": groups,
        "coordination": not args.no_coordination,
        "clause_bounding": not args.no_clause_bounding,
        "sections": not args.no_sections,
        "subject": not args.no_subject,
        "history": not args.no_history,
        "uncertainty": not args.no_uncertainty,
        "conditional": args.conditional,
        # `None` from argparse means the flag was not given, so the profile's
        # value stands. Translating that to False here would make every CLI
        # invocation an explicit override and silently defeat --profile.
        "history_sections": (
            UNSET if args.history_sections is None else args.history_sections
        ),
        "drop_header_mentions": (
            UNSET if args.drop_header_mentions is None else args.drop_header_mentions
        ),
    }
    # `--max-scope N` maps straight through to the ClinicalPipeline argument of
    # the same name, including 0 -- which means a zero-token scope, not an
    # uncapped one. Translating 0 to None would have the CLI and the API read
    # the same literal in opposite directions, so uncapped is its own flag.
    if args.no_max_scope:
        kwargs["max_scope"] = None
    elif args.max_scope is not None:
        kwargs["max_scope"] = args.max_scope

    # The stream and the run directory are independent, which is what lets the
    # default do both: a plain invocation prints the listing exactly as it
    # always has *and* keeps a copy on disk. `-o` names a single destination,
    # so it does neither of the others.
    run_root = None if (args.out or args.no_save) else (args.out_dir or DEFAULT_RUN_ROOT)
    run = None
    if run_root is not None:
        try:
            run = RunWriter(
                run_root,
                run_id=args.run_id,
                # Each file holds a single JSON object on one line, so the
                # suffix is still .jsonl rather than .json: concatenating the
                # directory has to yield the same JSONL `-o` would have.
                suffix=".jsonl" if args.json else ".txt",
            )
        except FileExistsError:
            print(
                f"error: run directory already exists: "
                f"{Path(run_root) / args.run_id}\n"
                "Pick another --run-id, or omit it for a generated one.",
                file=sys.stderr,
            )
            return 2
        except OSError as exc:
            print(f"error: cannot create run directory: {exc}", file=sys.stderr)
            return 2

    # Not a `with`: the sink is either a file we own and must close, or
    # sys.stdout which we must NOT close. The try/finally below handles both;
    # a context manager would need an ExitStack to say the same thing.
    # `--out-dir` is the one mode with no stream -- it asked for files only.
    sink = None
    if args.out:
        sink = open(args.out, "w", encoding="utf-8")  # noqa: SIM115
    elif not args.out_dir:
        sink = sys.stdout
    total = 0
    try:
        with ClinicalPipeline(db, **kwargs) as nlp:
            for label, text in documents:
                anns = nlp.analyze(text)
                if args.negated_only:
                    anns = [a for a in anns if a.negated]
                elif args.affirmed_only:
                    anns = [a for a in anns if not a.negated]
                total += len(anns)

                # Both, not either: by default the listing goes to stdout and
                # the same annotations are saved to the run directory.
                if sink is not None:
                    _write(
                        anns,
                        label,
                        sink,
                        as_json=args.json,
                        header=len(documents) > 1,
                    )
                if run is not None:
                    path = run.path_for(label)
                    with open(path, "w", encoding="utf-8") as doc_sink:
                        _write(anns, label, doc_sink, as_json=args.json, header=False)
                    # Recorded even when empty, and the file is written even
                    # when empty: a missing file would make "no concepts here"
                    # indistinguishable from "this note was never processed".
                    run.record(label, path, len(anns))

            # Inside the `with`, so the effective settings are read off a live
            # pipeline. Last, so a run that died mid-corpus leaves no manifest
            # claiming a complete run.
            if run is not None:
                run.write_manifest(**_manifest_fields(args, nlp, db))
    finally:
        if sink is not None:
            # Flush before the stderr summary, or a piped stdout (block-buffered)
            # prints after it and the output reads out of order.
            sink.flush()
            if args.out:
                sink.close()

    if not args.json:
        print(
            f"\n{total:,} annotation(s) across {len(documents)} document(s)",
            file=sys.stderr,
        )
    if run is not None:
        # Printed even under --json, unlike the summary above. The directory
        # name is generated, so a caller that cannot see it cannot find the
        # files that were just written on its behalf.
        print(f"-> {run.directory}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

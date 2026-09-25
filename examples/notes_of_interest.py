#!/usr/bin/env python3
"""Find documents that affirm a clinical concept — simple phenotyping.

    python examples/notes_of_interest.py free_texts/synthetic "diabetes mellitus"
    python examples/notes_of_interest.py free_texts/synthetic --cui C0011860 C0018802
    python examples/notes_of_interest.py free_texts/synthetic "chest pain" --show-negated

**Why this needs NLP and not grep.** A keyword search for "chest pain" matches
"patient denies chest pain" — the exact opposite of what you want. This script
counts a document only when the concept is mentioned *and not negated*, and
`--show-negated` exists so you can see how often that distinction fires.

Concepts are matched by CUI, so morphological and synonym variants
("diabetes mellitus type 2", "T2DM", "DM2") collapse to one concept
automatically. That is the other thing grep cannot do.

**Every search saves itself** to `out/runs/<run-id>/notes_of_interest.sqlite`,
the same way the other scripts here keep their output. The database holds four
tables: `search` (one row -- what was asked, over what corpus, with what
totals), `concepts` (what the phrase resolved to), `documents` (one row per
file scanned, including the ones with no mention) and `console` (the printed
report, verbatim, one row per line).

`documents` carries a row for every file, not only the hits, because "absent"
is a result: a cohort denominator cannot be reconstructed from a list of
matches. And the console text is stored beside the structured tables rather
than instead of them -- the tables are what you query, the transcript is what
you show someone who asks what you actually ran.

`--no-save` turns it off. Nothing is written when the search fails early (an
unresolvable phrase, an empty corpus), so a run directory always means a search
that ran.

Caveat worth keeping in view: negation precision is ~0.50 (see docs/USER_GUIDE.md
§7), so some documents excluded as "negated" were not really negated. For
anything consequential, review the hits rather than trusting the count.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from umlsmatch import ClinicalPipeline
from umlsmatch.analyze import find_dictionary
from umlsmatch.corpus import iter_text_files, read_text
from umlsmatch.runs import DEFAULT_RUN_ROOT, RunWriter, manifest_stamp

ARTIFACT = "notes_of_interest.sqlite"

SCHEMA = """
CREATE TABLE search (
    run_id            TEXT NOT NULL,
    created           TEXT NOT NULL,
    input_dir         TEXT NOT NULL,
    dictionary        TEXT NOT NULL,
    -- Exactly one of these is set: the phrase that was typed, or the CUIs that
    -- were given directly. Keeping both columns records which route was taken,
    -- which changes how the result should be read -- a phrase had to resolve,
    -- and could have resolved to more concepts than intended.
    term              TEXT,
    cuis              TEXT,
    show_negated      INTEGER NOT NULL,
    documents_scanned INTEGER NOT NULL,
    affirmed          INTEGER NOT NULL,
    negated_only      INTEGER NOT NULL,
    absent            INTEGER NOT NULL
);

CREATE TABLE concepts (
    cui            TEXT PRIMARY KEY,
    preferred_text TEXT,
    semantic_group TEXT
);

CREATE TABLE documents (
    source            TEXT PRIMARY KEY,
    -- 'affirmed' | 'negated_only' | 'absent'. Stored rather than derived from
    -- the counts: the rule that a single affirmed mention outweighs any number
    -- of negated ones is this script's editorial decision, and a reader of the
    -- database should not have to rediscover it.
    status            TEXT NOT NULL,
    affirmed_mentions INTEGER NOT NULL,
    negated_mentions  INTEGER NOT NULL
);

CREATE INDEX ix_documents_status ON documents(status);

CREATE TABLE console (
    line_no INTEGER PRIMARY KEY,
    stream  TEXT NOT NULL,   -- 'stdout' | 'stderr'
    text    TEXT NOT NULL
);
"""


class Console:
    """Prints, and remembers what it printed.

    A tee rather than a capture-at-the-end: the concept resolution is printed
    before the corpus is scanned, so anything that only started recording once
    the run directory existed would save a transcript missing its own header.
    """

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __call__(self, text: str = "", *, err: bool = False) -> None:
        stream = "stderr" if err else "stdout"
        print(text, file=sys.stderr if err else sys.stdout)
        # Split so `console` is one row per printed line, which is what makes
        # it readable with a plain SELECT rather than a blob to re-parse.
        self.lines.extend((stream, line) for line in str(text).split("\n"))


def cuis_for_term(db: Path, term: str) -> list[tuple[str, str, str]]:
    """Look up (cui, preferred_text, group) for a search phrase.

    The dictionary stores terms in tokenizer spelling, so a phrase is matched
    against the normalized form -- see docs/UMLS_UPDATE_GUIDE.md on retokenization.
    """
    norm = " ".join(term.casefold().split())
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT DISTINCT c.cui, c.preferred_text, c.best_group "
            "FROM term t JOIN concept c ON c.cui = t.cui "
            "WHERE t.norm = ? ORDER BY c.cui",
            (norm,),
        ).fetchall()
    finally:
        conn.close()


def save(
    path: Path,
    *,
    run_id: str,
    created: str,
    args,
    db: Path,
    concepts: list[tuple[str, str, str]],
    results: list[tuple[str, str, int, int]],
    console: Console,
) -> None:
    """Write the search, its per-document results and the transcript."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        counts = {s: 0 for s in ("affirmed", "negated_only", "absent")}
        for _src, status, _yes, _no in results:
            counts[status] += 1

        conn.execute(
            "INSERT INTO search VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                created,
                str(args.input_dir),
                str(db),
                args.term,
                ",".join(sorted(c.upper() for c in args.cui)) if args.cui else None,
                int(args.show_negated),
                len(results),
                counts["affirmed"],
                counts["negated_only"],
                counts["absent"],
            ),
        )
        conn.executemany("INSERT INTO concepts VALUES (?,?,?)", concepts)
        conn.executemany("INSERT INTO documents VALUES (?,?,?,?)", results)
        conn.executemany(
            "INSERT INTO console (stream, text) VALUES (?,?)", console.lines
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("term", nargs="?", help="concept phrase, e.g. 'diabetes mellitus'")
    ap.add_argument("--cui", nargs="+", default=None, help="search by CUI instead")
    ap.add_argument("--db", default=None)
    ap.add_argument("--show-negated", action="store_true", help="also report negated mentions")
    ap.add_argument("--run-id", default=None, help="name this run's directory")
    ap.add_argument(
        "--no-save",
        action="store_true",
        help=f"print only; do not write {DEFAULT_RUN_ROOT}/<run-id>/{ARTIFACT}",
    )
    args = ap.parse_args()

    if not args.term and not args.cui:
        ap.error("give a search term or --cui")
    if args.run_id and args.no_save:
        ap.error("--run-id has nothing to name with --no-save")

    say = Console()

    db = find_dictionary(args.db)
    if not db.is_file():
        print(f"dictionary not found: {db} (see docs/USER_GUIDE.md §4)", file=sys.stderr)
        return 1

    if args.cui:
        targets = {c.upper() for c in args.cui}
        concepts: list[tuple[str, str, str]] = []
        say(f"Searching for CUIs: {', '.join(sorted(targets))}")
    else:
        hits = cuis_for_term(db, args.term)
        if not hits:
            print(
                f"No concept matches {args.term!r} exactly.\n"
                "Try the concept's canonical name, or search by --cui.",
                file=sys.stderr,
            )
            return 1
        targets = {cui for cui, _, _ in hits}
        # (cui, preferred_text, group) from the lookup; the table stores them
        # in that order, so reorder here rather than in the INSERT.
        concepts = [(cui, pref, group) for cui, pref, group in hits]
        say(f"{args.term!r} resolves to {len(hits)} concept(s):")
        for cui, pref, group in hits:
            say(f"  {cui}  {group:<10} {pref}")

    files = list(iter_text_files(args.input_dir))
    if not files:
        print(f"no .txt files under {args.input_dir}", file=sys.stderr)
        return 1

    affirmed: list[tuple[str, int]] = []
    negated_only: list[str] = []
    # Every file, including the ones with no mention: "absent" is a result, and
    # a denominator cannot be recovered from a list of hits.
    results: list[tuple[str, str, int, int]] = []

    with ClinicalPipeline(db) as nlp:
        for path in files:
            text = read_text(path)
            found = [a for a in nlp.analyze(text) if a.cui in targets]
            yes = sum(1 for a in found if not a.negated)
            no = sum(1 for a in found if a.negated)
            if yes:
                status = "affirmed"
                affirmed.append((path.name, yes))
            elif no:
                status = "negated_only"
                negated_only.append(path.name)
            else:
                status = "absent"
            results.append((str(path), status, yes, no))

    say(f"\nScanned {len(files)} documents.")
    say(f"  {len(affirmed)} affirm the concept")
    say(f"  {len(negated_only)} mention it only as negated")
    say(f"  {len(files) - len(affirmed) - len(negated_only)} do not mention it\n")

    if affirmed:
        say("Affirmed in:")
        for name, n in sorted(affirmed, key=lambda kv: -kv[1]):
            say(f"  {name:<52} {n} mention(s)")

    if args.show_negated and negated_only:
        say("\nNegated-only (excluded from the cohort):")
        for name in negated_only:
            say(f"  {name}")
    elif negated_only:
        say(f"\n({len(negated_only)} negated-only documents hidden; --show-negated to list)")

    if args.no_save:
        return 0

    # Created only now: an early return above means the search never ran, and a
    # run directory should not exist for one that did not.
    try:
        run = RunWriter(DEFAULT_RUN_ROOT, run_id=args.run_id)
    except FileExistsError:
        print(f"run directory already exists: {args.run_id}", file=sys.stderr)
        return 2

    stamp = manifest_stamp(
        script=Path(__file__).name,
        input_dir=str(args.input_dir),
        dictionary=str(db),
        query={
            "term": args.term,
            "cuis": sorted(targets),
            "show_negated": args.show_negated,
        },
    )
    path = run.artifact(ARTIFACT)
    save(
        path,
        run_id=run.run_id,
        created=stamp["created"],
        args=args,
        db=db,
        concepts=concepts,
        results=results,
        console=say,
    )
    run.record_artifact(
        path,
        documents=len(results),
        affirmed=len(affirmed),
        negated_only=len(negated_only),
        console_lines=len(say.lines),
    )
    run.write_manifest(
        **stamp,
        totals={
            "documents": len(results),
            "affirmed": len(affirmed),
            "negated_only": len(negated_only),
        },
    )

    # To stderr, so piping the report to a file still shows where it was saved
    # -- but flush stdout first, or a redirected (block-buffered) report lands
    # after this line and the output reads out of order. Same reasoning as
    # umlsmatch.__main__.
    sys.stdout.flush()
    print(f"\n-> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

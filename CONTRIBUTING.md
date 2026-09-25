# Contributing

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev,nlp]"
python -m spacy download en_core_web_sm   # not an extra: spaCy models aren't on PyPI
git config core.hooksPath .githooks
```

That last line is not optional housekeeping — see below.

## The one rule that matters: no PHI in git

The corpus under `free_texts/` is **real clinical notes**, and everything
derived from it carries either verbatim note text or patient/note identifiers:

| path | what leaks |
|---|---|
| `free_texts/notes/*.csv` | the notes themselves |
| `free_texts/json/silver.jsonl` | Java cTAKES output, includes the full `sofaString` |
| `free_texts/xmi_out/*.xmi` | derived annotations; patient/note ids in the filenames |
| `free_texts/adjudication/**` | review files quote whole sentences; the sidecars carry offsets and verdicts |
| `out/*.jsonl`, `out/*.csv`, `out/*.db` | matched spans quoted verbatim |
| `*.sqlite` | dictionaries (large) and annotation stores (PHI) |

`free_texts/synthetic/*.txt` is the one exception and **is** committed. Those
notes are generated — no patient behind any of them — and they are in git so
the examples and corpus tests have something to run against on a fresh clone.
Nothing else in that directory is exempt: a `.jsonl` or `.csv` written there
comes from the same pipeline that runs over the real corpus and stays blocked.

Three layers protect this, and all three are deliberate:

1. **`.gitignore`** excludes those paths. This stops an accident, not intent.
2. **`.githooks/pre-commit`** refuses any staged path matching the pattern in
   **`.githooks/phi-paths.pattern`** — because `git add -f` overrides
   `.gitignore` and a newly-invented output directory would not be covered by
   it at all. Enable it once per clone:
   `git config core.hooksPath .githooks`.
3. **`.github/workflows/ci.yml`** runs the same pattern over the tree *and* the
   full history on every push, which is what makes the gate non-optional — the
   hook only protects people who enabled it, and `--no-verify` skips it.

**The pattern is defined once**, in `.githooks/phi-paths.pattern`, and both
gates read it from there. Two copies drift, and a drifted backstop is worse
than none: the hook and CI are meant to be independent checks of the same rule,
not two rules. `tests/test_phi_guard.py` fails if either gate stops reading the
shared file, if the pattern stops matching the corpus layout, or if it starts
matching ordinary project files. **Add new rules there**, not inline.

**Exceptions are defined once too**, in `.githooks/phi-allow.pattern`: a path
is refused when it matches the block pattern *and* does not match the allow
pattern. Treat that file with more suspicion than the other one. Widening the
block pattern is safe and announces itself by refusing something; widening the
carve-out fails open and announces nothing, because a gate that has stopped
refusing looks exactly like a repository with nothing to refuse. Both gates
skip the carve-out entirely when the file is missing or empty — `grep -Eiv ""`
matches every line and would invert the check into refusing nothing — so
deleting it restores the strict gate rather than removing it.

`--no-verify` skips the hook. Do not use it to get past the PHI check: every
other failure here is fixable in the next commit, and that one is not — once
note text is in history, removing it means rewriting history. CI will fail the
push in any case.

The same applies to **test fixtures and documentation**. Use synthetic
narrative, never a paste from a real note. `tests/test_spacy_tokenizer.py` and
`tests/test_csv_notes_to_txt.py` show the shape to imitate — realistic clinical
phrasing and id formats, invented content.

## Checks

The hook runs these; run them directly while iterating:

```bash
ruff check . --exclude .venv        # --fix handles most
pytest -q
```

The ruff rule set is pinned explicitly in `pyproject.toml` rather than left to
ruff's defaults, which change between releases. If a rule fires on something
deliberate, suppress it with a targeted `# noqa: RULE` **and a comment saying
why** — `dictionary/exclusions.py` and `dictionary/matcher.py` have worked
examples, both cases where matching the Java source line-for-line beats
satisfying the linter.

Tests that need the built dictionary or the spaCy model skip themselves when
those are absent, so a fresh clone runs green without a 589 MB download.

## Releasing

Don't `twine upload`. Releases come from CI: bump `__version__` in
`src/umlsmatch/__init__.py`, commit, `git tag v0.1.1 && git push origin v0.1.1`.
The tag runs the test matrix and the PHI gate, then publishes with a
short-lived OIDC credential — there is no PyPI token in this repository, and
uploading by hand bypasses every check. Full procedure, including what to do
when a release fails or ships broken, is in [docs/RELEASING.md](docs/RELEASING.md).

## Generated files

`src/umlsmatch/umls/semantic_tui.py` and `src/umlsmatch/umls/ctakes_tuis.py` are
generated from the cTAKES Java source and carry `DO NOT EDIT BY HAND`. Fix the
generator (`tools/transcode_semantic_tui.py`, `tools/extract_ctakes_tuis.py`)
and re-run it; an edit made in place is reverted the next time anyone does.
They have per-file lint ignores in `pyproject.toml` for the same reason.

## Measurement claims

This codebase states numbers in docstrings. If you change something those
numbers describe, re-run the relevant scorer and update them —
stale claims in source are a real defect here, because the docstrings are what a
reader meets first.

```bash
python tools/score_parity.py         --jsonl free_texts/json/silver.jsonl --quiet
python tools/score_negation_spans.py --jsonl free_texts/json/silver.jsonl --quiet
python tools/score_boundaries.py     --jsonl free_texts/json/silver.jsonl --quiet
python tools/score_pos.py            --jsonl free_texts/json/silver.jsonl --quiet

# every assertion attribute the pipeline assesses, with its caveat attached
python tools/score_attributes.py --jsonl free_texts/json/silver.jsonl \
    --db data/umls_ctakes_16ab.sqlite --attribute all

# and why each disagreement happened, by rule and by cue phrase
python tools/diff_attributes.py --jsonl free_texts/json/silver.jsonl \
    --db data/umls_ctakes_16ab.sqlite --attribute subject
```

Three more diff tools exist for narrower questions, and they are worth knowing
about before writing a one-off script that answers the same thing:

| tool | question it answers |
|---|---|
| `tools/diff_parity.py` | which CUIs we emit that Java does not, and vice versa, with example text and POS |
| `tools/diff_negation.py` | negation disagreements per CUI, the per-document view `diff_attributes.py` generalizes |
| `tools/diff_negspacy.py` | how our negation differs from negspaCy's ConText, on the same notes — an outside opinion rather than a reference |

Run `diff_attributes.py` **before** changing a lexicon, not after. Every useful
change to the assertion rules so far came out of its per-rule table and none
came from staring at an aggregate F1: it is what showed that a forward window
from a kinship term was reaching into the next row of an EHR family-history
table, and that `history_of`'s section rule was running at precision 0.116.

**The assertion attributes are the exception: do not trust the cTAKES-based
scorers for them.** cTAKES marks Review-of-Systems negatives as affirmed,
labels the same family-history table both ways across notes, and gives 113
distinct mention texts both `historyOf` values. Those numbers measure agreement
with a known-wrong oracle. Score against human verdicts instead:

```bash
python tools/make_adjudication_set.py --jsonl free_texts/json/silver.jsonl \
    --db data/umls_ctakes_16ab.sqlite --attribute subject
# writes free_texts/adjudication/subject/{review.csv,design.json}
# fill the `verdict` column with the words the tool prints
python tools/score_adjudicated.py --review free_texts/adjudication/subject/review.csv
```

The review file is deliberately **blind** — it does not show what any system
predicted, because seeing a machine label anchors the judgement it exists to
overrule. Do not "helpfully" add those columns back. It quotes whole sentences
of note text, so it is PHI and the hook refuses to commit `*.csv`.

**Record what did not work, not only what did.** Several obvious improvements
measure worse: bullets as scope terminators for `subject`, the Past Medical
History section rule for `history_of`, suppressing "history of present illness"
as a pseudo-cue. Each is documented where someone would next have the idea,
because a negative result that is written down is worth as much as the change
that shipped — and saves the next person the measurement.

Always say which dictionary a number came from. Parity against cTAKES' own
shipped 2016AB build (`data/umls_ctakes_16ab.sqlite`) measures the *algorithm*;
against a modern UMLS build it measures the *dictionary*. The two differ by
about 0.2 F1 and quoting one as the other is the easiest mistake to make here.

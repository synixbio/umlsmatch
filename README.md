# umlsmatch

A Python-native clinical NLP pipeline for clinical concept extraction. Extracts
UMLS concepts from clinical text and assesses four assertion attributes —
`negated`, `subject`, `history_of` and `uncertain` — with no JVM and no UIMA.

```python
from umlsmatch import ClinicalPipeline

with ClinicalPipeline() as nlp:
    for a in nlp.analyze("Patient denies chest pain. History of CHF."):
        print(a.cui, a.text, a.group, a.negated, a.subject, a.history_of)
```

```
C0817096 chest ANATOMY True patient False
C0008031 chest pain FINDING True patient False
C0030193 pain FINDING True patient False
C0262926 History FINDING False patient False
C0262926 History of FINDING False patient False
C0009714 CHF DISORDER False patient True
C0018802 CHF DISORDER False patient True
```

Overlapping matches are deliberate, not noise: `chest`, `chest pain` and
`pain` are all emitted, and two CUIs share the `CHF` span. cTAKES does the
same, and suppressing them was measured to cost more recall than it buys
precision. Filter with `groups=` and `negated`.

**An attribute that was not assessed is `None`, never `False`.** `generic` is
deliberately never assessed and `conditional` is off unless asked for (see
Scope), so both are `None` on every annotation a default pipeline produces.
`False` would claim an assessment nobody made, and in a clinical record that is
the expensive direction to be wrong in. Branch on `None`; do not coerce it.

## Validation

Concept extraction is validated against Apache cTAKES as a reference
implementation: the 20 synthetic clinical notes in `free_texts/synthetic/`,
processed by real Java cTAKES (`DefaultFastPipeline`), supply the comparison
set. cTAKES is the reference here, not the goal — and below, where the reference
is demonstrably wrong, it is scored as such.

**Always read an agreement number together with the dictionary that produced
it** — the spread between dictionaries is larger than any algorithmic
difference, and quoting one as the other is the easiest mistake to make here.

| | dictionary | result |
|---|---|---:|
| **Concept extraction (algorithm agreement)** | cTAKES' own shipped 2016AB | **F1 0.971** (P 0.982 / R 0.960) |
| Concept extraction (modern UMLS) | 2026AA, cTAKES' 7 synonym sources | F1 0.782 |
| Concept extraction (modern UMLS) | 2026AA, all synonym sources *(default)* | F1 0.756 (R 0.840) |
| Token boundaries | — | F1 0.953 |
| POS anchor agreement | — | 0.974 |

The first row is the one that says whether the extraction algorithm is correct:
given the same dictionary Java used, the matcher reproduces it at F1 0.971. The
modern-UMLS rows differ because of dictionary *content*, not matcher logic.

All rows are measured on the 20 synthetic notes in `free_texts/synthetic/`,
which ship with the repository — so every figure here is reproducible on a fresh
clone. Assertion figures measured against a larger corpus of real notes differ
substantially; [docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md)
says which numbers this corpus can and cannot carry.

### Targets

The acceptance criteria each component is measured against. Where a target has
been measured *not* to predict downstream quality it is marked a diagnostic and
says so, rather than being quietly dropped.

| component | target | status |
|---|---:|---|
| Concept extraction, cTAKES' own 2016AB dictionary | ≥0.90 | met — F1 0.971 |
| Token boundaries | ≥0.95 | met — F1 0.953 |
| Sentence boundaries | ≥0.90 | **a diagnostic, not a gate** — F1 0.533; substituting cTAKES' exact sentence spans barely moves CUI agreement or negation |
| POS anchor agreement | — | diagnostic; gates the concept rows |
| `negated` vs. cTAKES | ≥0.90 | not met against a faulty oracle — F1 0.678; see the caveat below |
| `history_of` vs. cTAKES | ≥0.80 | not met, and ships saying so — F1 0.667 on 17 positives |
| `subject` (family member) vs. cTAKES | ≥0.80 | **not measurable here** — 2 reference positives; reaches F1 0.809 on a real-note corpus |
| `uncertain` vs. cTAKES | — | measures the reference, not the rules; zero overlap with cTAKES here |
| Service throughput | your own | measure on your own notes with `tools/load_test.py` |

The attribute rows are not comparable to the concept rows: they are agreement
with cTAKES' own labels, and cTAKES is measurably wrong on constructions common
in these notes. That argument is directly below.

### Assertion attributes

Per-mention agreement with cTAKES' own labels, 20 notes, 1,708 aligned mentions
(`tools/score_attributes.py`). **Every figure in the middle column is agreement
with a faulty oracle, not accuracy** — the adjudicated column is the one to read.

| attribute | aligned positives | vs cTAKES (P / R / F1) | adjudicated F1 | read it as |
|---|---:|---|---:|---|
| `negated` | 106 | 0.520 / 0.972 / 0.678 | **0.870** | the one usable result |
| `history_of` | 17 | 0.500 / 1.000 / 0.667 | **0.277** | low recall — see the warning below |
| `conditional` | 3 | prototype, off by default | — | **not a result** |
| `generic` | 7 | not implemented | — | — |

**`subject` and `uncertain` are left out of that table because neither is
measurable on this corpus** — 2 and 15 reference positives, and disagreement
strata of 10 and 25 cases, which cannot carry a figure in either direction. Both
still run and still return values; what is missing is evidence, not rules. The
counts, and what the same rules measured before this corpus, are in
[docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md).

**The reference is wrong often enough to matter.** cTAKES marks
Review-of-Systems negatives (`"Negative for chills, fever, night sweats..."`) as
*affirmed*, which this pipeline correctly negates and is then scored against. It
gives the same mention text opposite labels across notes — 6 distinct texts get
both `historyOf` values here, `illness` 9 yes against 2 no. And what it marks
`uncertainty` are confirmed findings and an ordered test rather than hedges:
`pain`, `consolidation`, `screening mammogram`. cTAKES is a sound benchmark for
concept extraction and **not** for assertion, which is why the adjudicated
column exists. `tools/make_adjudication_set.py --attribute <name>` builds the
blind review file.

Those verdicts are a model pre-annotation rather than a clinician's, and the
blind review files are untouched and still want a human.
[docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md) carries the method
and the sampling fragility behind every recall number.

> **`history_of` does not meet its ≥0.80 target and ships saying so.** In the
> default configuration it is a low-recall instrument: adjudicated P 0.460 /
> R 0.198, so it finds roughly a fifth of the history in the chart. Most of what
> it misses is bare entries under a `PAST MEDICAL HISTORY` heading carrying no
> per-item cue — which cTAKES misses too, so agreement cannot see it. **Do not
> use the default for problem-list compilation or cohort selection** without
> knowing that. The `clinical_recall` profile below raises recall to 0.670, but
> read the circularity warning in
> [docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md) first: the
> verdicts rewarding it share a convention with the rule it enables.

`conditional` is a prototype behind `ClinicalPipeline(conditional=True)` — 3
reference positives, 4 predictions, zero overlap. That it fires at a sane volume
is the only question a corpus this size answers; it is promoted on adjudicated
evidence or deleted, and
[conditional.py](https://github.com/synixbio/umlsmatch/blob/master/src/umlsmatch/assertion/conditional.py) holds the reasoning and
the known failures. Left off, the attribute stays `None`.

### Negation scope

Negation scope uses the dependency parse as well as the token window, so a
trigger follows a coordinate list to its end (`"denies chest pain, shortness
of breath, or fever"` negates *fever*, which a fixed window misses) and stops
at a coordinated clause (`"Patient denies chest pain, and reports pneumonia
and diabetes mellitus."` negates neither the pneumonia nor the diabetes). The
second of those raises measured precision; the first lowers it against cTAKES
while raising recall, and is on by default as a deliberate judgement rather
than a measurement — `coordination=False` prefers agreement with cTAKES
instead. The reasoning is in
[negation.py](https://github.com/synixbio/umlsmatch/blob/master/src/umlsmatch/assertion/negation.py).

Quote that second example in full. Shortened to `"...and reports pneumonia."`
spaCy tags `reports` as a noun, the coordination disappears, and the sentence
then demonstrates the opposite of the rule.

## Scope

Implemented: sentence splitting, tokenization, POS tagging, UMLS dictionary
lookup, semantic grouping, and four assertion attributes — `negated`,
`subject`, `history_of` and `uncertain`. All four are rules over a shared
lexicon-and-scope module ([scope.py](https://github.com/synixbio/umlsmatch/blob/master/src/umlsmatch/assertion/scope.py)) and run
on the standard library alone; the zero-dependency core is a rule here, not a
preference, and a classifier is not the escalation path.

**Deliberately not implemented: `generic`.** Not "not yet" — the corpus carries
7 positives, which cannot distinguish a working rule set from a broken one in
either direction, and unlike the attributes above there is no external cue
lexicon to adopt: `generic` is a discourse judgement, which is why cTAKES uses a
trained classifier for it. Shipping an unmeasurable attribute into a clinical
record is worse than shipping nothing, because a `False` nobody can audit reads
as an assessment. It stays `None`.

**A prototype, off by default: `conditional`.** Same too-few-positives problem
(3 of them), one
crucial difference — ConText's HYPOTHETICAL category is an external cue list to
adopt, the same standing `uncertain`'s lexicon has. So the rules exist and the
switch is off: `ClinicalPipeline(conditional=True)`. The `None` contract above
holds by default. What makes this different from shipping it anyway is that a
stratified adjudication samples *this pipeline's* positives, so precision is
estimable however few cTAKES found. That is the method that, on an earlier
real-note corpus, separated `uncertain`'s F1 0.162 against cTAKES from its
P 0.724 against verdicts. Until it runs here, nothing is claimed.

Also not implemented: relation extraction, temporal reasoning, coreference.
These are a different problem class — a relation is a label on an *ordered
pair* of mentions, so the candidate space is quadratic and the silver standard
does not carry them at all. §9 of the development plan explains why starting
them speculatively is the wrong move and what order they would go in.

## Install

```bash
pip install umlsmatch[nlp]               # pipeline (spaCy)
python -m spacy download en_core_web_sm  # POS model — a separate step, see below
```

From a clone, for development:

```bash
pip install -e ".[nlp]"          # pipeline (spaCy)
pip install -e ".[dev]"          # ruff + pytest
pip install -e ".[service]"      # FastAPI service
```

**The `en_core_web_sm` download is always its own step.** spaCy models are not
published on PyPI, so no extra can depend on one — the only way to name it in
metadata is a direct URL, which PyPI rejects outright. Installing `[nlp]`
therefore gives you a working spaCy and no model; the pipeline raises with the
`spacy download` command in the message when it hits that.

The base install is **stdlib-only on purpose**: the dictionary builder and the
matcher have no third-party dependencies, so the build tools stay usable in
environments where spaCy will not install. `ClinicalPipeline` raises an
`ImportError` naming the missing extra rather than failing obscurely.

You also need a dictionary — it is not shipped (UMLS licensing, and it is
~590 MB). Building one needs a
[UMLS Metathesaurus licence](https://uts.nlm.nih.gov/uts/signup-login):

```bash
python -m umlsmatch.build --umls-dir <UMLS_META_DIR> --ctakes-root ../ctakes-java
```

See [docs/UMLS_UPDATE_GUIDE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/UMLS_UPDATE_GUIDE.md). If you have a cTAKES install,
`tools/import_ctakes_dictionary.py` imports its shipped dictionary in ~14 s and
is the build the 0.971 figure above was measured with.

## Use it

```bash
# CLI -- prints to stdout, and saves this run under out/runs/<run-id>/
python -m umlsmatch note.txt
python -m umlsmatch notes_dir/ --json --groups DISORDER,DRUG
echo "Patient denies chest pain." | python -m umlsmatch -

python -m umlsmatch notes_dir/ --no-save          # stdout only
python -m umlsmatch notes_dir/ --json -o out.jsonl  # one merged file

# HTTP service
uvicorn umlsmatch.service:app --port 8000
curl -s localhost:8000/analyze -H 'content-type: application/json' \
     -d '{"text": "Patient denies chest pain."}'
```

[docs/USER_GUIDE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/USER_GUIDE.md) covers the Python API;
[examples/](https://github.com/synixbio/umlsmatch/tree/master/examples/) has nine runnable scripts from quickstart through
parallel batch processing and SQLite loading; [docs/SERVICE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/SERVICE.md) covers
deployment.

### Every run saves its own annotation files

You do not have to ask. Alongside the listing on stdout, each run writes a
directory under `out/runs/`:

```
out/runs/
├── 20260921T071455Z-3f9a1c/
│   ├── run_manifest.json
│   ├── doc_01.jsonl
│   ├── doc_02.jsonl
│   └── ...                  (one file per input note)
└── 20260921T072310Z-b1e0d4/
    └── ...
```

Nothing is ever overwritten, so two profiles can be run back to back and
diffed. `run_manifest.json` records the **effective** settings — the ones the
pipeline actually ran with, not the flags you typed, which differ whenever
`--profile` supplied a value — alongside the dictionary, the version, and a
per-document table of source path, output file and annotation count. Add
`--run-id NAME` to name a run instead of taking the generated timestamp; it
fails rather than write into an existing directory.

Three flags redirect that. They are mutually exclusive, because each answers
the same question — where does this run's output go?

| flag | what you get |
|---|---|
| *(none)* | stdout **and** a run directory under `out/runs/` |
| `--out-dir DIR` | a run directory under `DIR`, nothing on stdout |
| `-o FILE` | one merged file, no run directory |
| `--no-save` | stdout only, nothing written |

Each JSON file holds one object on one line, so `cat out/runs/<id>/*.jsonl`
reproduces byte-for-byte what `-o` would have written. A note with no concepts
still gets a file: absence would not distinguish "none found" from "never
processed".

> **These files are PHI**, the same as any other output here — annotation
> records quote note text verbatim, and `run_manifest.json` lists source paths,
> which in a real corpus are patient identifiers. Saving by default means the
> CLI writes note text to disk without being asked, which is exactly why the
> default root sits under `out/`: already covered by `.gitignore` and the
> pre-commit gate. `--no-save` is the opt out.

### Two profiles

One configuration question here has two defensible answers, so it has a name
rather than a keyword argument you had to already know about.

| profile | what it is for | sets |
|---|---|---|
| `strict` *(default)* | reproducing the figures above | `history_sections=False`, `drop_header_mentions=False` |
| `clinical_recall` | answering what is in the chart | `history_sections=True`, `drop_header_mentions=True` |

```python
ClinicalPipeline(profile="clinical_recall")
```
```bash
python -m umlsmatch note.txt --profile clinical_recall
UMLSMATCH_PROFILE=clinical_recall uvicorn umlsmatch.service:app
```

`clinical_recall` moves adjudicated `history_of` from F1 0.277 to 0.704 (recall
0.198 → 0.670) and raises precision too, and drops concepts that fall inside a
section heading — 1.9% of annotations on this corpus, at a cost of 0.008
concept-extraction F1 (0.971 → 0.963), so header-dropping is **not** free here
as it is on real notes.

The default stays `strict`, and on this corpus the reason is stronger rather
than weaker: the verdicts rewarding `history_sections=True` are a model's
pre-annotation whose stated convention — entries under `PAST MEDICAL HISTORY`
are history — is the very rule the switch implements, so most of that F1 gain is
circular. The reasoning is in
[docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md). A profile supplies
defaults only — any argument you pass explicitly still wins. `/info` reports
which profile a running service is using.

## Working with PHI

The corpus this was developed against is real clinical notes. Nothing derived
from it is in version control, and a pre-commit hook refuses to add any. If you
point this at real notes, the same applies to your outputs — annotation records
quote note text verbatim. See [CONTRIBUTING.md](https://github.com/synixbio/umlsmatch/blob/master/CONTRIBUTING.md).

The service is designed to run egress-blocked; it makes no outbound network
calls.

## Documentation

| file | what it is |
|---|---|
| [docs/USER_GUIDE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/USER_GUIDE.md) | Python API reference and recipes |
| [docs/SERVICE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/SERVICE.md) | HTTP service and deployment |
| [docs/ADJUDICATION_RESULTS.md](https://github.com/synixbio/umlsmatch/blob/master/docs/ADJUDICATION_RESULTS.md) | the assertion attributes scored against verdicts instead of cTAKES |
| [docs/UMLS_UPDATE_GUIDE.md](https://github.com/synixbio/umlsmatch/blob/master/docs/UMLS_UPDATE_GUIDE.md) | rebuilding against a newer UMLS release |
| [docs/RELEASING.md](https://github.com/synixbio/umlsmatch/blob/master/docs/RELEASING.md) | cutting a release, and what to do when one goes wrong |
| [CONTRIBUTING.md](https://github.com/synixbio/umlsmatch/blob/master/CONTRIBUTING.md) | setup, PHI rules, conventions |

## Licence

Apache 2.0 — full text in [LICENSE](https://github.com/synixbio/umlsmatch/blob/master/LICENSE) — matching Apache cTAKES, from
which the semantic-type tables, the exclusion lists and the rare-word matching
algorithm are derived. [NOTICE](https://github.com/synixbio/umlsmatch/blob/master/NOTICE) carries the attribution that licence
requires, and lists file by file what is derived from where.

**UMLS content is licensed separately by the NLM and is not distributed here.**
No UMLS data is in this repository or in any wheel built from it; the
dictionary under `data/` is generated locally and gitignored. Building one
requires your own UMLS Metathesaurus License and a UTS account, which oblige
you to respect the constituent vocabularies' copyrights and to file a brief
annual usage report. SNOMED CT has additional affiliate terms outside the US.

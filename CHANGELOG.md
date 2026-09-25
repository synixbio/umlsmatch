# Changelog

Notable changes to umlsmatch. Format follows [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [semantic
versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **The corpus export examples are no longer in the repository.**
  `parse_to_csv.py`, `parse_to_jsonl.py`, `parse_to_jsonl_batch.py`,
  `parse_to_parquet.py`, `parse_to_sqlite.py` and `compare_exports.py` are now
  local-only and gitignored. None was part of the installed package. Export with
  `python -m umlsmatch --json -o FILE` and load that with
  `examples/load_to_sqlite.py --basename`; `examples/README.md` walks through it.

## [0.2.0] — 2026-09-25

### Added

- **`ClinicalPipeline.assess(text, spans)`** runs this pipeline's sentence
  splitting, parse, section tracking and assertion rules over concept spans
  found by another extractor, given as `(start, end, cui)`. It returns one
  `Annotation` per span in input order, or `None` for a span that crosses a
  sentence boundary or covers no token. `analyze()` and `assess()` now share one
  private method, `_assess_window`, so a span both see gets the same attributes
  from either; a test pins that. For pairing these rules with MetaMapLite's
  concepts (cuiflow's `ensemble:hybrid_staged`).

## [0.1.0] — 2026-09-22

Initial release.

### Added

#### Core

- **Clinical concept extraction.** Extracts UMLS concepts (SNOMED CT + RxNorm)
  from clinical text with CUIs, semantic groups and character offsets, using a
  rare-word index rather than an automaton. No JVM, no UIMA.
- **Four assertion attributes** — `negated`, `subject`, `history_of` and
  `uncertain` — each with its own constructor switch. An attribute that was not
  assessed reports `None`, never `False`: "we did not look" and "we looked and
  found nothing" are different claims, and the type keeps them different.
- **Two named profiles**, `strict` (default) and `clinical_recall`, reachable
  from Python, the `--profile` CLI flag and `UMLSMATCH_PROFILE`. `strict` is the
  configuration under which every published figure was measured; `clinical_recall`
  turns on section-aware history and drops header mentions, which more than
  raises adjudicated `history_of` recall (.198 → .670).
- **A FastAPI service** with a pipeline pool, `/analyze`, `/analyze/batch`,
  `/health`, `/ready`, `/info` and Prometheus metrics. See
  [docs/SERVICE.md](docs/SERVICE.md).
- **A dictionary builder** (`python -m umlsmatch.build`) and a CLI, reachable
  either as the installed `umlsmatch` command or as `python -m umlsmatch`.
  Usage text names whichever one was used to invoke it.
- **A validation harness** — `src/umlsmatch/eval/`, `tools/score_*.py`,
  `tools/diff_*.py` — and a comparison corpus.

#### Run output

- **Every CLI run saves its own annotation files, without being asked.** Each
  run writes a timestamped directory under `out/runs/` holding one annotation
  file per input note plus a `run_manifest.json` recording the **effective**
  pipeline settings — read off the constructed pipeline, not echoed from
  `argv`, so a value `--profile` supplied is recorded as what actually ran.
  `--run-id NAME` names a run instead of generating an id, and fails rather
  than write into an existing directory. JSON files hold one object on one
  line, so `cat out/runs/<id>/*.jsonl` reproduces `-o` byte-for-byte.
- `-o` merges every document into a single file. `--out-dir DIR` puts the run
  directory somewhere other than `out/runs/` and sends nothing to stdout.
  `--no-save` sends output to stdout only and writes nothing. With `-o` these
  are mutually exclusive: each answers the same question about where a run's
  output goes.
- The default root is under `out/` rather than a new top-level directory
  precisely because saving unasked writes note text to disk, and `out/` is
  already covered by `.gitignore` and the pre-commit PHI gate.
- Every run reports documents, empty and failed counts, and records them in its
  manifest, so nothing is dropped silently.
- `umlsmatch.runs` provides `RunWriter.artifact()` / `record_artifact()` for
  runs whose output is one merged file rather than a file per note, and
  `manifest_stamp()` so the CLI and the examples cannot drift into stamping
  runs differently.
- `umlsmatch.runs.find_artifacts()` returns matching files newest-first (the
  walk order auto-selection needs); `latest_artifact()` is its head, so the two
  cannot disagree about what "newest" means. `umlsmatch.runs.SQLITE_SUFFIXES`
  is the single answer to "which files count as a database" — the same
  reasoning as `umlsmatch.corpus.TEXT_SUFFIXES`, and the reason neither script
  carries its own copy.

#### Exporters

- **Five exporters in `examples/`, all writing the same rows.**
  `parse_to_csv.py`, `parse_to_jsonl.py`, `parse_to_parquet.py`,
  `parse_to_sqlite.py` and `parse_to_jsonl_batch.py` hold to four rules, so
  exports of one corpus in two formats carry identical rows and join on
  identical keys — verified across all five on the shipped corpus, 2,188 rows
  equal on all 14 fields:
  - A document is keyed by its **file name**
    (`umlsmatch.corpus.document_label`). All five warn when two notes in
    different subdirectories share a name, rather than merging them silently.
  - A note the pipeline finds nothing in is **not written** — the tabular
    formats have no way to represent one, so the formats that do are kept in
    step with them.
  - A note that fails analysis is reported and **skipped**, never aborting the
    run.
  - **An attribute the pipeline never assesses gets no column or key.** A
    default pipeline leaves `conditional` and `generic` unassessed, and a column
    of empty cells says nothing about a mention, only about the pipeline.
    Columns come from `ClinicalPipeline.assessed_attributes`, so building with
    `conditional=True` restores that column with no edit to any column list, and
    each run prints what it dropped. `--all-attributes` keeps every column, for
    a downstream schema that must not change shape between runs — at a cost of
    4.15 MB against 3.54 MB for the default JSONL export of a real-note corpus.
- `term` — the dictionary entry that matched — is carried by all five, and
  populates `annotations.term` when a CSV is loaded by `load_to_sqlite.py`.
- **`examples/parse_to_parquet.py`: the CSV table as Parquet.** Typed, columnar
  and compressed — that corpus is 1.7 MB as CSV and 214 kB as Parquet,
  and a column-pruned read of it is ~7× faster. Unassessed assertion attributes
  are stored as real nulls rather than the CSV's empty cell, so "not assessed"
  and "assessed false" stay distinct. The run's provenance stamp is embedded in
  the file's own key-value metadata as well as written to the manifest beside
  it. Needs `pyarrow`, the only example with a dependency of its own.
- **`examples/parse_to_sqlite.py`: analyze straight into a database.** The
  one-step version of analyze → export → load, for when the intermediate export
  is not itself wanted. It imports `SCHEMA`, `INDEXES` and `Loader` from
  `load_to_sqlite.py` rather than restating them, so both routes end at the same
  database — verified by building the corpus both ways and diffing all 16,095
  rows. With no output path it writes `annotations.db` into the run directory,
  which `sqlite_browser.py` then opens with no argument.
- `load_to_sqlite.py` defines its table with `ANNOTATION_COLUMNS` column by
  column, and `schema(omit)` / `indexes(omit)` generate both shapes from it, so
  the narrow table is the wide one minus columns rather than a second schema
  that can drift. This keeps it compatible with `parse_to_sqlite.py`, whose
  other input is a Java cTAKES silver standard that assesses all six attributes.
  `Loader` reads its column list off the table it was given, so `--append` works
  in either direction, and reports how many values it had to drop when the
  target has no column for them. Its `--basename` flag stays, since a silver
  standard and `python -m umlsmatch --json` carry paths rather than file names.
- **`examples/compare_exports.py`: check that the exporters agree.** Normalizes
  a CSV, JSONL, Parquet and SQLite export into one shape — undoing the SQLite
  column renames, collapsing each format's spelling of an optional boolean —
  and compares the rows exactly, reporting what each format costs in bytes per
  row and in time to answer the query these exports exist for. Exit status is 0
  when they agree and 1 when they differ, so it runs as a check after changing
  an exporter. With no arguments it takes the newest readable export of each
  format under `out/runs/` and says what it skipped to get there. Fields one
  export lacks are excluded rather than failing the comparison, exports that
  differ only in how they key documents are diagnosed as such, and note text is
  never printed unless `--show-text` asks for it.
- **The batch example scripts save their runs too.** `parse_to_jsonl.py`,
  `parse_to_csv.py` and `parse_to_jsonl_batch.py` take their output path
  optionally; omit it and the run saves itself under `out/runs/<run-id>/` as
  `annotations.jsonl` (or `.csv`) beside a manifest, the same shape the CLI
  produces. Each also accepts `--run-id`. Passing a path explicitly writes no
  run directory. `problem_and_med_list.py` and `sqlite_browser.py` print reports
  and write nothing; `load_to_sqlite.py` consumes output another script
  produced, so it takes its paths explicitly.

#### Browsing and search

- **`examples/sqlite_reader.py`: browse a database.** Tables with row counts,
  full DDL (`--schema`), and rows from one table (`--table NAME --limit N`).
  With no path it opens the most recently modified `.db`/`.sqlite` under
  `out/runs/` — by modification time rather than run id, since a long job
  finishes after a short one that started later. Complements
  `sqlite_browser.py`, which runs fixed clinical reports: `sqlite_reader.py`
  assumes no schema and works on any SQLite file. Read-only, and `--no-text`
  omits the verbatim-note-text column.
- **`examples/sqlite_browser.py` takes the same default**: with no path it opens
  the newest database under `out/runs/` **that carries the tables its reports
  read**, and `--runs DIR` looks elsewhere. The qualifier is load-bearing —
  unlike `sqlite_reader.py` this script knows a schema, and
  `notes_of_interest.py` writes a differently-shaped `.sqlite` into the same run
  directories, frequently as the newest file. Auto-selection walks past those,
  and past anything unreadable, to the newest usable one. A path passed
  explicitly is never skipped: a named database of the wrong shape is reported
  as such rather than silently swapped for another. When nothing qualifies, the
  error gives a count and names the newest candidate.
- **`examples/notes_of_interest.py` saves every search** to
  `out/runs/<run-id>/notes_of_interest.sqlite`: a `search` row, the resolved
  `concepts`, a `documents` row per file scanned (*including* files with no
  mention, since a cohort denominator cannot be rebuilt from a list of hits),
  and `console` — the printed report verbatim, one row per line. The structured
  tables recover what the report elides: negated-only documents are hidden
  behind `--show-negated` on screen but always present in `documents`.
  `--no-save` opts out, and a search that fails early writes nothing, so a run
  directory always means a search that ran.

#### Library API

- **`ClinicalPipeline.assessed_attributes`** — which assertion attributes *this*
  pipeline populates, derived from the registry in
  `umlsmatch.assertion.attributes` and the instance's own switches. A consumer
  writing a fixed schema needs this before the first annotation exists, and the
  answer depends on how the pipeline was configured rather than on what the data
  turned out to contain.
- `umlsmatch.corpus.document_label()` — the single answer to "how is a document
  named in an export?" — plus `duplicate_labels()` and
  `duplicate_label_warning()`, which catch the case that answer cannot handle.

#### PHI gate

- `.gitignore` and `.githooks/phi-paths.pattern` cover `out/`, `data/`,
  `free_texts/`, the export suffixes (`.jsonl`, `.csv`, `.tsv`, `.txt`,
  `.sqlite`, `.db`, `.parquet`, `.xmi`) and `run_manifest.json`. The pattern
  lives in one file that both the pre-commit hook and CI read. A manifest lists
  every source path it processed, which in this corpus are patient and note
  identifiers; Parquet and the other export formats quote note text verbatim.

#### Packaging

- **Published on PyPI**, as a pure-Python wheel and an sdist. The base install
  pulls nothing: spaCy (`nlp`), FastAPI (`service`) and negspaCy (`compare`)
  are each an extra, so a batch or CLI user installs no web stack.

  ```bash
  pip install umlsmatch[nlp]
  python -m spacy download en_core_web_sm
  ```

- **The POS model is a separate command, and cannot be anything else.** spaCy
  models are not distributed on PyPI, so the only way for the `nlp` extra to
  name `en_core_web_sm` is a direct URL reference — and PyPI refuses any
  distribution whose metadata contains one. Installing the extra therefore
  leaves a working spaCy with no model. `pipeline/tokenizer.py` re-raises
  spaCy's load failure with the `spacy download` command in the message, so
  that gap reports itself rather than surfacing as an E050 that reads like a
  typo in the model name.
- **Released from CI by Trusted Publishing, never from a laptop.** A `v*` tag
  runs `.github/workflows/release.yml`, which builds, clears the same gate an
  ordinary push clears — it calls `ci.yml` rather than restating it, so a
  release cannot pass a weaker check — and uploads with a short-lived OIDC
  credential GitHub mints for that workflow in this repository. No API token
  exists to leak or rotate, and each upload carries PEP 740 attestations.

  Two guards run before the upload, both protecting something that cannot be
  undone: the tag must match the built version, so a `v0.2.0` tag over a tree
  still reading `0.1.0` fails rather than spending the wrong number; and
  `twine check` runs first, because a `long_description` PyPI rejects is only
  visible once that version is already gone.
- `LICENSE` and `NOTICE` both ship in the wheel, as Apache-2.0 §4(d) requires;
  `py.typed` ships with them, so the annotations are visible to consumers.
- The PHI gate's working-tree walk skips `dist/` and `build/` alongside `data/`
  and `out/`: all four are gitignored, so nothing inside them can be committed
  and the gate has no say over them. The skip is load-bearing rather than
  tidiness — a built wheel contains `entry_points.txt`, which the pattern's
  `\.txt$` clause matches, so without it `python -m build` before `pytest`
  fails the suite.

### Validation

Concept extraction is validated against Apache cTAKES as a reference
implementation, not as a target. Against cTAKES' own shipped 2016AB dictionary
it measures **F1 0.971** (P 0.982 / R 0.960); against modern UMLS 2026AA it
measures F1 0.756–0.782, a difference in dictionary *content* rather than
matcher logic.

The assertion figures are agreement with a faulty oracle, not accuracy — cTAKES
is a sound benchmark for concept extraction and **not** for assertion. Scored
against adjudicated verdicts the same rules read very differently (`negated`
F1 0.870 against 0.678). Read [README](README.md#validation) and
[docs/ADJUDICATION_RESULTS.md](docs/ADJUDICATION_RESULTS.md) before quoting any
of them.

### Licence and attribution

Apache-2.0, matching Apache cTAKES. Five modules are derived from cTAKES and one
lexicon from MetaMapLite/NegEx; Apache-2.0 §4(d) requires that attribution travel
with the work, so `NOTICE` ships in the wheel alongside `LICENSE`.

# Examples

Runnable scripts showing how to use umlsmatch. Every one was executed against
the 20-note corpus in `free_texts/synthetic/`; the outputs quoted below are real.

**Prerequisites:** install the package and build the dictionary first — see
[../docs/USER_GUIDE.md](../docs/USER_GUIDE.md) §3–4. Without a dictionary these all exit
with a message telling you to build one.

```bash
pip install -e ".[nlp,dev]"
python -m umlsmatch.build --umls-dir "<path>/2026AA/META" --ctakes-root ../ctakes-java
```

## Where output goes

The five scripts that run the pipeline — `parse_to_jsonl.py`, `parse_to_csv.py`,
`parse_to_parquet.py`, `parse_to_sqlite.py`, `parse_to_jsonl_batch.py` — save
each run under `out/runs/` when you do not name a destination, matching
`python -m umlsmatch`:

```
out/runs/20260921T071455Z-3f9a1c/
├── run_manifest.json     what was run, against which dictionary, with what filters
└── annotations.jsonl     (or .csv, .parquet, .db — one per format)
```

Nothing overwrites a previous run, so two configurations can be run back to back
and compared; `--run-id NAME` names one instead of taking the timestamp. Passing
an output path explicitly writes no run directory — this is a default, not a
policy.

`notes_of_interest.py` saves its search the same way. `problem_and_med_list.py`
and `sqlite_browser.py` print and write nothing. `load_to_sqlite.py` reads
output another script produced, so it takes paths explicitly. `sqlite_browser.py`
and `sqlite_reader.py` open the newest database under `out/runs/` when given no
path — which is what makes `parse_to_sqlite.py` followed by a bare
`sqlite_browser.py` work.

> **Everything under `out/` is PHI** — annotation records quote note text
> verbatim, and a manifest lists source paths. `out/` is gitignored and matched
> by the pre-commit gate.

## What every exporter agrees on

`parse_to_csv.py`, `parse_to_jsonl.py`, `parse_to_parquet.py`,
`parse_to_sqlite.py` and `parse_to_jsonl_batch.py` write the same annotations in
different containers, so two exports of one corpus carry the same rows and join
on the same key — verified by running all five and diffing: **2,188 rows,
identical on `(document, start, end, cui)`**. (`parse_to_sqlite.py` normalizes
the document into a `documents` table, so it joins via `doc_id`.)

| | rule | why |
|---|---|---|
| **Document key** | the note's **file name** (`umlsmatch.corpus.document_label`) | a full path records where the corpus was mounted, which differs per machine and stops two exports joining |
| **Empty notes** | **not written** — counted and reported instead | a row-per-annotation table cannot represent "this document, and nothing in it" |
| **Bad documents** | reported and **skipped**, never fatal | one malformed note should not lose a long corpus run |
| **`term`** | carried by all five | it is what explains a surprising match |
| **Unassessed attributes** | **no column at all** | a default pipeline never assesses `conditional` or `generic`; an empty column there is a fact about the pipeline, not the mention |

Every run reports documents, empty and failed, and records them in its manifest,
so nothing is dropped silently. Attribute columns come from
`ClinicalPipeline.assessed_attributes`, read off the pipeline the run actually
built — so `conditional=True` gets that column back with no edit to any column
list, and `--all-attributes` keeps every column for a schema that must not
change shape between runs.

**That includes the database.** `parse_to_sqlite.py` creates `annotations`
without those columns, which is why `load_to_sqlite.py` generates its schema
from `ANNOTATION_COLUMNS` rather than holding one `CREATE TABLE` literal, and
`Loader` reads the column list off whatever table it is pointed at. So
`load_to_sqlite.py` still builds the **full** table (its other input is a Java
cTAKES silver standard, which assesses all six); `--append` works in both
directions, the narrow table being the wide one minus columns; and appending
values a table cannot hold prints how many were dropped per column instead of
losing them silently. `--all-attributes` builds the full table here too.

Keying by file name has one cost: `iter_text_files` recurses, so two notes with
the same name in different subdirectories become one document. All five detect
this and warn before writing rather than merging quietly.

---

| Script | What it shows |
|---|---|
| [quick_start.py](quick_start.py) | Analyze one note; read offsets and negation |
| [parse_to_jsonl.py](parse_to_jsonl.py) | Process a directory; stream JSONL; one record per document |
| [parse_to_csv.py](parse_to_csv.py) | Flatten to CSV for pandas/Excel/SQL |
| [parse_to_parquet.py](parse_to_parquet.py) | The same table as Parquet — typed, nullable, compressed |
| [notes_of_interest.py](notes_of_interest.py) | Find patients with a condition — why this isn't grep |
| [problem_and_med_list.py](problem_and_med_list.py) | Per-note problem list, med list, documented negatives |
| [parse_to_jsonl_batch.py](parse_to_jsonl_batch.py) | Scale across cores with multiprocessing |
| [parse_to_sqlite.py](parse_to_sqlite.py) | Analyze straight into a queryable SQLite database |
| [load_to_sqlite.py](load_to_sqlite.py) | Load CSV **or** JSONL exports into SQLite |
| [sqlite_browser.py](sqlite_browser.py) | Analytical queries — a SQL cookbook |
| [sqlite_reader.py](sqlite_reader.py) | Browse any database — tables, schema, rows |
| [compare_exports.py](compare_exports.py) | Check the formats agree; what each costs |

The full workflow is **analyze → export → load → query**: `parse_to_jsonl.py`,
`parse_to_csv.py`, `parse_to_parquet.py` or `parse_to_jsonl_batch.py`, then
`load_to_sqlite.py`, then `sqlite_browser.py`. `parse_to_sqlite.py` collapses
the first three steps into one when the export itself is not needed.

---

## quick_start.py — Quickstart

```bash
python examples/quick_start.py
```

```
  132:156    C0011860   DISORDER         type 2 diabetes mellitus
  200:205    C0817096   ANATOMY    yes   chest
```

The one habit to carry over: **build the pipeline once and reuse it.**
Construction loads a spaCy model and opens the dictionary (seconds); `analyze()`
afterwards is fast. Constructing per document is the most common performance
mistake.

## parse_to_jsonl.py — Batch to JSONL

```bash
python examples/parse_to_jsonl.py free_texts/synthetic --groups DISORDER,DRUG
# 20/20 documents -> out/runs/20260921T071455Z-3f9a1c/annotations.jsonl
```

JSONL streams, survives a crash mid-run, and loads directly into pandas or `jq`.
The script also shows per-document error handling — one malformed note should
not abort a corpus run.

With no output path the run saves itself — see [Where output
goes](#where-output-goes) below. Pass a path to choose the destination instead.

## parse_to_csv.py — CSV export

```bash
python examples/parse_to_csv.py free_texts/synthetic --groups DISORDER
# 684 rows from 20 documents -> out/runs/<run-id>/annotations.csv
```

One row per annotation. `--no-text` drops the verbatim matched span, which is
what makes the export shareable — see the PHI note below.

## parse_to_parquet.py — Parquet export

```bash
pip install pyarrow          # the only example with a dependency of its own
python examples/parse_to_parquet.py free_texts/synthetic --groups DISORDER
# 684 rows from 20 documents -> out/runs/<run-id>/annotations.parquet (22 kB, zstd)
```

The same table as the CSV, typed and compressed — the unfiltered corpus is
1.6 MB as CSV and 214 kB as Parquet. The difference that matters more than size
is nulls: CSV has none, so an attribute the pipeline never assessed has to be
written as an empty cell, while Parquet stores it as an actual null — distinct
from `false`, which means it *was* assessed. `--compression` selects the codec
(zstd by default), and the run's provenance stamp is embedded in the file's own
metadata as well as written to the manifest beside it, so a file that gets
copied out of its run directory still says what produced it.

## notes_of_interest.py — Cohort search

```bash
python examples/notes_of_interest.py free_texts/synthetic "chest pain"
```

```
Scanned 20 documents.
  11 affirm the concept
  10 mention it only as negated
  23 do not mention it
```

**This is the example worth reading.** Of 21 documents mentioning chest pain,
**10 mention it only to rule it out** — a keyword search would put all 21 in the
cohort and be wrong about nearly half.

Matching is by CUI, so synonyms and morphological variants collapse
automatically: `"diabetes mellitus"` resolves to C0011849 and finds the concept
however it was written.

**Every search saves itself** to
`out/runs/<run-id>/notes_of_interest.sqlite` — four tables:

| table | holds |
|---|---|
| `search` | one row: what was asked, over what corpus, with what totals |
| `concepts` | what the phrase resolved to (empty when `--cui` was used) |
| `documents` | one row per file scanned, **including the ones with no mention** |
| `console` | the printed report, verbatim, one row per line |

`documents` carries every file because *absent* is a result — a cohort
denominator cannot be reconstructed from a list of matches. It also recovers
what the console elides: the report hides negated-only documents behind
`--show-negated`, but they are always in the table.

```sql
SELECT source, negated_mentions FROM documents WHERE status = 'negated_only';
```

The transcript is stored *beside* the structured tables, not instead of them:
the tables are what you query, the transcript is what you show someone who asks
what you actually ran. `--no-save` turns saving off. A search that fails early —
an unresolvable phrase, an empty corpus — writes nothing, so a run directory
always means a search that ran.

Browse the result with [sqlite_reader.py](sqlite_reader.py), which will pick it
up automatically as the newest database under `out/runs/`.

## problem_and_med_list.py — Problem and medication lists

```bash
python examples/problem_and_med_list.py free_texts/synthetic --limit 1
```

Groups annotations into problems, medications and procedures, drops negated
mentions, and **deduplicates by CUI** — so "diabetes", "DM" and "diabetes
mellitus type 2" become one entry, not three. Deduplicating by text would not do
that.

It also reports *documented negatives* separately, because "denies chest pain"
is clinically meaningful information, not an absence of it.

## parse_to_jsonl_batch.py — Parallel batch

```bash
python examples/parse_to_jsonl_batch.py free_texts/synthetic --workers 2
# 2,188 annotations in 2.2s (9.1 docs/sec)   [--workers 4]
# -> out/runs/<run-id>/annotations.jsonl
```

A pipeline is **not thread-safe**, so threads don't help — but the work is
CPU-bound, so processes do. Two details are load-bearing:

- Each worker builds **one** pipeline in an initializer, not per document.
- The `if __name__ == "__main__"` guard is **required**, not stylistic: without
  it, Windows/macOS spawn makes every child re-import and re-execute the module.

Size `--workers` against RAM as well as cores. Each worker holds its own model
and SQLite connection: ~210 MB resident once warm, climbing toward ~400 MB on a
long run as the SQLite page cache and the matcher's candidate memo fill.

## parse_to_sqlite.py — Straight into SQLite

```bash
python examples/parse_to_sqlite.py free_texts/synthetic
# wrote out/runs/<run-id>/annotations.db in 7.5s (5.9 docs/sec)
python examples/sqlite_browser.py          # no path needed: newest under out/runs/
```

```
  documents        20
  annotations      2,188
  distinct CUIs    1,078
  negated          245 (11.2%)
```

The one-step version of analyze → export → load: annotations go into the
database as each document is produced, and no intermediate export is written.
It imports its schema from `load_to_sqlite.py` rather than restating it, so both
routes end at the same database — verified by building the corpus both ways and
diffing: identical schema, and all 2,188 rows equal.

Prefer the two-step route when the export is itself a deliverable, when the same
annotations must be loaded more than once without re-running the pipeline, or
when you want the expensive analysis pass and the cheap load pass to be
separately restartable. Prefer this one when the database is all you wanted.

It follows the same conventions as every other exporter here, including leaving
out columns for attributes the pipeline never assesses — see [What every
exporter agrees on](#what-every-exporter-agrees-on).

## load_to_sqlite.py — Load into SQLite

```bash
python examples/load_to_sqlite.py out/annotations.csv   out/annotations.db
python examples/load_to_sqlite.py out/annotations.jsonl out/annotations.db
```

Accepts **three** inputs, detected by extension and record shape:

| input | from |
|---|---|
| `.csv` | `parse_to_csv.py` |
| `.jsonl` with `annotations` | `parse_to_jsonl.py`, `parse_to_jsonl_batch.py`, `python -m umlsmatch --json` |
| `.jsonl` with `mentions` | a Java cTAKES silver standard (`tools/run_java_ctakes.py`) |

CSV and JSONL produce an identical database. Loading a silver standard lets you
query cTAKES' own output with the same SQL — useful for diffing the two
pipelines; mentions expand to one row per concept, and only negation carries
across, so `term` is always NULL there (Java's XMI records spans and concepts
but not the string that matched). A file with neither key, or one that yields no
annotations, **fails and removes the database** rather than leaving an empty one.

Two column renames happen on the way in, and they are not cosmetic: `group` →
`semantic_group` and `start`/`end` → `start_offset`/`end_offset`, because
`GROUP` and `END` are reserved SQL. Keeping the originals would force every
later query to quote them.

**On document keys.** Every exporter records a document by *file name*, so
appending a CSV and a JSONL export of one corpus lands on the same `documents`
rows rather than doubling every count:

```bash
python examples/load_to_sqlite.py out/annotations.jsonl out/db.sqlite --append
# 20 documents, not 40
```

`--basename` remains for inputs that do not follow that convention — a silver
standard, or `python -m umlsmatch --json` output, whose `source` is the path it
was given.

## sqlite_browser.py — Query it

```bash
python examples/sqlite_browser.py                    # newest db under out/runs/
python examples/sqlite_browser.py --cui C0011849
python examples/sqlite_browser.py --sql "SELECT ..."
```

Prints five reports, each with **the SQL it ran**, so it doubles as a cookbook:
annotations by group, top affirmed concepts, largest documents, most-negated
concepts, and co-occurring disorder pairs. Opened **read-only**, so a stray
`DELETE` in `--sql` bounces off rather than mutating your corpus.

Unlike `sqlite_reader.py` it **knows the schema**, so with no path it opens the
newest database under `out/runs/` *that it can actually report on* —
`notes_of_interest.py` writes a differently-shaped `.sqlite` into the same run
directories and is often newer. It walks past those to the newest one with the
right tables and says which it landed on. A path you pass explicitly is a
decision, so it is not skipped: name a database of the wrong shape and it names
the missing tables and points you at `sqlite_reader.py` instead.

The negation report is the interesting one:

```
  cui       concept             mentions  negated  pct_negated
  C0231303  Distress (finding)  18        18       100.0
  C0019080  Blood loss          35        31        88.6
```

Some concepts appear almost *exclusively* to be ruled out. Counting raw mentions
would treat every one of those as a positive finding.

## sqlite_reader.py — Browse it

Unnumbered on purpose: the numbered scripts are a progression, this one is a
utility you reach for at any point in it.

```bash
python examples/sqlite_reader.py                    # newest db under out/runs/
python examples/sqlite_reader.py --table annotations --limit 5 --no-text
python examples/sqlite_reader.py --schema
```

Where `sqlite_browser.py` answers clinical questions with fixed reports, this
answers "what tables exist, what columns, what does a row look like" — for a
database from elsewhere, or a query returning something surprising. It assumes
no schema, so it works on any SQLite file. Read-only. With no path it opens the
newest database under `out/runs/`, judged by **modification time, not run id**:
a long job finishes after a short one that started later.

Row output shows `NULL` distinctly from `0`, which is how the tri-state
attributes read at rest:

```
  id  cui       preferred_text  negated  history_of  conditional  generic
  1   C0262926  Hx              0        0           NULL         NULL
```

`conditional` and `generic` are *not assessed*, not *assessed and absent* — the
distinction the whole pipeline is built around, visible in the stored rows.

Nothing writes a `.db` into a run directory on its own; build one beside the
annotations it came from:

```bash
python examples/load_to_sqlite.py \
    out/runs/<run-id>/annotations.jsonl out/runs/<run-id>/annotations.db
```

---

## compare_exports.py — Do the formats agree?

```bash
python examples/compare_exports.py           # newest of each format under out/runs/
python examples/compare_exports.py out/runs/a/annotations.csv out/runs/b/annotations.db
```

```
             csv == jsonl          identical (2,188 rows)
             csv == parquet        identical (2,188 rows)
             csv == sqlite         identical (2,188 rows)

cost
  format           size   per row   normalize   top-CUI query
  csv              1,647,303B      102B        95ms          16.8ms   (1,987 CUIs)
  jsonl            3,539,354B      220B        86ms          24.2ms   (1,987 CUIs)
  parquet            213,608B       13B       350ms           3.4ms   (1,987 CUIs)
  sqlite           2,654,208B      165B        91ms           6.3ms   (1,987 CUIs)
```

Checks the claim in [What every exporter agrees on](#what-every-exporter-agrees-on)
instead of asking you to take it on trust. It normalizes each format into one
shape — undoing SQLite's `semantic_group`/`start_offset` renames, collapsing
every spelling of an optional boolean onto True/False/None — and compares the
rows exactly. **Exit status is 0 when they agree, 1 when they differ**, so it
runs as a check after changing an exporter, not only as a report.

A field one export lacks is excluded rather than failing the comparison (a
`--no-text` CSV against a full Parquet export is a fair question about the other
13 fields), and two exports that differ *only* in how they key documents are
told so by name rather than by thousands of rows of diff. It reads PHI but prints
none: a differing row is reported by document, CUI and offsets unless you pass
`--show-text`.

The `top-CUI query` column is the one to choose a format on — each format
answers "affirmed mentions per concept" in its own idiom, so Parquet reads two
columns and SQLite walks an index while the text formats parse everything.
`normalize` is this script's own row-by-row load, not a property of the format.

## Two things to keep in mind

**Output contains PHI.** Every annotation's `text` field is verbatim note
content — and that includes the SQLite databases built by `parse_to_sqlite.py`
and `load_to_sqlite.py`, whose `documents.source` column additionally holds note
paths. Treat JSONL, CSV, Parquet
and `.db` files with the same controls as the source notes; the `--no-text` flag
on `parse_to_csv.py`, `parse_to_parquet.py` and `parse_to_sqlite.py` exists for
when you only need counts.

**Check the accuracy limits before trusting results.**
[../docs/USER_GUIDE.md](../docs/USER_GUIDE.md) §7 has the measured numbers. The one that
most affects these examples: negation precision is ~0.50, so roughly half of
"negated" flags are wrong. Cohort counts from `notes_of_interest.py` and the negation report in
`sqlite_browser.py` are starting points for review, not final answers.

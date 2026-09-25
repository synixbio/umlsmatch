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

`notes_of_interest.py` saves each search under `out/runs/` unless you pass
`--no-save`, matching `python -m umlsmatch`. `quick_start.py`,
`problem_and_med_list.py` and `sqlite_browser.py` print and write nothing.
`load_to_sqlite.py` reads an export produced elsewhere, so it takes paths
explicitly. `sqlite_browser.py` and `sqlite_reader.py` open the newest database
under `out/runs/` when given no path.

> **Everything under `out/` is PHI** — annotation records quote note text
> verbatim, and a manifest lists source paths. `out/` is gitignored and matched
> by the pre-commit gate.

---

| Script | What it shows |
|---|---|
| [quick_start.py](quick_start.py) | Analyze one note; read offsets and negation |
| [notes_of_interest.py](notes_of_interest.py) | Find patients with a condition — why this isn't grep |
| [problem_and_med_list.py](problem_and_med_list.py) | Per-note problem list, med list, documented negatives |
| [load_to_sqlite.py](load_to_sqlite.py) | Load CSV **or** JSONL exports into SQLite |
| [sqlite_browser.py](sqlite_browser.py) | Analytical queries — a SQL cookbook |
| [sqlite_reader.py](sqlite_reader.py) | Browse any database — tables, schema, rows |

The full workflow is **analyze → export → load → query**:

```bash
python -m umlsmatch free_texts/synthetic --json -o out/annotations.jsonl
python examples/load_to_sqlite.py --basename out/annotations.jsonl out/runs/demo/annotations.db
python examples/sqlite_browser.py          # no path needed: newest under out/runs/
```

```
  documents        20
  annotations      2,188
```

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

## load_to_sqlite.py — Load into SQLite

```bash
python examples/load_to_sqlite.py --basename out/annotations.jsonl out/annotations.db
python examples/load_to_sqlite.py out/annotations.csv out/annotations.db
```

Accepts **three** inputs, detected by extension and record shape:

| input | from |
|---|---|
| `.csv` | one row per annotation, keyed by a `document` column |
| `.jsonl` with `annotations` | `python -m umlsmatch --json` |
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

**On document keys.** `documents` is keyed by whatever the export records as a
note's source. `python -m umlsmatch --json` records the path it was given,
which differs per machine; `--basename` keys by file name instead, so appending
two exports of one corpus lands on the same `documents` rows rather than
doubling every count:

```bash
python examples/load_to_sqlite.py --basename out/second.jsonl out/db.sqlite --append
# 20 documents, not 40
```

Pass it for silver standards too, whose `source` is likewise a path.

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

Nothing writes a `.db` into a run directory on its own; load an export into
one:

```bash
python examples/load_to_sqlite.py --basename out/annotations.jsonl out/runs/<run-id>/annotations.db
```

## Two things to keep in mind

**Output contains PHI.** Every annotation's `text` field is verbatim note
content — and that includes the SQLite databases built by `load_to_sqlite.py`,
whose `documents.source` column additionally holds note paths. Treat JSONL, CSV,
Parquet and `.db` files with the same controls as the source notes.

**Check the accuracy limits before trusting results.**
[../docs/USER_GUIDE.md](../docs/USER_GUIDE.md) §7 has the measured numbers. The one that
most affects these examples: negation precision is ~0.50, so roughly half of
"negated" flags are wrong. Cohort counts from `notes_of_interest.py` and the negation report in
`sqlite_browser.py` are starting points for review, not final answers.

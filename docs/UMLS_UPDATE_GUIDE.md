# Rebuilding the dictionary for a new UMLS release

The application dictionary is a **generated artifact**, not source. It is
gitignored and rebuilt from the UMLS Metathesaurus files. When NLM publishes a
new release (twice a year — `AA` in spring, `AB` in autumn), this is the whole
procedure.

Current build: **UMLS 2026AA**, 398,037 concepts / 1,006,158 terms.

---

## TL;DR

```bash
python -m umlsmatch.build --umls-dir "<path>/2026AB/META" --ctakes-root ../ctakes-java
python -m pytest tests/ -q
```

That runs all three stages in the one order that works. Each is still a script
you can run on its own — see the steps below, and `--from <stage>` to resume a
long build after a failure rather than redoing stage 1.

**Order matters, and getting it wrong is silent.** Stage 2 changes term
spellings and stage 3 computes rare-word statistics over them, so 3 must follow
2. Three things now enforce that, at three different moments:

| when | what catches it |
|---|---|
| while building | `python -m umlsmatch.build` runs the stages in order |
| end of stage 2 | `retokenize_terms.py` drops `rare_term`, so a forgotten stage 3 fails loudly instead of leaving a stale index |
| at load time | `RareWordMatcher` refuses a dictionary with no `terms_retokenized` marker |

The last of those is the one that matters most, because skipping stage 2 has no
other symptom: the build succeeds, the pipeline starts, and ~96k hyphenated and
clitic terms just never match.

---

## Prerequisites

- A UMLS Metathesaurus licence from NLM, and the **Full Release** downloaded and
  unpacked. Only `META/MRCONSO.RRF` and `META/MRSTY.RRF` are used.
- The optional `nlp` extra for step 2: `pip install -e ".[nlp,dev]"`.
- A clone of `apache/ctakes` for step 3 (it parses `BAD_POS_TERMS` from the Java
  source rather than hand-copying it). Default `../ctakes-java`.

Expect ~2.5 GB for `MRCONSO.RRF` and ~590 MB of SQLite output for the default
all-sources build (a narrower `--sources` build is smaller: cTAKES' own seven
come to ~370 MB). Budget **tens of minutes** for a full run — stage 1 alone
takes many minutes over ~2.1 GB of input, which is why the three stages stay
separately resumable with `--from`. Importing cTAKES' *already-built*
dictionary instead is a different operation and takes ~14 s.

---

## Step 1 — Build

```bash
python tools/build_dictionary.py \
  --umls-dir "<path>/<RELEASE>/META" \
  --out data/umls_sno_rx.sqlite
```

Applies cTAKES' filters: English, `SUPPRESS='N'`,
`SAB IN (SNOMEDCT_US, RXNORM)`, and `--tui-set ctakes` — the **49-TUI
whitelist** recovered from cTAKES' shipped dictionary. It also replaces SNOMED
fully-specified-name spellings with their tag-stripped forms
(`pneumonia (disorder)` → `pneumonia`), keeping the original in the `text`
column.

The release label is taken from the **parent directory name** of `--umls-dir`
and stored in `meta.umls_release`, so keep NLM's `<RELEASE>/META` layout or the
provenance recorded in the database will be wrong.

Smoke-test a new path before committing to the full read:

```bash
python tools/build_dictionary.py --umls-dir "<...>/META" --out /tmp/t.sqlite --limit 400000
```

`--limit` builds are marked `limited=yes` in `meta` and rejected by the test
suite, so a partial artifact can't quietly become the production dictionary.

### Flags worth knowing

| Flag | When |
|---|---|
| `--tui-set all` | Keep all 136 mapped TUIs instead of cTAKES' 49. **Not recommended** — adds ~44k concepts of generic nouns (`tablet`, `patient`, `daily`) and roughly halves precision. |
| `--sources` | Add vocabularies beyond SNOMEDCT_US + RXNORM (e.g. LOINC). Diverges from cTAKES; measure before adopting. |
| `--keep-semantic-tags` | Keep `(disorder)`-style SNOMED tags. Inflates the dictionary ~30% with strings that never occur in narrative. |
| `--keep-excluded-texts` | Skip the ported cTAKES stop-term list. |

---

## Step 2 — Align spellings with the tokenizer

```bash
python tools/retokenize_terms.py --db data/umls_sno_rx.sqlite
```

**Do not skip this.** The matcher verifies a candidate by joining the document's
token norms with single spaces and comparing to the stored `norm`. A term only
matches if the dictionary spells it as the tokenizer segments it. Raw UMLS
spells it `chest x-ray`; spaCy produces `chest x - ray`. Without this step every
hyphenated term — ~96k of them — silently never matches.

It rewrites ~22% of spellings and iterates to a tokenization fixed point (a
single pass isn't stable for some chemical names). Re-run with `--force`;
`--max-passes` bounds the iteration.

> Never run this on `data/umls_ctakes_16ab.sqlite` — cTAKES' shipped dictionary
> is already pre-tokenized, and re-tokenizing would corrupt it. The tool detects
> this and refuses.

If you change the spaCy model, re-run this step and step 3: term spellings are
tied to the tokenizer that produced them.

---

## Step 3 — Rare-word index

```bash
python tools/build_rare_word_index.py --db data/umls_sno_rx.sqlite --ctakes-root ../ctakes-java
```

Indexes each term by its statistically rarest token plus that token's offset —
the structure `RareWordMatcher` looks up. Takes a few seconds. Sanity figures
for a current build: ~700k indexed terms, ~100k distinct rare words, under 10
candidates per key.

If the top rare-word keys look like punctuation fragments (`(qualifier`) rather
than clinical words, something upstream went wrong.

---

## Step 4 — Verify

```bash
python -m pytest tests/ -q
```

[tests/test_dictionary.py](../tests/test_dictionary.py) is the acceptance gate. It checks the build wasn't
`--limit`ed, the TUI filter was applied as configured, every concept has a
semantic type and a group, norms are normalized and tokenizer-aligned, scale is
plausible, and a handful of stable CUIs resolve.

### Expected failure after a release bump: `test_known_concept_lookup`

This is usually **release drift, not a defect**. UMLS retires and re-words
synonyms between releases. A real example already in the file: `chest x-ray`
stopped resolving to C0039985 on 2021AB → 2026AA; the concept is unchanged
("Plain X-ray of chest"), but SNOMED dropped the bare synonym, keeping only
`plain chest x-ray` / `plain x-ray of chest` / `plain cxr (chest x-ray)`.

Confirm before editing the test:

```bash
python -c "
import sqlite3; c=sqlite3.connect('data/umls_sno_rx.sqlite')
print(c.execute('SELECT preferred_text,best_group FROM concept WHERE cui=?',('C0039985',)).fetchone())
for r in c.execute('SELECT DISTINCT norm FROM term WHERE cui=? LIMIT 20',('C0039985',)): print(' ', r[0])"
```

If the concept is present and only its synonyms moved, update the probe string
in `KNOWN_CONCEPTS` and note why. If the concept vanished entirely, that's worth
investigating — check `MRCUI.RRF` for a retirement mapping.

Remember probe strings must use **tokenizer spelling** (`plain chest x - ray`,
not `plain chest x-ray`).

---

## Step 5 — Re-measure (optional but recommended)

```bash
python tools/score_parity.py     --jsonl free_texts/json/silver.jsonl --db data/umls_sno_rx.sqlite
python tools/score_negation.py   --jsonl free_texts/json/silver.jsonl --db data/umls_sno_rx.sqlite
python tools/score_boundaries.py --jsonl free_texts/json/silver.jsonl
```

**Interpret parity carefully.** The silver standard was generated by Java cTAKES
running its 2016AB dictionary, so its gold CUIs *are* 2016AB CUIs. A newer
release will score lower for reasons that are not quality problems — newer
builds contain ~94% of the gold CUIs, and surface forms drift further with each
release.

Scoring a modern build against 2016AB output measures *agreement with
decade-old terminology*, not clinical correctness. **Never tune the application
dictionary to raise this number.** For genuine quality measurement you need
clinician-annotated gold, which this project does not yet have.

Use `data/umls_ctakes_16ab.sqlite` when the question really is "does the Python
pipeline behave like Java cTAKES?" That build holds the dictionary constant so
differences are attributable to the pipeline.

---

## Keeping multiple releases side by side

Nothing forces a single dictionary — `--out` and `--db` are free:

```bash
python tools/build_dictionary.py --umls-dir "<...>/2026AB/META" --out data/umls_2026AB.sqlite
python tools/retokenize_terms.py --db data/umls_2026AB.sqlite
python tools/build_rare_word_index.py --db data/umls_2026AB.sqlite --ctakes-root ../ctakes-java
python tools/score_parity.py --jsonl free_texts/json/silver.jsonl --db data/umls_2026AB.sqlite
```

Compare before promoting, then swap the file into place. Every database records
its own provenance:

```bash
python -c "
import sqlite3;c=sqlite3.connect('data/umls_sno_rx.sqlite')
[print(f'{k:<24}{v}') for k,v in sorted(c.execute('SELECT key,value FROM meta'))]"
```

---

## Regenerating the reference dictionary and TUI whitelist

Only needed when upgrading the **cTAKES** version, not the UMLS release.
[src/umlsmatch/umls/ctakes_tuis.py](../src/umlsmatch/umls/ctakes_tuis.py) is checked in, so a cTAKES install is not
required for a normal rebuild.

```bash
python tools/import_ctakes_dictionary.py \
  --script "D:/apps/apache-ctakes-<VERSION>/resources/org/apache/ctakes/dictionary/lookup/fast/sno_rx_16ab/sno_rx_16ab.script" \
  --out data/umls_ctakes_16ab.sqlite

python tools/extract_ctakes_tuis.py --db data/umls_ctakes_16ab.sqlite
```

If the whitelist changes, `git diff` on the generated module shows exactly what
moved — then rebuild the application dictionary so the two stay consistent.

Related: [tools/transcode_semantic_tui.py](../tools/transcode_semantic_tui.py) regenerates the TUI→semantic-group
tables from the cTAKES Java source and has a `--check` mode suitable for CI.

---

## Licensing

UMLS content stays under your NLM licence; SNOMED CT has separate affiliate
terms outside the US. The built dictionary is UMLS-derived — fine for internal
LAN use, **not redistributable**. `data/` is gitignored for this reason as well
as size.

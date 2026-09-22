# umlsmatch — User Guide

Extract UMLS clinical concepts from free-text notes, with negation detection.
A Python-native clinical NLP pipeline designed to run entirely inside a private
network — no cloud calls, no PHI leaving your machines. Concept extraction is
validated against [Apache cTAKES](https://github.com/apache/ctakes) as a
reference implementation; see [§7](#7-accuracy-and-limitations).

**Read [§7 Accuracy and limitations](#7-accuracy-and-limitations) before using
this on anything that matters.** This is pre-1.0 software with measured, known
gaps.

---

## 1. What it does

Give it clinical text; get back concepts with UMLS CUIs, semantic groups,
character offsets, and a negation flag.

```
Input:   "Patient denies chest pain. Started on metformin 500 mg."

Output:  15:25  C0008031  FINDING  "chest pain"  negated=True
         38:47  C0025598  DRUG     "metformin"   negated=False
```

**In scope today:** sentence splitting, tokenization, POS tagging, UMLS concept
lookup (SNOMED CT + RxNorm), and four assertion attributes — `negated`,
`subject` (patient vs. family member), `history_of` and `uncertain`.

**Also in scope:** section awareness (assertion scope only, see below) and a
REST service — see [SERVICE.md](SERVICE.md).

**Deliberately not implemented:** `generic`. Too few positives to tell a
working rule set from a broken one, and no external cue lexicon to adopt, so it
reports `None` rather than a `False` nobody could audit.

**A prototype, off by default:** `conditional`. Rules exist but nothing has
measured them, so it also reports `None` unless you pass
`ClinicalPipeline(conditional=True)`.

**Not implemented:** relation extraction, temporal reasoning, coreference —
a different problem class.

**Read `None` as "not assessed".** Every attribute except `negated` is
three-state. `uncertain=None` means nobody looked; `uncertain=False` means the
rules looked and found no hedge. Coercing the two together is the one mistake
this API is shaped to prevent.

---

## 2. Requirements

- **Python 3.10+**
- **A UMLS Metathesaurus licence** from NLM — free, but approval takes days.
  Register at the [UTS portal](https://uts.nlm.nih.gov/uts/signup-login).
  Without it there is no dictionary and nothing will be found.
- ~4 GB disk: the UMLS release (~2.5 GB) plus a ~590 MB generated dictionary.
- No JVM. No network at runtime.

SNOMED CT has separate affiliate licensing outside the US — confirm your
jurisdiction's terms before deploying.

---

## 3. Install

```bash
git clone <your-repo> umlsmatch && cd umlsmatch
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

The `nlp` extra brings spaCy; `dev` adds pytest. The `en_core_web_sm` model is
a second command because spaCy models are not on PyPI and so cannot be an
extra's dependency — both spaCy and the model are required at runtime.

---

## 4. Build the dictionary

**This is mandatory** — the dictionary is a generated artifact, not shipped, and
nothing works without it.

Download and unpack the UMLS **Full Release**, then:

```bash
python -m umlsmatch.build --umls-dir "<path>/2026AA/META" --ctakes-root ../ctakes-java
python -m pytest tests/ -q
```

Only `MRCONSO.RRF` and `MRSTY.RRF` are read. Budget **tens of minutes** — stage
1 alone takes many minutes over ~2.1 GB. The three stages are separately
resumable (`--from`) so a late failure need not redo the first.

**Run all three steps in order.** Step 2 rewrites term spellings so they match
how the tokenizer segments text; skip it and every hyphenated term silently
never matches. Step 3 indexes over those spellings, so it must come after.

Step 3 wants a clone of `apache/ctakes` — it parses a stop-word list from the
Java source rather than hand-copying it:

```bash
git clone --depth 1 https://github.com/apache/ctakes.git ../ctakes-java
```

On Windows that clone needs long-path support:
`git -c core.longpaths=true clone --depth 1 ...`

Full detail, including upgrading to a newer UMLS release:
[UMLS_UPDATE_GUIDE.md](UMLS_UPDATE_GUIDE.md).

---

## 5. Use it

### Python

```python
from umlsmatch import ClinicalPipeline

with ClinicalPipeline() as nlp:
    for a in nlp.analyze("Patient denies chest pain. Started on metformin 500 mg."):
        print(a.cui, a.group, repr(a.text), a.negated, a.preferred_text)
```

Each result is an `Annotation`:

| Field | Meaning |
|---|---|
| `cui` | UMLS concept id, e.g. `C0008031` |
| `text` | exact matched document text |
| `start`, `end` | character offsets into the document |
| `group` | `DISORDER`, `DRUG`, `PROCEDURE`, `FINDING`, `ANATOMY`, `LAB`, `DEVICE`, `EVENT` |
| `negated` | `True` when a negation trigger scopes over it. Always assessed. **Scope, not semantics** — see below |
| `preferred_text` | the concept's canonical UMLS label |
| `term` | dictionary form that matched |
| `subject` | `"patient"`, `"family_member"`, or `None` if not assessed |
| `history_of` | `True` in a history-taking context; `None` if not assessed |
| `uncertain` | `True` when hedged; `None` if not assessed |
| `conditional` | `True` when asserted under a condition; `None` unless built with `conditional=True`, see §1 |
| `generic` | always `None` — never assessed, see §1 |

**`negated` means "inside a negation's scope", not "asserted absent".** In
`"denies chest pain"` the `ANATOMY` concept *chest* comes back `negated=True`
alongside the `FINDING` *chest pain*. The note plainly does not say the patient
has no chest — but the mention sits in the trigger's scope, and scope is what
this attribute reports. That is the ConText convention, and cTAKES does the
same.

It is deliberate, and it moves the measured number: adjudication marks such
mentions `unclear` and excludes them, so the published P 0.887 describes the
rest. If bare sites are noise for you, drop the mention rather than reinterpret
the flag — filter on `group`, or see the chart-furniture filter in
[ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md).

**Probe with a sentence, not a bare word.** `nlp.analyze("metformin")` returns
nothing while `nlp.analyze("Started on metformin.")` finds `C0025598`. The
dictionary is fine; a one-word input gives the part-of-speech tagger no context,
it guesses verb, and a verb tag is not a valid lookup anchor. See
[§9](#9-troubleshooting) for how to check membership directly.

Construction options:

```python
ClinicalPipeline(
    db_path=None,            # default: $UMLSMATCH_DB, else data/umls_sno_rx.sqlite
    groups={"DISORDER"},     # keep only these semantic groups
    resolve_overlaps=False,  # see §7 before enabling
    max_scope=8,             # negation scope cap, in tokens
    subject=True,            # assess patient vs. family member
    history=True,            # assess history-taking context
    uncertainty=True,        # assess hedging (a diagnostic — see below)
)
```

Turning an attribute off sets its field to `None`, not `False`. That is the
point of the switch: a pipeline that did not look should not be able to claim
it looked and found nothing.

**How much to trust each one.** Agreement with cTAKES on the 20-note reference
corpus in `free_texts/synthetic/` — and cTAKES is a faulty oracle for assertion,
so read these as a ceiling on disagreement rather than as accuracy:

| attribute | F1 vs. cTAKES | F1 adjudicated | what to do with it |
|---|---:|---:|---|
| `negated` | 0.678 | **0.870** | the only one this corpus still evidences; over-negates on some triggers |
| `history_of` | 0.667 | **0.277** | the weakest. Precision 0.46, recall 0.20 — trust a `True`, never infer from a `False` |
| `subject` | 0.182 | — | **2 reference positives.** Not measurable here; see below |
| `uncertain` | 0.000 | — | **measures the reference, not the rules**; zero overlap with cTAKES |

**`subject` and `uncertain` lost their evidence when the reference corpus
changed; the rules did not change with them.** The reference is now 20 synthetic
notes — single-patient prose with no EHR family-history tables and little
hedging. On the real-note corpus `subject` scored F1 0.809 vs cTAKES and
**0.956 adjudicated**, `uncertain` 0.162 against **0.564**; those are in
[ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md) and **cannot be reproduced
from anything shipped**. If you depend on either, validate on your own notes.

The two columns disagree because cTAKES is a faulty oracle for assertion, and
they disagree in *both* directions — `history_of` is worse than the left column
says, `negated` is better. The adjudicated column is a model pre-annotation and
still wants a clinician; both caveats, and the sampling fragility behind the
recall figures, are in [ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md).

Many documents — reuse one pipeline, since it holds the model and dictionary:

```python
with ClinicalPipeline() as nlp:
    for anns in nlp.analyze_documents(texts):
        ...
```

### Command line

```bash
python -m umlsmatch note.txt                          # readable table
python -m umlsmatch free_texts/synthetic/ --json -o out.jsonl
echo "Patient denies chest pain." | python -m umlsmatch -
python -m umlsmatch note.txt --groups DISORDER,DRUG --negated-only
python -m umlsmatch note.txt --profile clinical_recall
```

| Flag | Effect |
|---|---|
| `--db PATH` | dictionary to use |
| `--json` | JSONL, one object per document |
| `-o FILE` | write to a file instead of stdout |
| `--profile NAME` | `strict` (default) or `clinical_recall` — see [Two profiles](#two-profiles) |
| `--groups A,B` | keep only these semantic groups |
| `--negated-only` / `--affirmed-only` | filter by polarity |
| `--resolve-overlaps` | keep only the longest of overlapping matches |
| `--max-scope N` | negation scope cap in tokens (default 8). `0` means a zero-token scope — nothing is negated. Must be ≥ 0. |
| `--no-max-scope` | remove the cap; a trigger reaches to the terminator or sentence edge. Over-negates badly — for diagnosis, not production. |
| `--history-sections` | treat everything under a `PAST MEDICAL HISTORY` header as history. Adjudicated recall 0.198 → 0.670, but see the circularity warning in [ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md). |
| `--drop-header-mentions` | discard concepts inside a section heading — 1.9% of annotations, at a cost of 0.008 concept F1 |
| `--conditional` | assess conditional mentions. A prototype; off leaves the field `null`. |
| `--no-subject` / `--no-history` / `--no-uncertainty` | do not assess that attribute; its field becomes `null` |

Directories are searched recursively for `.txt`, `.text`, `.note`.

Every flag maps onto the `ClinicalPipeline` argument of the same name, so
`--max-scope 4` and `max_scope=4` behave identically; `--no-max-scope` is the
command-line spelling of `max_scope=None`.

**The `--no-*` attribute flags produce `null`, not `false`.** That is the same
three-state contract the Python API keeps, and it is why they are switches
rather than a post-hoc filter: "we did not assess this" is a different claim
from "we assessed this and it is absent". Use `--json` to see it — the readable
table cannot show the difference.

#### Two profiles

| profile | sets | for |
|---|---|---|
| `strict` *(default)* | `history_sections=False`, `drop_header_mentions=False` | reproducing the published validation figures |
| `clinical_recall` | both `True` | answering what is in the chart |

A profile supplies defaults; a flag still wins, so
`--profile clinical_recall --no-drop-header-mentions` means what it reads as.
The reasoning behind the default is in
[ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md).

During pre-release development `strict` was called `ctakes_parity`, after the
configuration the published cTAKES comparison figures were measured under. The
name never shipped and is not accepted; it is recorded here only so the term
stays decodable where it appears in the validation writeups.

### Worked examples

[examples/](../examples/) has eight runnable scripts — batch processing, CSV
export, cohort search, problem/medication lists, multiprocessing, and loading
and querying results in SQLite. Start with
[examples/README.md](../examples/README.md).

Point at a dictionary without repeating `--db`:

```bash
export UMLSMATCH_DB=/srv/ctakes/umls_sno_rx.sqlite   # Windows: $env:UMLSMATCH_DB=...
```

---

## 6. Performance

Measured on 244,538 chars of real notes, single-threaded, warm caches:

- **12.7 notes/sec** (~70,800 chars/sec)
- ~200 annotations per note
- ~210 MB RSS with the model loaded and caches warm (the dictionary is read
  from SQLite on demand, not loaded into memory). Expect growth toward ~400 MB
  on a long run: SQLite's page cache is capped at ~195 MB per connection, and
  the matcher's candidate memo is never evicted.

Model load takes a few seconds — **reuse one `ClinicalPipeline`** rather than
constructing per document. For higher throughput, run several processes; a
pipeline is not thread-safe for concurrent `analyze()` calls.

---

## 7. Accuracy and limitations

Read this section before trusting output. The full validation tables, including
the per-attribute figures, are in the
[README](../README.md#validation) — not repeated here.

### Measured agreement with Java cTAKES

On the 20 notes in `free_texts/synthetic/`, against Java cTAKES as a silver
standard. The two columns answer different questions:

| Metric | Application build (2026AA) | Matched-release build (2016AB) |
|---|---|---|
| Concept extraction (CUI-level) F1 | **0.76** (P 0.69 / R 0.84) | **0.97** (P 0.98 / R 0.96) |
| Negation F1 | **0.67** (accuracy 0.92) | 0.67 (accuracy 0.93) |

The **right** column is the engineering result: given cTAKES' own dictionary the
matcher reproduces it at F1 0.971, so the extraction algorithm is correct. The
**left** column is what you get in practice, because the application runs a
current UMLS release and reads synonyms from every vocabulary — roughly a third
of reported concepts would not be reported by cTAKES.

That balance is tunable. cTAKES' own seven vocabularies score P 0.79 / R 0.78
(F1 0.782, the best on a modern release), and that build already exists:

```bash
python -m umlsmatch note.txt --db data/umls_ctakes7.sqlite     # CLI
ClinicalPipeline("data/umls_ctakes7.sqlite")                   # Python
```

To rebuild it: `--sources SNOMEDCT_US,RXNORM,MTH,MSH,LNC,CHV,HPO`.

**Negation is the genuine weak spot** at F1 0.67 — below the ≥0.90 target on
both builds. Accuracy is ~0.92, but it over-flags (precision 0.51 against recall
0.97); see the limitations below.

Two caveats, in opposite directions:

- **20 notes is a very small sample, and they are synthetic** — machine-written,
  one per specialty, with none of the malformed EHR table dumps that drive the
  hardest failures on real notes. If anything these figures are optimistic.
- **Agreement with cTAKES is not clinical correctness.** A high score means
  "behaves like cTAKES", including reproducing its mistakes.

There is **no clinician-annotated evaluation**. The adjudicated figures are a
*model* pre-annotation of sampled mentions — a useful second opinion on cTAKES,
not ground truth. Nothing here has been verified by someone qualified to sign
off on it. See [ADJUDICATION_RESULTS.md](ADJUDICATION_RESULTS.md) for what that
set does and does not support.

### Known limitations

- **Spurious concepts.** Reading synonyms from every UMLS vocabulary pulls in
  eponyms and taxonomy names, so ordinary words can match: `"The quick brown fox
  jumped."` yields *Brown syndrome* and *Fox*. Filter with `groups=` and review
  output. A narrower `--sources` build avoids this.
- **Negation over-flags, but less than agreement suggests.** Against cTAKES
  polarity precision is 0.52 — *agreement*, not accuracy, and cTAKES marks
  textbook Review-of-Systems negatives (`"Negative for chills, fever, night
  sweats"`) as affirmed. Adjudicated, precision is **0.887**; of the mentions
  this pipeline negates and cTAKES does not, **24% are genuine over-negation**.
  `tools/diff_negation_adjudicated.py` attributes those per trigger. Treat a
  negated flag as good evidence, not proof.
- **`conditional` and `generic` are not usable.** `conditional` is a prototype
  behind `ClinicalPipeline(conditional=True)`; `generic` has no rules at all.
  Both return `None` rather than a value nobody measured. Of the other four,
  `negated` is the only one this corpus evidences — `subject` and `uncertain`
  are not measurable on it at all. See the table in §5.
- **General-domain POS tagger.** `en_core_web_sm`, not a clinical model —
  measured not to matter much: anchor-level agreement is 0.974, and substituting
  cTAKES' own tags moved precision by +0.009. Where it *is* visible is on input
  with no context: a bare `"metformin"` is tagged a verb and never anchors a
  lookup, while `"Started on metformin."` matches — see
  [§9](#9-troubleshooting).
- **Sentence segmentation diverges** from cTAKES (F1 0.53). Measured to have
  almost no effect on output, but it means sentence-scoped behaviour differs.
- **English only.**
- **Section awareness is narrow, and never sets polarity on its own.** Inside
  Review of Systems and Allergies a trigger's token cap is lifted, but a trigger
  must still fire; every other recognized header just ends the preceding
  section. Headers feed the *other* attributes — "Family History" text is
  attributed to a relative, and under `clinical_recall` a Past Medical History
  body becomes `history_of` and concepts inside a heading are dropped. Under the
  default `strict` profile neither of those is on, so headings change nothing
  about what is extracted. Disable with `ClinicalPipeline(sections=False)`.

### Not for clinical decision-making

This is research and data-processing software. It is not validated, not
regulated, and not suitable for direct patient-care decisions. Any use touching
patient care needs its own validation against clinician-annotated data for your
document types.

---

## 8. Handling PHI

Designed for private-network deployment: no network calls at runtime, and the
dictionary is local.

If real patient data flows through it:

- Pre-download the spaCy model and build the dictionary, then run with outbound
  network blocked. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if you
  later add transformer components.
- **Annotations carry PHI** — `text` holds verbatim note content. Treat output
  files with the same controls as the notes.
- Log annotation counts, not annotation text.
- The built dictionary is UMLS-derived: fine internally, **not
  redistributable**. `data/` is gitignored accordingly.
- Get your compliance owner to review before real data. This is the highest-
  consequence detail in deployment.

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'umlsmatch'`**
Not installed. Run `pip install -e ".[nlp,dev]"` from the repo root.

**`ImportError: ClinicalPipeline needs the optional 'nlp' extra`**
spaCy missing: `pip install -e ".[nlp]"`.

**`OSError: spaCy model 'en_core_web_sm' is not installed`**
spaCy is there but the model is not — they install separately. Run
`python -m spacy download en_core_web_sm`.

**`FileNotFoundError: dictionary not found: data/umls_sno_rx.sqlite`**
Build it — §4.

**`dictionary is missing table(s): rare_term`**
You skipped or interrupted step 3:
`python tools/build_rare_word_index.py --db data/umls_sno_rx.sqlite --ctakes-root ../ctakes-java`

**Nothing matches, or hyphenated terms never match**
You skipped `retokenize_terms.py` (§4 step 2). Check:
```bash
python -c "
import sqlite3;c=sqlite3.connect('data/umls_sno_rx.sqlite')
print(dict(c.execute('SELECT key,value FROM meta')).get('terms_retokenized'))"
```
Should print `yes`.

**A single word matches nothing, but the same word in a sentence does**
Working as intended, and it is the tagger talking, not the dictionary. cTAKES
will not anchor a dictionary lookup on a verb tag (`DEFAULT_EXCLUSION_TAGS`, 25
Penn tags inherited from it), and a general-domain tagger handed one bare word
has no context to do better:

```python
>>> nlp.analyze("metformin")               # tagged VB -> not an anchor -> []
[]
>>> nlp.analyze("Started on metformin.")   # tagged NN -> anchors -> C0025598
[Annotation(cui='C0025598', text='metformin', ...)]
```

**Probe with a sentence, not a bare term.** A one-word probe measures
`en_core_web_sm`'s part-of-speech guess, not whether the concept is in your
build. To check membership directly, query the dictionary instead:

```bash
python -c "
import sqlite3;c=sqlite3.connect('data/umls_sno_rx.sqlite')
print(c.execute('SELECT cui,text FROM term WHERE norm=?', ('metformin',)).fetchall())"
```

`norm` is the retokenized form, so a multi-word term is space-separated exactly
as `retokenize_terms.py` wrote it (`chest x - ray`, not `chest x-ray`).

**`test_known_concept_lookup` fails after a UMLS upgrade**
Usually synonym drift, not a bug — see
[UMLS_UPDATE_GUIDE.md](UMLS_UPDATE_GUIDE.md) §4.

**Too many spurious concepts**
Restrict groups (`--groups DISORDER,DRUG`). `--resolve-overlaps` reduces output
but measurably lowers recall (0.96 → 0.75) — it suppresses genuine nested
concepts, so prefer group filtering.

**Which dictionary am I using?**
```bash
python -c "
import sqlite3;c=sqlite3.connect('data/umls_sno_rx.sqlite')
[print(f'{k:<24}{v}') for k,v in sorted(c.execute('SELECT key,value FROM meta'))]"
```

---

## 10. Where things live

| Path | What |
|---|---|
| [src/umlsmatch/analyze.py](../src/umlsmatch/analyze.py) | `ClinicalPipeline` — the front door |
| [src/umlsmatch/dictionary/](../src/umlsmatch/dictionary/) | rare-word matcher |
| [src/umlsmatch/assertion/](../src/umlsmatch/assertion/) | negation |
| [src/umlsmatch/pipeline/](../src/umlsmatch/pipeline/) | sentences, tokens, POS |
| [src/umlsmatch/eval/](../src/umlsmatch/eval/) | scoring against cTAKES |
| [tools/](../tools/) | dictionary build and evaluation scripts |
| `data/` | generated artifacts (gitignored) |

**Documents:** [USER_GUIDE.md](USER_GUIDE.md) (this) ·
[UMLS_UPDATE_GUIDE.md](UMLS_UPDATE_GUIDE.md) (release upgrades) ·
[SERVICE.md](SERVICE.md) (HTTP service and deployment)

---

## 11. Licence and attribution

Derived from Apache cTAKES (Apache License 2.0) — retain attribution and the
`NOTICE` file in derived work. Algorithms, the semantic-type tables and the
dictionary curation are ported from the cTAKES source.

UMLS content is governed by your NLM licence.

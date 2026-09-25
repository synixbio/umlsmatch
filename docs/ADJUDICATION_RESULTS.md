# Adjudicated assertion scores

Every number in the README's assertion table is agreement with cTAKES, and
cTAKES is a faulty oracle for assertion. This is what the same attributes score
against adjudicated verdicts instead.

**The review sets on disk are built from the 20-note synthetic corpus**
(`free_texts/synthetic/*.txt`). A handful of shipped defaults rest on evidence
from a larger real-note corpus that is not distributed here; those are kept
separately at the end, because this corpus cannot re-test them.

> **These verdicts are a model pre-annotation, not a clinician's.** They were
> produced by an LLM reading the same sentences the pipeline reads, which makes
> them a strong baseline and *not* an independent standard. The blind
> `review.csv` in each directory is deliberately untouched so a clinician can
> still adjudicate without being anchored; comparing the two afterwards also
> yields a model-vs-clinician agreement figure. Treat everything below as
> provisional and re-run `tools/score_adjudicated.py` once real verdicts exist.

Conventions applied to the verdicts are recorded once, in
`free_texts/adjudication/MODEL_PREFILL_README.md`, not duplicated here.

## Read this before quoting anything

**The synthetic corpus produces near-degenerate disagreement strata.** The two
systems almost never disagree on these notes, so the strata that carry precision
are tiny:

| attribute | both_positive | python_only | ctakes_only | both_negative (pop → sampled) | cases |
|---|---:|---:|---:|---:|---:|
| `negated` | 103 | 95 | 3 | 1,507 → 600 | 801 |
| `history_of` | 17 | 17 | 0 | 1,674 → 600 | 634 |
| `subject` | 1 | 8 | 1 | 1,698 → 600 | 610 |
| `uncertain` | 0 | 10 | 15 | 1,683 → 600 | 625 |

`subject` rests on 10 non-`both_negative` cases and `uncertain` has no
`both_positive` stratum at all. **Neither was pre-annotated**, deliberately: a
precision figure resting on single-digit case counts reads as a measurement and
is not one. Their `review.csv` and `design.json` are regenerated and valid; the
sidecars are absent.

The cause is structural. These notes are single-patient prose with no EHR
family-history tables, and that corpus's `subject` and `history_of`
results were driven almost entirely by such tables. `negated` is the only attribute whose
strata still support the design.

## Results — 20-note synthetic corpus

1,724 mentions, 1,708 aligned, `data/umls_ctakes_16ab.sqlite`.

| attribute | vs cTAKES | vs adjudication | |
|---|---|---|---|
| `negated` | P .520 / R .972 / **F1 .678** | P .887 / R .854 / **F1 .870** | better than agreement suggests |
| `history_of` | P .500 / R 1.000 / **F1 .667** | P .460 / R .198 / **F1 .277** | **much worse** |
| `subject` | P .111 / R .500 / **F1 .182** | not adjudicated | 2 cTAKES positives |
| `uncertain` | P .000 / R .000 / **F1 .000** | not adjudicated | no overlap at all |

The two directions that corpus found survive in shape: `negated` is
undersold by the cTAKES comparison, `history_of` is flattered by it. The
magnitudes are not transferable — the vs-cTAKES column here is computed against
17 `history_of` and 2 `subject` reference positives, which is not enough to
carry a score in either direction.

**`negated` is the one usable result.** P 0.887 on 103 `both_positive` and 95
`python_only` cases adjudicated at ~100%. The Review-of-Systems-negatives
problem is real and in the pipeline's favour, and on prose notes the
over-negation that dogged that corpus is much less visible.

**`history_of` precision 0.460 is well below that corpus's 0.881**, and the reason
is a convention change rather than a regression. See the section rule below.

**33% of `negated` cases and 25% of `history_of` cases were excluded as
`unclear`.** That rate is itself a result: a quarter to a third of what the
pipeline emits on prose clinical notes is chart furniture (`Abdomen:`,
`ESTIMATED BLOOD LOSS:`, `SEX:`) or lexical noise (`air` in "room air", `today`,
`normal`, `drill`) rather than clinical content. The scores above describe
performance on the mentions that carry meaning.

## The Past Medical History section rule

The rule is off by default, rejected on agreement with cTAKES. Both references
still disagree about it in opposite directions, and on this corpus the gap is
much wider than before:

| reference | rule off | rule on |
|---|---|---|
| cTAKES agreement | P .500 / R 1.000 / **F1 .667** | P .304 / R 1.000 / **F1 .466** |
| adjudicated verdicts | P .460 / R .198 / **F1 .277** | P .742 / R .670 / **F1 .704** |

Against verdicts the rule now improves *both* precision and recall, which it did
not do on that corpus (there it traded .045 of precision for recall).

**Do not read that as evidence the rule should ship on.** It is the clearest
example in this document of a rule scored against a labeller that shares its
rule. The prefill convention says entries under `PAST MEDICAL HISTORY` are
`history` — including ones marked `(active)` — which is precisely what
`history_sections=True` implements. The variant is being rewarded for agreeing
with the assumption used to label it. **It ships off**, and a clinician's
verdicts are what would settle it.

It remains available as `ClinicalPipeline(profile="clinical_recall")`,
`--profile clinical_recall`, or `UMLSMATCH_PROFILE=clinical_recall`; the
individual `history_sections=True` switch still works and still wins over the
profile. `tools/score_variant_adjudicated.py` re-runs this on any review file.

Note that `tools/score_attributes.py` has **no** flag for this rule, so the
vs-cTAKES row above is not reproducible from a shipped command; it was computed
by calling `score_corpus` directly with `history_sections` set both ways.

## Read the recall figures with care

Precision is robust here. Recall is not, and the bootstrap interval does not
always say so.

Both adjudicated attributes sample the huge "both systems said no" stratum at
~36%, so each adjudicated positive there extrapolates to ~3.7 corpus positives —
far better than the ~65 those sets carried, because the stratum is smaller
and `--agree-negative` now defaults to 600. The intervals are correspondingly
tighter:

| attribute | recall | 95% interval | span |
|---|---:|---|---:|
| `negated` | 0.854 | 0.767 – 0.923 | 0.16 |
| `history_of` | 0.198 | 0.139 – 0.295 | 0.16 |

The printed **precision** intervals are degenerate (`[0.887, 0.887]`,
`[0.460, 0.460]`) because the strata carrying precision are sampled at ~100%.
There is no sampling variance left to bootstrap — which is not the same as
certainty, since it says nothing about whether the verdicts are right.

---

# Decisions this repository cannot re-derive

The three findings below come from a **corpus of real clinical notes that is not
distributed with this repository**. They are recorded because shipped defaults,
a renamed API field and two rejected designs rest on them, and a decision whose
evidence is absent is indistinguishable from an arbitrary one. **None of it can
be reproduced from anything on disk**, and none of it describes the synthetic
corpus. Re-establishing any of it needs real notes.

### `uncertain` was never a diagnostic; the reference was

This attribute was once called "a diagnostic, not a result", on the grounds that
too few reference positives cannot separate a working rule set from a broken
one. That reasoning is about *cTAKES' positives*, and it did not survive contact
with verdicts:

| stratum | adjudicated hedged | judged |
|---|---:|---:|
| both called it hedged | 11 | 15 |
| pipeline only | 52 | 72 |
| **cTAKES only** | **3** | **90** |
| neither | 1 | 168 |

**97% of the mentions cTAKES called uncertain were not hedged** — the standing
example is `"Taking?"` from a medication table header — while 72% of the
pipeline's solo calls were genuinely hedged. Agreement with cTAKES was measuring
cTAKES. Adjudication put precision at **0.724** with recall unestimable.

> Acted on: the README, `uncertainty.py` and `analyze.py` state the asymmetry
> directly — usable precision, unestimable recall — rather than the blanket
> "diagnostic", and `AttributeSpec.measurable` was renamed
> `measurable_against_ctakes`, because measurability turned out to be a property
> of the reference, not of the attribute. `attributes.py` cites this heading by
> name, and a test pins the caveat text so it cannot regress to the disproved
> reason.

**What this implies for `conditional` and `generic`.** A stratified adjudication
samples the *pipeline's* positives, so precision is estimable however few cTAKES
found. A rare attribute is not unmeasurable; it is unmeasurable *against
cTAKES*. `conditional` acts on that — a prototype behind
`ClinicalPipeline(conditional=True)`, with `make_adjudication_set.py --attribute
conditional` recording the switch in `design.json`. On that corpus the sampler
reported 84 pipeline positives against 27 cTAKES positives with **zero
overlap**; the review file was never adjudicated, so nothing about `conditional`
is settled — what the prototype buys is that it *can* be. **`generic` does
not**, and the reason is not the positive count: it has no external cue lexicon
to adopt, so rules for it could only be phrases invented to fit the examples.

### Header noise, and the two designs rejected for it

`ClinicalPipeline(drop_header_mentions=True)` removes the mentions that sit
inside a *recognized section heading*. It is deliberately narrow, and two wider
designs were measured and rejected first.

A **CUI blocklist** does not exist: of 709 adjudicated CUIs only three were
judged at least four times and at least 95% `unclear`. The dominant furniture
concepts are *shared* with clinical uses — `C0262926` is 71%
`Past Medical History:` and 29% the `history of` cue that drives the attribute,
and no blocklist separates those.

A **window-level rule** is worse than nothing: dropping every mention in a
header-only window removes 25% of annotations, of which **57% is real clinical
content**, because EHR table dumps routinely end a window with `:` while
carrying problem-list entries before it.

What is left uncaught is table column labels (`Problem Relation Age of Onset`),
coding markers (`HCC`, `MAR Hold`) and form fields (`Sig:`). Reaching those
needs table-structure awareness, which this pipeline does not have.

### The negation scope cap

`MAX_SCOPE_TOKENS = 8` was chosen because that corpus put the F1 optimum in a
broad plateau around 5–10 tokens. **The synthetic corpus does not reproduce
that** — there F1 falls as scope widens and peaks at 3. The disagreement is left
standing rather than resolved by retuning, because fitting a shipped default to
20 synthetic notes is the overfitting the cap exists to avoid; the sweep and the
reasoning are in `umlsmatch.assertion.negation`.

## Reproducing

```bash
python tools/run_java_ctakes.py --input-dir free_texts/synthetic \
    --output-dir free_texts/xmi_out --jsonl free_texts/json/silver.jsonl

python tools/make_adjudication_set.py --jsonl free_texts/json/silver.jsonl \
    --db data/umls_ctakes_16ab.sqlite --attribute negated
python tools/score_adjudicated.py \
    --review free_texts/adjudication/negated/review_model_prefill.csv
```

Review files quote note text and stay under `free_texts/`, which is gitignored
and hook-protected; `free_texts/synthetic/*.txt` is the one carve-out. See
CONTRIBUTING.

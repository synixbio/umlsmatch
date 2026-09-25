"""Build a human-adjudicated gold standard for any assertion attribute.

cTAKES' polarity is wrong on constructions that are common in these notes
(Review-of-Systems negatives), so scoring negation against it measures agreement
with a known-faulty oracle, and the only fix is human labels. This module makes
that affordable.

**It is not specific to polarity.** The design -- stratify by agreement, sample
the agreements, adjudicate all disagreements, reweight -- is parameterized over
an :class:`~umlsmatch.assertion.attributes.AttributeSpec`, so
``--attribute history_of`` produces the same review file for a different
question. That matters more for the other attributes than it does for negation:
``subject`` disagrees with cTAKES on EHR family-history tables where cTAKES is
internally inconsistent, and ``history_of`` misses its target against a
reference that labels 113 distinct mention texts both ways. Neither of those is
settleable against cTAKES, by anyone, ever. See
docs/ADJUDICATION_RESULTS.md.

**The idea.** Labelling every mention in the corpus is unnecessary. Where two
independently-built systems already agree, the label is very probably right;
the doubt is concentrated in the cases where they disagree. So adjudicate
every disagreement, sample the agreements, and reweight. That turns a
multi-day annotation job into a few hours without giving up a corpus-level
estimate.

Four strata, by what each side said about one aligned mention:

================  ===============  ==============  =======================
stratum           python            cTAKES          default sample
================  ===============  ==============  =======================
``both_positive`` positive          positive        150 (verify, don't assume)
``python_only``   positive          negative        **all** -- the FP candidates
``ctakes_only``   negative          positive        **all** -- the FN candidates
``both_negative`` negative          negative        600 (finds what both missed)
================  ===============  ==============  =======================

The strata were called ``both_negated``/``both_affirmed`` while this was a
polarity-only module. They are spelled neutrally now because "both_negated" is
an actively misleading name for a ``history_of`` stratum. A ``design.json``
written before the rename will not load; regenerate it, which costs a minute
and is safe because no adjudication pass has been completed.

``both_positive`` is sampled rather than assumed correct: it is the stratum most
tempting to take on faith, and an error rate there biases precision upward
invisibly. ``both_negative`` is what makes **recall** estimable at all -- a
positive both systems missed appears in no disagreement, so adjudicating only
disagreements would leave recall unmeasurable while looking complete.

**Adjudication is blind.** The review file carries the sentence, the mention and
the question being asked, and *not* what either system predicted. Showing an
annotator a machine label anchors them to it, and since the whole purpose is to
overrule a machine label, that would defeat the exercise. The sampling design --
which case sat in which stratum -- is written separately and only read back at
scoring time.

**The verdict words are the attribute's own.** "negated / affirmed" for
polarity, "family_member / patient" for subject, "history / current" for
history_of. A generic yes/no would have been less code and a worse instrument:
an annotator judging 400 rows against domain words is faster and mistypes in
ways that :func:`load_verdicts` can catch.

**Estimation is stratified, with bootstrap intervals.** Each stratum is sampled
at its own rate, so a raw count over the adjudicated cases is not a corpus
estimate; ``python_only`` is 100% sampled and ``both_affirmed`` perhaps 2%, and
averaging them directly would weight the rare stratum absurdly. Precision and
recall are both ratios of estimated totals, and recall's numerator appears in
its denominator, so the intervals come from resampling within strata rather than
a closed form. Fully-adjudicated strata are held fixed during the bootstrap --
they carry no sampling uncertainty, and resampling them would invent some.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from umlsmatch.assertion.attributes import AttributeSpec, get_attribute
from umlsmatch.eval.attributes import gold_occurrences, predicted_occurrences
from umlsmatch.eval.negation_spans import Occurrence, align_mentions
from umlsmatch.eval.records import iter_records

__all__ = [
    "BOTH_NEGATIVE",
    "BOTH_POSITIVE",
    "CTAKES_ONLY",
    "DEFAULT_SAMPLE_SIZES",
    "PYTHON_ONLY",
    "STRATA",
    "Case",
    "Design",
    "Estimate",
    "collect_cases",
    "estimate",
    "load_verdicts",
    "sample_cases",
    "write_review_csv",
]

BOTH_POSITIVE = "both_positive"
PYTHON_ONLY = "python_only"
CTAKES_ONLY = "ctakes_only"
BOTH_NEGATIVE = "both_negative"

STRATA = (BOTH_POSITIVE, PYTHON_ONLY, CTAKES_ONLY, BOTH_NEGATIVE)

#: ``None`` means "take the whole stratum".
#:
#: ``BOTH_NEGATIVE`` was 200 and is 600, because 200 turned out not to support a
#: recall figure. That stratum holds ~11,000 mentions, so each adjudicated case
#: carries a weight near 60 and one positive moves the estimate by tens of
#: corpus positives. The first full adjudication pass made the cost concrete --
#: recall intervals implied by this stratum alone, holding TP fixed:
#:
#: ==============  ==========  ============  ============
#: attribute       n=200       n=600         n=1200
#: ==============  ==========  ============  ============
#: ``negated``     0.59-0.85   0.65-0.80     0.68-0.79
#: ``subject``     0.77-1.00   0.91-1.00     0.95-1.00
#: ``history_of``  0.11-0.18   0.12-0.16     0.13-0.15
#: ``uncertain``   0.16-0.85   0.23-0.67     0.30-0.65
#: ==============  ==========  ============  ============
#:
#: 600 takes most of the tightening; 1200 costs another 600 judgements for
#: little more. The price is real -- 400 extra cases is roughly 80 minutes of
#: review per attribute -- and it buys the difference between a recall number
#: and a gesture. Precision is unaffected by this stratum either way, so an
#: adjudication that only needs precision can still pass ``--agree-negative``
#: down.
DEFAULT_SAMPLE_SIZES: dict[str, int | None] = {
    BOTH_POSITIVE: 150,
    PYTHON_ONLY: None,
    CTAKES_ONLY: None,
    BOTH_NEGATIVE: 600,
}

_REVIEW_COLUMNS = (
    "case_id",
    "document",
    "attribute",
    "question",
    "cui",
    "start",
    "end",
    "mention",
    "preferred_text",
    "sentence",
    "verdict",
    "notes",
)


@dataclass(frozen=True)
class Case:
    """One mention to adjudicate, with the context needed to judge it."""

    case_id: str
    document: str
    cui: str
    start: int
    end: int
    mention: str
    preferred_text: str
    sentence: str
    #: Which stratum this came from. Never written to the review file.
    stratum: str
    #: Which attribute is being judged. Constant per file, and written to the
    #: review file: it reveals nothing about either prediction, and a CSV that
    #: does not say what question it is asking is a CSV that gets answered for
    #: the wrong one.
    attribute: str = "negated"
    #: The question, in words. Same reasoning.
    question: str = ""

    def review_row(self) -> dict[str, object]:
        """The blind row an annotator sees -- no stratum, no predictions."""
        return {
            "case_id": self.case_id,
            "document": self.document,
            "attribute": self.attribute,
            "question": self.question,
            "cui": self.cui,
            "start": self.start,
            "end": self.end,
            "mention": self.mention,
            "preferred_text": self.preferred_text,
            "sentence": self.sentence,
            "verdict": "",
            "notes": "",
        }


@dataclass
class Design:
    """The sampling design, needed to reweight verdicts into a corpus estimate."""

    #: Stratum -> total population size in the corpus.
    population: dict[str, int] = field(default_factory=dict)
    #: Stratum -> number sampled for adjudication.
    sampled: dict[str, int] = field(default_factory=dict)
    #: case_id -> stratum.
    case_stratum: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    #: Which attribute this design adjudicates. Recorded because a review file
    #: and a design that disagree about the question would reweight one
    #: attribute's verdicts into another's estimate, and nothing downstream
    #: would notice.
    attribute: str = "negated"
    #: How the pipeline was configured for the run that produced the strata.
    #: A free-form record rather than named fields: negation has ``max_scope``,
    #: ``subject`` has three different windows, and the next attribute will
    #: have its own. What matters is that the design says what produced it.
    pipeline_options: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Design:
        """Load a design, rejecting one this code cannot reweight correctly.

        An unknown key is an error rather than something to ignore. The likely
        source is a ``design.json`` written before the strata were renamed
        (``both_negated`` -> ``both_positive``), and silently accepting it would
        produce a plausible estimate from strata that no longer line up --
        exactly the failure this whole module exists to avoid.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{path}: unrecognized design fields {sorted(unknown)}. This is "
                "probably a design written before the strata were renamed for "
                "multi-attribute support; regenerate it with "
                "tools/make_adjudication_set.py."
            )
        stale = set(raw.get("population", {})) - set(STRATA)
        if stale:
            raise ValueError(
                f"{path}: unrecognized strata {sorted(stale)}; expected "
                f"{', '.join(STRATA)}. Regenerate the design."
            )
        return cls(**raw)


@dataclass(frozen=True)
class Estimate:
    """Corpus-level precision/recall with bootstrap intervals."""

    precision: float
    precision_lo: float
    precision_hi: float
    recall: float
    recall_lo: float
    recall_hi: float
    f1: float
    n_adjudicated: int
    n_unclear: int
    #: Stratum -> (adjudicated positive, adjudicated total, population size).
    per_stratum: dict[str, tuple[int, int, int]]
    #: Which attribute was estimated. Carried so a printed estimate says what
    #: it is an estimate *of*.
    attribute: str = "negated"

    def __str__(self) -> str:  # pragma: no cover - presentation
        return (
            f"{self.attribute}: "
            f"P={self.precision:.3f} [{self.precision_lo:.3f}, {self.precision_hi:.3f}]  "
            f"R={self.recall:.3f} [{self.recall_lo:.3f}, {self.recall_hi:.3f}]  "
            f"F1={self.f1:.3f}"
        )


# --- collection --------------------------------------------------------------


def _sentences(text: str) -> list[tuple[int, int, str]]:
    """(start, end, display text) for every window, for locating a mention.

    Imported lazily via the pipeline rather than at module scope: the sentences
    are only needed to give an annotator context, and this module is otherwise
    importable without the ``nlp`` extra.
    """
    from umlsmatch.pipeline.tokenizer import annotate_sentences

    out: list[tuple[int, int, str]] = []
    for tokens in annotate_sentences(text):
        if not tokens:
            continue
        start, end = tokens[0].start, tokens[-1].end
        out.append((start, end, " ".join(text[start:end].split())))
    return out


def _sentence_for(sentences: list[tuple[int, int, str]], start: int, end: int) -> str:
    """The window containing a mention, or '' when none does."""
    for s_start, s_end, display in sentences:
        if s_start <= start and end <= s_end:
            return display
    return ""


def collect_cases(
    jsonl_path: str | Path,
    pipeline,
    attribute: str | AttributeSpec = "negated",
) -> list[Case]:
    """Every aligned mention in the corpus, tagged with its stratum.

    Alignment is :func:`~umlsmatch.eval.negation_spans.align_mentions`, so these
    cases correspond one-for-one with what
    :mod:`umlsmatch.eval.attributes` measures. Gold occurrences with no Python
    counterpart are skipped: they are a concept-recall miss, already measured by
    ``eval.parity``, and an annotator has no prediction to adjudicate.

    `pipeline` is a built :class:`~umlsmatch.analyze.ClinicalPipeline` rather
    than a dictionary path. That is the substantive change from the
    negation-only version, which re-implemented the analysis loop inline and so
    could only ever adjudicate the one attribute it had inlined. Taking the
    pipeline means the strata are built from *exactly* what the pipeline
    returns, for any attribute, with whatever options it was constructed with.
    """
    spec = attribute if isinstance(attribute, AttributeSpec) else get_attribute(attribute)
    if not spec.assessable:
        raise ValueError(
            f"{spec.name} is not assessed by this pipeline, so every prediction "
            f"would be 'not assessed' and two of the four strata would be empty. "
            f"{spec.note}"
        )
    # A prototype is assessable and off by default, so the same two strata are
    # empty unless the caller actually asked for it. Refusing here rather than
    # returning a degenerate sample is the difference between "you forgot a
    # flag" and an estimate silently reweighted from nothing. `True` is the
    # getattr default so a test double without the switch is taken at its word;
    # a real pipeline always carries the attribute.
    option = spec.pipeline_option
    if spec.prototype and option and not getattr(pipeline, option, True):
        raise ValueError(
            f"{spec.name} is a prototype and is off by default; build the "
            f"pipeline with ClinicalPipeline({option}=True) or every prediction "
            f"here would be 'not assessed'."
        )

    cases: list[Case] = []
    for record in iter_records(jsonl_path):
        doc = record["source_file"]
        text = record["text"]

        annotations = pipeline.analyze(text)
        predicted: list[Occurrence] = predicted_occurrences(annotations, spec)
        preferred = {a.cui: a.preferred_text for a in annotations}
        sentences = _sentences(text)

        pairs, _ = align_mentions(gold_occurrences(record, spec), predicted)

        for gold, pred in pairs:
            p_start, p_end, cui, py_positive = pred
            gold_positive = gold[3]
            if py_positive and gold_positive:
                stratum = BOTH_POSITIVE
            elif py_positive:
                stratum = PYTHON_ONLY
            elif gold_positive:
                stratum = CTAKES_ONLY
            else:
                stratum = BOTH_NEGATIVE

            cases.append(
                Case(
                    # Deterministic and content-free, so re-running the
                    # extractor reproduces ids and partially-completed
                    # review files stay joinable. The attribute is part of it
                    # so two attributes' review files cannot collide.
                    case_id=f"{spec.name}:{doc}:{p_start}:{p_end}:{cui}",
                    document=doc,
                    cui=cui,
                    start=p_start,
                    end=p_end,
                    mention=text[p_start:p_end],
                    preferred_text=preferred.get(cui, ""),
                    sentence=_sentence_for(sentences, p_start, p_end),
                    stratum=stratum,
                    attribute=spec.name,
                    question=spec.question,
                )
            )
    return cases


# --- sampling ----------------------------------------------------------------


def sample_cases(
    cases: Iterable[Case],
    sizes: dict[str, int | None] | None = None,
    *,
    seed: int = 0,
    attribute: str | None = None,
    pipeline_options: dict | None = None,
) -> tuple[list[Case], Design]:
    """Draw the adjudication sample and record the design that produced it.

    Sampling is seeded so the same corpus yields the same review file -- a
    reviewer who is half-way through should not have the set reshuffled under
    them by a re-run.

    `attribute` defaults to whatever the cases carry, which is what
    :func:`collect_cases` set. Passing it explicitly is only useful when
    building a design by hand, as the tests do.
    """
    cases = list(cases)
    sizes = dict(DEFAULT_SAMPLE_SIZES if sizes is None else sizes)
    by_stratum: dict[str, list[Case]] = {s: [] for s in STRATA}
    for case in cases:
        by_stratum[case.stratum].append(case)

    if attribute is None:
        attributes = {c.attribute for c in cases}
        if len(attributes) > 1:
            raise ValueError(
                f"cases mix attributes {sorted(attributes)}; one design "
                "reweights one attribute, and mixing them would average two "
                "different questions into one estimate"
            )
        attribute = attributes.pop() if attributes else "negated"

    rng = random.Random(seed)
    design = Design(
        seed=seed, attribute=attribute, pipeline_options=dict(pipeline_options or {})
    )
    sampled: list[Case] = []

    for stratum in STRATA:
        pool = sorted(by_stratum[stratum], key=lambda c: c.case_id)
        want = sizes.get(stratum)
        take = pool if want is None or want >= len(pool) else rng.sample(pool, want)
        take = sorted(take, key=lambda c: (c.document, c.start, c.cui))

        design.population[stratum] = len(pool)
        design.sampled[stratum] = len(take)
        for case in take:
            design.case_stratum[case.case_id] = stratum
        sampled.extend(take)

    return sampled, design


def write_review_csv(cases: Sequence[Case], path: str | Path) -> None:
    """Write the blind review file.

    UTF-8 BOM and CRLF: the realistic tool here is Excel, which mis-renders
    accented characters in UTF-8 without a BOM and is the reason the encoding
    looks over-specified.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REVIEW_COLUMNS)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.review_row())


# --- reading verdicts back ---------------------------------------------------


def load_verdicts(
    path: str | Path, attribute: str | AttributeSpec | None = None
) -> dict[str, str]:
    """Read ``case_id -> verdict`` from a filled review file, skipping blanks.

    Returns the *normalized* verdict: the attribute's positive label, its
    negative label, or ``"unclear"``.

    Raises on an unrecognized verdict rather than silently dropping it. A typo
    like "negatd" would otherwise quietly shrink the sample and bias the
    estimate by however many rows the annotator fat-fingered -- and with
    per-attribute verdict words there is a second way to get it wrong, which is
    to answer a ``history_of`` file with "negated". Both fail loudly.

    `attribute` defaults to whatever the file's ``attribute`` column says, so
    the caller normally need not pass it. A file without that column is read as
    ``negated``, which is what the pre-generalization writer produced.
    """
    out: dict[str, str] = {}
    path = Path(path)
    spec = (
        attribute
        if isinstance(attribute, AttributeSpec)
        else get_attribute(attribute) if attribute else None
    )

    with path.open(encoding="utf-8-sig", newline="") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), 2):
            if spec is None:
                spec = get_attribute((row.get("attribute") or "negated").strip())
            verdict = (row.get("verdict") or "").strip().lower()
            if not verdict:
                continue
            if verdict not in spec.verdicts:
                raise ValueError(
                    f"{path}:{lineno}: verdict {verdict!r} is not one of "
                    f"{', '.join(spec.verdicts)} (attribute {spec.name})"
                )
            case_id = (row.get("case_id") or "").strip()
            if case_id:
                out[case_id] = verdict
    return out


# --- estimation --------------------------------------------------------------


def _stratum_counts(
    design: Design, verdicts: dict[str, str]
) -> tuple[dict[str, list[int]], int]:
    """Stratum -> [positive, total adjudicated]; plus the unclear count.

    "Positive" is the attribute's own positive label -- "negated" for polarity,
    "family_member" for subject. Read off the design rather than hard-coded,
    which is the whole of what generalizing this module required here.
    """
    positive = get_attribute(design.attribute).positive_label
    counts = {s: [0, 0] for s in STRATA}
    unclear = 0
    for case_id, verdict in verdicts.items():
        stratum = design.case_stratum.get(case_id)
        if stratum is None:
            continue
        if verdict == "unclear":
            unclear += 1
            continue
        counts[stratum][1] += 1
        if verdict == positive:
            counts[stratum][0] += 1
    return counts, unclear


def _pr_from_rates(design: Design, rates: dict[str, float]) -> tuple[float, float]:
    """Precision and recall from per-stratum true-positive rates."""
    n = design.population
    # System-flagged positive: both_positive + python_only, whatever the truth.
    flagged = n.get(BOTH_POSITIVE, 0) + n.get(PYTHON_ONLY, 0)
    true_positives = (
        n.get(BOTH_POSITIVE, 0) * rates.get(BOTH_POSITIVE, 0.0)
        + n.get(PYTHON_ONLY, 0) * rates.get(PYTHON_ONLY, 0.0)
    )
    # Everything truly positive, including what the system called negative.
    missed = (
        n.get(CTAKES_ONLY, 0) * rates.get(CTAKES_ONLY, 0.0)
        + n.get(BOTH_NEGATIVE, 0) * rates.get(BOTH_NEGATIVE, 0.0)
    )
    precision = true_positives / flagged if flagged else 0.0
    denom = true_positives + missed
    recall = true_positives / denom if denom else 0.0
    return precision, recall


def estimate(
    design: Design,
    verdicts: dict[str, str],
    *,
    bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Estimate:
    """Corpus-level precision/recall from a stratified sample of verdicts.

    Intervals come from resampling within each stratum. A stratum adjudicated
    in full is held fixed: its rate is known, not estimated, and bootstrapping
    it would report uncertainty that does not exist.
    """
    counts, unclear = _stratum_counts(design, verdicts)
    rates = {
        s: (counts[s][0] / counts[s][1] if counts[s][1] else 0.0) for s in STRATA
    }
    precision, recall = _pr_from_rates(design, rates)

    rng = random.Random(seed)
    p_samples: list[float] = []
    r_samples: list[float] = []
    for _ in range(max(0, bootstrap)):
        draw: dict[str, float] = {}
        for s in STRATA:
            positive, total = counts[s]
            if not total:
                draw[s] = 0.0
                continue
            # Census of the stratum -> no sampling error to simulate.
            if design.sampled.get(s, 0) >= design.population.get(s, 0) > 0:
                draw[s] = positive / total
                continue
            resampled = sum(1 for _ in range(total) if rng.random() < positive / total)
            draw[s] = resampled / total
        p, r = _pr_from_rates(design, draw)
        p_samples.append(p)
        r_samples.append(r)

    def interval(values: list[float], point: float) -> tuple[float, float]:
        if not values:
            return (point, point)
        values = sorted(values)
        tail = (1 - confidence) / 2
        lo = values[min(int(tail * len(values)), len(values) - 1)]
        hi = values[min(int((1 - tail) * len(values)), len(values) - 1)]
        return (lo, hi)

    p_lo, p_hi = interval(p_samples, precision)
    r_lo, r_hi = interval(r_samples, recall)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return Estimate(
        precision=precision,
        precision_lo=p_lo,
        precision_hi=p_hi,
        recall=recall,
        recall_lo=r_lo,
        recall_hi=r_hi,
        f1=f1,
        n_adjudicated=sum(c[1] for c in counts.values()),
        n_unclear=unclear,
        per_stratum={
            s: (counts[s][0], counts[s][1], design.population.get(s, 0)) for s in STRATA
        },
        attribute=design.attribute,
    )

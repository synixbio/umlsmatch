"""Tests for the adjudication sampling design and stratified estimator.

The estimator is the load-bearing part: strata are sampled at wildly different
rates (disagreements at 100%, agreements at ~2%), so a reweighting bug produces
a plausible-looking number that is simply wrong. These construct designs with
known answers and check the arithmetic, plus the blindness and round-trip
properties the method depends on.

No dictionary or spaCy model needed -- ``collect_cases`` is the only part that
touches those and it is covered by the corpus-level tools.

The estimator is exercised on ``negated`` throughout, and separately checked to
be attribute-agnostic. That split is deliberate: the arithmetic tests below
carry known answers worked out by hand for polarity, and rewriting them per
attribute would multiply the surface without testing anything new -- what the
generalization can break is the mapping from a verdict *word* to a positive,
and that gets its own tests.
"""

from __future__ import annotations

import csv

import pytest

from umlsmatch.assertion.attributes import get_attribute
from umlsmatch.eval.adjudication import (
    BOTH_NEGATIVE,
    BOTH_POSITIVE,
    CTAKES_ONLY,
    DEFAULT_SAMPLE_SIZES,
    PYTHON_ONLY,
    Case,
    Design,
    estimate,
    load_verdicts,
    sample_cases,
    write_review_csv,
)

POSITIVE = "negated"
NEGATIVE = "affirmed"


def _case(i, stratum, doc="a.txt", attribute="negated"):
    spec = get_attribute(attribute)
    return Case(
        case_id=f"{attribute}:{doc}:{i}:{i + 5}:C{i}",
        document=doc,
        cui=f"C{i}",
        start=i,
        end=i + 5,
        mention="chest pain",
        preferred_text="Chest Pain",
        sentence="Patient denies chest pain.",
        stratum=stratum,
        attribute=attribute,
        question=spec.question,
    )


def _design(population, sampled, case_stratum, attribute="negated"):
    return Design(
        population=population,
        sampled=sampled,
        case_stratum=case_stratum,
        attribute=attribute,
    )


def _set_verdicts(path, verdicts):
    """Fill the verdict column in order, the way an annotator would."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row, verdict in zip(rows, verdicts, strict=False):
        row["verdict"] = verdict
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- blindness ---------------------------------------------------------------


def test_review_row_hides_the_stratum():
    """Stratum leaks both predictions -- 'python_only' means python said negated."""
    row = _case(1, PYTHON_ONLY).review_row()
    assert "stratum" not in row
    assert PYTHON_ONLY not in str(row.values())


def test_review_csv_has_no_prediction_columns(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv([_case(1, PYTHON_ONLY), _case(2, CTAKES_ONLY)], path)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert "verdict" in header
    for leaky in ("stratum", "python_says", "ctakes_says", POSITIVE):
        assert leaky not in header


def test_review_csv_round_trips_a_verdict(tmp_path):
    path = tmp_path / "review.csv"
    case = _case(1, PYTHON_ONLY)
    write_review_csv([case], path)

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["verdict"] = POSITIVE
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    assert load_verdicts(path) == {case.case_id: POSITIVE}


# --- sampling ----------------------------------------------------------------


def test_disagreements_are_taken_in_full_by_default():
    cases = (
        [_case(i, PYTHON_ONLY) for i in range(300)]
        + [_case(1000 + i, CTAKES_ONLY) for i in range(90)]
        + [_case(2000 + i, BOTH_POSITIVE) for i in range(400)]
        + [_case(3000 + i, BOTH_NEGATIVE) for i in range(5000)]
    )
    _, design = sample_cases(cases)
    assert design.sampled[PYTHON_ONLY] == 300
    assert design.sampled[CTAKES_ONLY] == 90
    # Agreements are sampled, not censused.
    assert design.sampled[BOTH_POSITIVE] == 150
    assert design.sampled[BOTH_NEGATIVE] == DEFAULT_SAMPLE_SIZES[BOTH_NEGATIVE]


def test_population_records_the_whole_corpus_not_the_sample():
    cases = [_case(i, BOTH_NEGATIVE) for i in range(5000)]
    _, design = sample_cases(cases)
    assert design.population[BOTH_NEGATIVE] == 5000
    assert design.sampled[BOTH_NEGATIVE] == DEFAULT_SAMPLE_SIZES[BOTH_NEGATIVE]


def test_sampling_is_deterministic_for_a_seed():
    """A reviewer half-way through must not have the set reshuffled by a re-run."""
    cases = [_case(i, BOTH_NEGATIVE) for i in range(1000)]
    first, _ = sample_cases(cases, seed=7)
    second, _ = sample_cases(cases, seed=7)
    assert [c.case_id for c in first] == [c.case_id for c in second]


def test_a_stratum_smaller_than_its_quota_is_taken_whole():
    cases = [_case(i, BOTH_POSITIVE) for i in range(10)]
    sampled, design = sample_cases(cases)
    assert len(sampled) == 10
    assert design.sampled[BOTH_POSITIVE] == 10


def test_design_round_trips_through_json(tmp_path):
    cases = [_case(i, PYTHON_ONLY) for i in range(5)]
    _, design = sample_cases(cases)
    path = tmp_path / "design.json"
    design.to_json(path)
    assert Design.from_json(path).population == design.population


# --- verdict loading ---------------------------------------------------------


def test_blank_verdicts_are_skipped(tmp_path):
    path = tmp_path / "r.csv"
    write_review_csv([_case(1, PYTHON_ONLY), _case(2, PYTHON_ONLY)], path)
    assert load_verdicts(path) == {}


def test_unknown_verdict_raises_rather_than_being_dropped(tmp_path):
    """A typo must not silently shrink the sample and bias the estimate."""
    path = tmp_path / "r.csv"
    write_review_csv([_case(1, PYTHON_ONLY)], path)
    _set_verdicts(path, ["negatd"])
    with pytest.raises(ValueError, match="negatd"):
        load_verdicts(path)


# --- estimation --------------------------------------------------------------


def test_perfect_system_scores_one():
    """Everything flagged is truly negated; nothing missed."""
    design = _design(
        population={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 900},
        sampled={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 900},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(100)}
        | {f"a{i}": BOTH_NEGATIVE for i in range(900)},
    )
    verdicts = {f"n{i}": POSITIVE for i in range(100)}
    verdicts |= {f"a{i}": NEGATIVE for i in range(900)}
    est = estimate(design, verdicts, bootstrap=0)
    assert est.precision == 1.0
    assert est.recall == 1.0


def test_precision_reweights_by_population_not_sample_size():
    """The whole point: a 100%-sampled stratum must not outvote a 2% one.

    both_negated is 1000 strong but sampled at 100; python_only is 100 strong
    and censused. Judging every sampled case correctly, precision is
    1000/(1000+100) = 0.909 -- not the 100/200 = 0.5 a raw count would give.
    """
    design = _design(
        population={BOTH_POSITIVE: 1000, PYTHON_ONLY: 100, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 100, PYTHON_ONLY: 100, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(100)}
        | {f"p{i}": PYTHON_ONLY for i in range(100)},
    )
    verdicts = {f"n{i}": POSITIVE for i in range(100)}
    verdicts |= {f"p{i}": NEGATIVE for i in range(100)}
    est = estimate(design, verdicts, bootstrap=0)
    assert round(est.precision, 3) == 0.909
    assert est.recall == 1.0


def test_recall_counts_negations_both_systems_missed():
    """Extrapolated from the both_affirmed sample -- the reason it is sampled.

    20 of 200 sampled both_affirmed cases are truly negated, so ~10% of the
    9800-strong stratum is a miss neither system saw: ~980 of them.
    Recall = 100 / (100 + 980) = 0.093.
    """
    design = _design(
        population={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 9800},
        sampled={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 200},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(100)}
        | {f"a{i}": BOTH_NEGATIVE for i in range(200)},
    )
    verdicts = {f"n{i}": POSITIVE for i in range(100)}
    verdicts |= {f"a{i}": (POSITIVE if i < 20 else NEGATIVE) for i in range(200)}
    est = estimate(design, verdicts, bootstrap=0)
    assert round(est.recall, 3) == 0.093


def test_human_can_overrule_ctakes_in_our_favour():
    """The known case: cTAKES called a ROS negative affirmed, we called it negated.

    Every python_only case adjudicated 'negated' means the system was right and
    cTAKES was wrong -- precision should be 1.0, not the ~0.5 measured against
    cTAKES.
    """
    design = _design(
        population={BOTH_POSITIVE: 700, PYTHON_ONLY: 700, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 100, PYTHON_ONLY: 700, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(100)}
        | {f"p{i}": PYTHON_ONLY for i in range(700)},
    )
    verdicts = {f"n{i}": POSITIVE for i in range(100)}
    verdicts |= {f"p{i}": POSITIVE for i in range(700)}
    est = estimate(design, verdicts, bootstrap=0)
    assert est.precision == 1.0


def test_unclear_verdicts_are_excluded_and_counted():
    design = _design(
        population={BOTH_POSITIVE: 10, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 10, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(10)},
    )
    verdicts = {f"n{i}": ("unclear" if i < 3 else POSITIVE) for i in range(10)}
    est = estimate(design, verdicts, bootstrap=0)
    assert est.n_unclear == 3
    assert est.n_adjudicated == 7
    assert est.precision == 1.0


def test_a_censused_stratum_gets_no_bootstrap_uncertainty():
    """Its rate is known, not estimated; inventing an interval would mislead."""
    design = _design(
        population={BOTH_POSITIVE: 0, PYTHON_ONLY: 50, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 0, PYTHON_ONLY: 50, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"p{i}": PYTHON_ONLY for i in range(50)},
    )
    verdicts = {f"p{i}": (POSITIVE if i < 30 else NEGATIVE) for i in range(50)}
    est = estimate(design, verdicts, bootstrap=500)
    assert est.precision_lo == est.precision == est.precision_hi


def test_a_sampled_stratum_does_get_an_interval():
    design = _design(
        population={BOTH_POSITIVE: 0, PYTHON_ONLY: 5000, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 0, PYTHON_ONLY: 100, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"p{i}": PYTHON_ONLY for i in range(100)},
    )
    verdicts = {f"p{i}": (POSITIVE if i < 60 else NEGATIVE) for i in range(100)}
    est = estimate(design, verdicts, bootstrap=1000)
    assert est.precision_lo < est.precision < est.precision_hi


def test_verdicts_for_unknown_cases_are_ignored():
    """A stale review file must not corrupt a newer design."""
    design = _design(
        population={BOTH_POSITIVE: 10, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 10, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(10)},
    )
    verdicts = {f"n{i}": POSITIVE for i in range(10)} | {"ghost": POSITIVE}
    assert estimate(design, verdicts, bootstrap=0).n_adjudicated == 10


# --- the generalization beyond polarity --------------------------------------


def test_the_review_file_says_what_question_it_is_asking():
    """A CSV that does not name its question gets answered for another one."""
    row = _case(1, PYTHON_ONLY, attribute="subject").review_row()
    assert row["attribute"] == "subject"
    assert "relative" in row["question"].casefold()
    # Still blind: naming the question reveals nothing about either prediction.
    assert "stratum" not in row
    assert PYTHON_ONLY not in str(row.values())


def test_verdict_words_are_the_attributes_own():
    assert get_attribute("subject").verdicts == ("family_member", "patient", "unclear")
    assert get_attribute("history_of").verdicts == ("history", "current", "unclear")
    assert get_attribute("negated").verdicts == ("negated", "affirmed", "unclear")


def test_a_verdict_from_the_wrong_attribute_is_rejected(tmp_path):
    """Answering a subject file with "negated" is the new way to get this wrong."""
    path = tmp_path / "review.csv"
    write_review_csv([_case(1, PYTHON_ONLY, attribute="subject")], path)
    _set_verdicts(path, ["negated"])
    with pytest.raises(ValueError, match="subject"):
        load_verdicts(path)


def test_the_attribute_is_read_from_the_file(tmp_path):
    path = tmp_path / "review.csv"
    case = _case(1, PYTHON_ONLY, attribute="subject")
    write_review_csv([case], path)
    _set_verdicts(path, ["family_member"])
    assert load_verdicts(path) == {case.case_id: "family_member"}


def test_the_estimator_counts_the_attributes_positive_label():
    """The one thing generalizing the estimator could have broken.

    A design for `subject` whose verdicts all say "family_member" must read as
    100% positive. Counting the literal string "negated" would read it as 0%
    and report a precision of zero for a perfect system.
    """
    design = _design(
        population={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        sampled={BOTH_POSITIVE: 100, PYTHON_ONLY: 0, CTAKES_ONLY: 0, BOTH_NEGATIVE: 0},
        case_stratum={f"n{i}": BOTH_POSITIVE for i in range(100)},
        attribute="subject",
    )
    verdicts = {f"n{i}": "family_member" for i in range(100)}
    est = estimate(design, verdicts, bootstrap=0)
    assert est.precision == 1.0
    assert est.attribute == "subject"


def test_a_design_carries_its_attribute_through_json(tmp_path):
    cases = [_case(i, PYTHON_ONLY, attribute="history_of") for i in range(5)]
    _, design = sample_cases(cases)
    assert design.attribute == "history_of"
    path = tmp_path / "design.json"
    design.to_json(path)
    assert Design.from_json(path).attribute == "history_of"


def test_mixed_attributes_in_one_design_are_rejected():
    """One design reweights one question; averaging two is meaningless."""
    cases = [
        _case(1, PYTHON_ONLY, attribute="subject"),
        _case(2, PYTHON_ONLY, attribute="history_of"),
    ]
    with pytest.raises(ValueError, match="mix attributes"):
        sample_cases(cases)


def test_a_pre_rename_design_is_rejected_rather_than_misread(tmp_path):
    """`both_negated` no longer exists; loading it would reweight empty strata."""
    import json

    path = tmp_path / "design.json"
    path.write_text(
        json.dumps(
            {
                "population": {"both_negated": 10, "both_affirmed": 5},
                "sampled": {"both_negated": 10, "both_affirmed": 5},
                "case_stratum": {},
                "seed": 0,
                "max_scope": 8,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"(?i)regenerate"):
        Design.from_json(path)


def test_the_design_records_how_the_pipeline_was_configured():
    cases = [_case(i, PYTHON_ONLY) for i in range(3)]
    _, design = sample_cases(cases, pipeline_options={"max_scope": 4})
    assert design.pipeline_options == {"max_scope": 4}


def test_empty_verdicts_do_not_raise():
    design = _design({s: 0 for s in (BOTH_POSITIVE, PYTHON_ONLY, CTAKES_ONLY, BOTH_NEGATIVE)},
                     {s: 0 for s in (BOTH_POSITIVE, PYTHON_ONLY, CTAKES_ONLY, BOTH_NEGATIVE)}, {})
    est = estimate(design, {}, bootstrap=0)
    assert est.precision == 0.0
    assert est.recall == 0.0
    assert est.f1 == 0.0

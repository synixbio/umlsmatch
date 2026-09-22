"""The assertion-attribute registry, and the three-state contract it implies.

The load-bearing claim is that an unassessed attribute is ``None`` and not
``False``. It is easy to break by accident -- one ``bool()`` coercion, one
``or False`` default -- and the breakage is invisible: the field still holds a
plausible value, it just now asserts something nobody measured. These pin it at
every surface that carries an annotation out of the process.
"""

from __future__ import annotations

from dataclasses import fields

from umlsmatch.analyze import Annotation
from umlsmatch.assertion.attributes import (
    ATTRIBUTES,
    SUBJECT_FAMILY_MEMBER,
    attribute_names,
    get_attribute,
)

#: The five attributes whose value may be "not assessed".
NULLABLE = ("subject", "history_of", "uncertain", "conditional", "generic")


def _annotation(**kwargs) -> Annotation:
    base = dict(cui="C0008031", text="chest pain", start=0, end=10,
                group="DISORDER", negated=False)
    return Annotation(**{**base, **kwargs})


# --- the registry ------------------------------------------------------------


def test_registry_covers_the_six_ctakes_attributes():
    assert len(ATTRIBUTES) == 6
    assert {a.ctakes_name for a in ATTRIBUTES} == {
        "polarity", "subject", "historyOf", "uncertainty", "generic", "conditional"
    }


def test_every_attribute_names_an_annotation_field():
    """The registry is only useful if `name` can be `getattr`'d off a result."""
    annotation_fields = {f.name for f in fields(Annotation)}
    for spec in ATTRIBUTES:
        assert spec.name in annotation_fields, (
            f"{spec.name!r} is in the registry but not on Annotation"
        )


def test_unsupported_attributes_are_the_corpus_blocked_ones():
    """generic and conditional are not shipped on, deliberately."""
    unsupported = {a.name for a in ATTRIBUTES if not a.supported}
    assert unsupported == {"generic", "conditional"}
    assert attribute_names(supported_only=True) == (
        "negated", "subject", "history_of", "uncertain"
    )


def test_conditional_is_a_prototype_and_generic_is_not():
    """The third state: rules exist, they are off, nothing is claimed for them.

    ``supported`` and ``prototype`` must not drift into meaning the same thing.
    A prototype is measurable-by-adjudication and unpublishable; ``generic`` is
    neither, because no rules for it exist at all.
    """
    conditional = get_attribute("conditional")
    assert conditional.supported is False
    assert conditional.prototype is True
    assert conditional.assessable is True
    assert conditional.pipeline_option == "conditional"

    generic = get_attribute("generic")
    assert generic.prototype is False
    assert generic.assessable is False
    assert generic.pipeline_option is None


def test_a_prototype_is_never_scored_by_default():
    """`--attribute all` fills a results table, and a prototype has no result."""
    from umlsmatch.eval.attributes import scorable_attributes

    assert "conditional" not in {s.name for s in scorable_attributes()}


def test_every_pipeline_option_names_a_real_constructor_argument():
    """The registry claims a keyword exists on ClinicalPipeline; check it does.

    A stale name here would make ``make_adjudication_set.py`` build a pipeline
    with the switch *off* and sample a stratum of "not assessed" -- or crash,
    depending on the typo. Introspected rather than hard-coded so a rename of
    either side fails here.
    """
    import inspect

    from umlsmatch.analyze import ClinicalPipeline

    accepted = set(inspect.signature(ClinicalPipeline.__init__).parameters)
    for spec in ATTRIBUTES:
        if spec.pipeline_option is not None:
            assert spec.pipeline_option in accepted, spec.name


def test_uncertain_ships_but_its_ctakes_figure_is_not_decidable():
    """The flag is about the *reference*, not the attribute.

    `uncertain` is assessed by a default pipeline and has a usable adjudicated
    precision (0.724); what it does not have is a cTAKES agreement figure worth
    deciding anything from, because 97% of cTAKES' positives are not hedged.
    Those are separate claims and the field name now says which one it makes.
    """
    spec = get_attribute("uncertain")
    assert spec.supported is True
    assert spec.measurable_against_ctakes is False


def test_the_unmeasurable_caveat_does_not_blame_the_positive_count():
    """The reason has to stay correct, not just the conclusion.

    docs/ADJUDICATION_RESULTS.md disproved "too few positives" as the reason
    `uncertain`'s cTAKES score is worthless -- the reference is 97% noise, which
    is a fact about cTAKES and not about the sample size. The two diagnoses
    point at different remedies (a bigger corpus versus adjudication), and only
    one of them worked, so printing the old reason would send the next person
    the wrong way.
    """
    from umlsmatch.eval.attributes import AggregateAttributeScore

    agg = AggregateAttributeScore(
        n_docs=20,
        micro_precision=0.0, micro_recall=0.0, micro_f1=0.0,
        macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
        attribute="uncertain", n_aligned=1_708, n_gold_positives=15,
    )
    caveat = agg.caveat()

    assert "Do not gate on it" in caveat, "the conclusion must survive"
    assert "positive count" not in caveat, "the disproved reason must not"
    assert "ADJUDICATION_RESULTS" in caveat, "point at what can be decided from"


def test_corpus_positives_match_the_silver_standard():
    """`corpus_positives` is the raw count, and this is what makes it so.

    The published P/R/F1 are computed over the *aligned* subset (1,708 of
    1,724 mentions), so the two counts differ -- `uncertain` is 16 here and
    15 there. Both are correct and they answer different questions. This test
    pins the one this module claims to hold; without it, "16 vs 15" reads as
    a typo in whichever document you happen to open second.

    The denominator comes from `_CORPUS_MENTIONS` rather than a second literal.
    A copy here would have to be found and edited by hand whenever the corpus
    changes; reading the registry's own constant means the corpus and the
    registry can disagree, but the test and the registry cannot.
    """
    import json
    from pathlib import Path

    from umlsmatch.assertion.attributes import _CORPUS_MENTIONS

    silver = Path(__file__).resolve().parent.parent / "free_texts/json/silver.jsonl"
    if not silver.is_file():
        import pytest

        pytest.skip("silver standard not present (it embeds PHI and is gitignored)")

    counts = dict.fromkeys(attribute_names(), 0)
    total = 0
    for line in silver.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for mention in json.loads(line)["mentions"]:
            total += 1
            for spec in ATTRIBUTES:
                if spec.gold(mention):
                    counts[spec.name] += 1

    assert total == _CORPUS_MENTIONS, (
        f"the corpus changed ({total:,} mentions, registry says "
        f"{_CORPUS_MENTIONS:,}); re-derive _CORPUS_MENTIONS and every "
        f"corpus_positives"
    )
    for spec in ATTRIBUTES:
        assert counts[spec.name] == spec.corpus_positives, (
            f"{spec.name}: registry says {spec.corpus_positives}, "
            f"corpus holds {counts[spec.name]}"
        )


def test_unknown_attribute_names_the_alternatives():
    try:
        get_attribute("historyOf")  # the cTAKES spelling, not ours
    except KeyError as exc:
        assert "history_of" in str(exc)
    else:
        raise AssertionError("an unknown attribute must raise")


def test_gold_reduces_a_silver_mention_to_a_boolean():
    mention = {
        "negated": True,
        "subject": SUBJECT_FAMILY_MEMBER,
        "history_of": True,
        "uncertain": False,
        "conditional": False,
        "generic": False,
    }
    assert get_attribute("negated").gold(mention) is True
    assert get_attribute("subject").gold(mention) is True
    assert get_attribute("history_of").gold(mention) is True
    assert get_attribute("uncertain").gold(mention) is False


def test_subject_other_is_not_scored_as_family_member():
    """'other' occurs once in 12,592 mentions and supports no claim."""
    assert get_attribute("subject").gold({"subject": "other"}) is False
    assert get_attribute("subject").gold({"subject": "patient"}) is False


def test_gold_tolerates_a_mention_missing_the_key():
    """A silver file without `history_of` is not an error -- older exports lack it."""
    for spec in ATTRIBUTES:
        assert spec.gold({}) is False


# --- the three-state contract ------------------------------------------------


def test_unassessed_attributes_default_to_none_not_false():
    a = _annotation()
    for name in NULLABLE:
        assert getattr(a, name) is None, (
            f"{name} defaults to {getattr(a, name)!r}; 'not assessed' must be None, "
            "since False is a claim nobody measured"
        )


def test_negated_stays_a_plain_bool():
    """Polarity is always assessed, so it is exempt from the three-state rule."""
    assert _annotation().negated is False
    negated = next(f for f in fields(Annotation) if f.name == "negated")
    assert negated.type == "bool"


def test_to_dict_preserves_none():
    d = _annotation(history_of=None, uncertain=True).to_dict()
    assert d["history_of"] is None
    assert d["uncertain"] is True
    assert "conditional" in d, "an unassessed attribute is still reported"


def test_conditional_and_generic_are_not_silently_defaulted_to_false():
    """A decision, not an omission: these stay None even when set-able.

    The dataclass does accept them -- a caller with their own classifier can
    fill them in, and ``ClinicalPipeline(conditional=True)`` now does -- but
    the default must never quietly become False. That is the claim a consumer
    branching on ``None`` depends on, and it is unchanged by the prototype:
    off still means unassessed, not absent.
    """
    assert _annotation().conditional is None
    assert _annotation().generic is None
    assert _annotation(conditional=True).conditional is True

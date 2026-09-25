"""The six cTAKES assertion attributes, and what this pipeline claims about each.

cTAKES writes six attributes on every ``IdentifiedAnnotation``: polarity,
uncertainty, subject, generic, conditional and historyOf. This module is the
one place that names them, maps each to its spelling on both sides of the
silver-standard diff, and records whether umlsmatch assesses it at all.

**Why a registry rather than six hard-coded field names.** Three consumers need
to iterate over the attributes rather than mention them individually: the
adjudication harness (:mod:`umlsmatch.eval.adjudication`), which is
parameterized over the attribute under review; the attribute scorer
(:mod:`umlsmatch.eval.attributes`); and the service schema, which has to keep
``AnnotationOut`` in step with :class:`~umlsmatch.analyze.Annotation`. Spelling
the list once means adding an attribute cannot half-land.

**Three spellings for the same idea.** cTAKES serializes polarity as ``1``/``-1``,
uncertainty and historyOf as ``0``/``1``, and conditional and generic as
``true``/``false``. :mod:`umlsmatch.silver.xmi` normalizes all of them to
booleans; :attr:`AttributeSpec.silver_key` is the key they land under in the
JSONL, which is not always the cTAKES attribute name (``polarity`` arrives as
``negated``, ``historyOf`` as ``history_of``).

**Subject is not a boolean and is treated as one here anyway.** cTAKES' subject
takes ``patient``, ``family_member``, ``other`` and a few rarer values. On the
20-note corpus ``other`` does **not occur at all** in 1,724 mentions (1,722
``patient``, 2 ``family_member``), so the only distinction the corpus can
support is patient vs. family member, and that is the one
:attr:`AttributeSpec.gold` reduces it to. The full string is still carried
on :class:`~umlsmatch.analyze.Annotation` -- the reduction is a scoring
convenience, not a claim that the other values do not exist.

**Not assessed is spelled ``None``.** See :class:`~umlsmatch.analyze.Annotation`
for the reasoning; ``supported=False`` here is what predicts a ``None`` from a
default pipeline.

**Three states, not two.** ``supported`` answers "does a default pipeline
assess this"; :attr:`AttributeSpec.prototype` answers "do rules for it exist at
all". ``conditional`` is the case that needs both: rules exist
(:mod:`umlsmatch.assertion.conditional`), they are off unless asked for, and
nothing about them is a published capability until an adjudication says so.
Tooling that *measures* gates on :attr:`AttributeSpec.assessable`; tooling that
*reports* gates on ``supported``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "ATTRIBUTES",
    "ATTRIBUTES_BY_NAME",
    "SUBJECT_FAMILY_MEMBER",
    "SUBJECT_OTHER",
    "SUBJECT_PATIENT",
    "AttributeSpec",
    "attribute_names",
    "get_attribute",
]

#: cTAKES' ``subject`` values, as they appear in the XMI. The corpus carries
#: 12,032 ``patient``, 559 ``family_member`` and a single ``other``.
SUBJECT_PATIENT = "patient"
SUBJECT_FAMILY_MEMBER = "family_member"
SUBJECT_OTHER = "other"


@dataclass(frozen=True)
class AttributeSpec:
    """One assertion attribute, on both sides of the silver-standard diff."""

    #: Attribute name, which is also the :class:`~umlsmatch.analyze.Annotation`
    #: field and the ``--attribute`` value on the command line.
    name: str
    #: The attribute's name in cTAKES' XMI, for error messages that have to
    #: point at the source.
    ctakes_name: str
    #: Key this attribute lands under in the silver JSONL. Differs from
    #: ``ctakes_name`` wherever :mod:`umlsmatch.silver.xmi` normalized it.
    silver_key: str
    #: Whether a default :class:`~umlsmatch.analyze.ClinicalPipeline` assesses
    #: this attribute. ``False`` means the pipeline reports ``None`` and no
    #: scorer should pretend otherwise. See :attr:`prototype` for the one state
    #: this boolean cannot express on its own.
    supported: bool
    #: Whether this attribute's **agreement with cTAKES** supports a decision.
    #:
    #: Named for the reference on purpose. Measurability is not a property of
    #: an attribute; it is a property of an attribute *and* the thing it is
    #: being measured against, and the bare word ``measurable`` hides that.
    #: ``False`` does **not** mean "too few positives to be more than a
    #: diagnostic" -- docs/ADJUDICATION_RESULTS.md refutes exactly that under a
    #: heading naming the error: for ``uncertain`` the problem is not the
    #: count, it is that
    #: 97% of cTAKES' positives are not hedged. The two diagnoses imply
    #: different remedies -- a bigger corpus, versus adjudication -- and only
    #: the second one worked.
    #:
    #: So ``False`` here means "do not decide anything from the cTAKES number",
    #: and says nothing about whether the attribute can be measured at all.
    #: ``uncertain`` is ``False`` and has a usable adjudicated precision; see
    #: :attr:`note` for the per-attribute reason and
    #: docs/ADJUDICATION_RESULTS.md for what a stratified sample of the
    #: *pipeline's* own positives can estimate instead.
    measurable_against_ctakes: bool
    #: Positives in the 20-note silver standard, out of 1,724 mentions. Carried
    #: so a report can print the denominator next to the score rather than
    #: leaving a reader to assume the sample was adequate.
    #:
    #: **This is not the number a score is computed over.** Scoring aligns
    #: mentions on CUI + overlapping span, and alignment drops the ones this
    #: pipeline's dictionary does not reproduce: 1,724 reference mentions
    #: become 1,708 aligned, and ``uncertain``'s 16 corpus positives become
    #: 15 *aligned* positives. Both numbers are right and they answer
    #: different questions -- how rare is this attribute in the corpus (here),
    #: versus how many positives did the scorer actually see
    #: (``AggregateAttributeScore.n_gold_positives``). Quote whichever you mean
    #: and say which; the two appearing unlabelled in different documents is
    #: how a reader starts doubting both.
    corpus_positives: int
    #: One line on what the attribute means and what its number is worth.
    note: str
    #: What a human adjudicator is being asked. Written as a question about the
    #: *mention*, not about what a system said, because the review file is
    #: blind -- see :mod:`umlsmatch.eval.adjudication`.
    question: str
    #: What an adjudicator types for yes and for no. Domain words rather than
    #: "yes"/"no": an annotator reading 400 rows judges faster against
    #: "family_member / patient" than against a boolean they must re-map each
    #: time, and a mistyped domain word is far likelier to be caught than a
    #: transposed yes/no.
    positive_label: str
    negative_label: str
    #: Reduce a silver mention dict to the boolean this attribute scores.
    gold: Callable[[dict], bool] = lambda m: False
    #: Rules exist but are off by default, so the attribute is assessed only
    #: when the pipeline is built with :attr:`pipeline_option` set.
    #:
    #: The third state ``supported`` cannot express. Without it, an attribute
    #: whose rules are being tried out has to be either ``supported=True`` --
    #: which ships it, and publishes a score for it under ``--attribute all`` --
    #: or ``supported=False``, which makes the adjudication sampler refuse it
    #: and so makes it impossible to find out whether it works. That circularity
    #: is what docs/ADJUDICATION_RESULTS.md identifies as a chicken-and-egg
    #: rather than a wall, and this field is the way out of it: the tooling that
    #: measures an attribute accepts a prototype, and the tooling that reports
    #: results does not.
    #:
    #: A prototype is not a backlog item. It is either promoted on adjudicated
    #: evidence or deleted; leaving one here indefinitely would reintroduce the
    #: unaudited ``False`` the three-state contract exists to prevent.
    prototype: bool = False
    #: ``ClinicalPipeline`` keyword that turns this attribute's rules on, or
    #: ``None`` when there is no switch (``negated``, which is always assessed,
    #: and ``generic``, which has no rules). Named here so the eval tools can
    #: build a pipeline that actually assesses the attribute they were asked to
    #: measure, rather than each keeping its own copy of the mapping.
    pipeline_option: str | None = None

    @property
    def assessable(self) -> bool:
        """Can this attribute be assessed at all, on some configuration?

        The gate for tooling that *measures* -- the adjudication sampler, the
        scorer when given an explicit ``--attribute``. Tooling that *reports* a
        result should keep using :attr:`supported`, which is the narrower claim.
        """
        return self.supported or self.prototype

    @property
    def verdicts(self) -> tuple[str, str, str]:
        """Accepted values in a review file's ``verdict`` column.

        ``unclear`` is offered on purpose: forcing a binary choice on a
        genuinely ambiguous mention manufactures a label that is worse than no
        label. Unclear cases are excluded from the estimate and reported, so a
        high rate is visible rather than buried.
        """
        return (self.positive_label, self.negative_label, "unclear")

    @property
    def corpus_rate(self) -> float:
        """Positive rate in the reference corpus. The denominator is fixed."""
        return self.corpus_positives / _CORPUS_MENTIONS


#: Mentions in the 20-note silver standard (``free_texts/synthetic``). Fixed
#: rather than computed so :attr:`AttributeSpec.corpus_rate` needs no corpus on
#: disk; re-derive both if the corpus changes.
#: ``tests/test_attributes.py::test_corpus_positives_match_the_silver_standard``
#: fails when they drift.
_CORPUS_MENTIONS = 1_724


ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec(
        name="negated",
        ctakes_name="polarity",
        silver_key="negated",
        supported=True,
        measurable_against_ctakes=True,
        corpus_positives=106,
        note="Polarity. The only attribute with a published, defensible score; "
             "see umlsmatch.assertion.negation.",
        question="Does this note say the patient does NOT have this?",
        positive_label="negated",
        negative_label="affirmed",
        gold=lambda m: bool(m.get("negated", False)),
    ),
    AttributeSpec(
        name="subject",
        ctakes_name="subject",
        silver_key="subject",
        supported=True,
        # This corpus holds 2 positives, so the agreement figure cannot support
        # a decision about the rules and must not print without the "do not gate
        # on it" caveat. A fact about the corpus, not the rules: on a real-note
        # corpus the same rules see 559 positives and reach F1 0.956 adjudicated.
        measurable_against_ctakes=False,
        corpus_positives=2,
        note="Patient vs. family member. Scored as 'is family_member' -- 'other' "
             "does not occur in the corpus. 2 positives cannot score it; the "
             "synthetic notes carry no family-history tables, which is what the "
             "attribute exists for. See docs/ADJUDICATION_RESULTS.md.",
        question="Whose problem is this -- a relative's, or the patient's own?",
        positive_label=SUBJECT_FAMILY_MEMBER,
        negative_label=SUBJECT_PATIENT,
        gold=lambda m: m.get("subject") == SUBJECT_FAMILY_MEMBER,
        pipeline_option="subject",
    ),
    AttributeSpec(
        name="history_of",
        ctakes_name="historyOf",
        silver_key="history_of",
        supported=True,
        measurable_against_ctakes=True,
        corpus_positives=17,
        note="The mention sits in a history-taking context. Narrower than "
             "past tense: 'X resolved' is history, 'X yesterday' often is not. "
             "17 corpus positives make the cTAKES figure thin; the adjudicated "
             "estimate is P 0.460 / R 0.198 (model pre-annotation).",
        question=(
            "Is this being reported as history -- something taken from the "
            "patient's past -- rather than as part of the present illness?"
        ),
        positive_label="history",
        negative_label="current",
        gold=lambda m: bool(m.get("history_of", False)),
        pipeline_option="history",
    ),
    AttributeSpec(
        name="uncertain",
        ctakes_name="uncertainty",
        silver_key="uncertain",
        supported=True,
        measurable_against_ctakes=False,
        corpus_positives=16,
        note="Hedged assertion. The cTAKES figure measures the reference, not "
             "the rules: on a real-note corpus not included here 97% of what cTAKES called "
             "uncertain was not hedged, and adjudicated sampling put precision "
             "at 0.724 with recall unestimable. The 20-note corpus cannot "
             "repeat that -- the two systems overlap on nothing, so the "
             "stratified design has no both_positive stratum and it has not "
             "been adjudicated. See docs/ADJUDICATION_RESULTS.md.",
        question="Is this hedged -- possible, suspected, to be ruled out?",
        positive_label="uncertain",
        negative_label="definite",
        gold=lambda m: bool(m.get("uncertain", False)),
        pipeline_option="uncertainty",
    ),
    AttributeSpec(
        name="generic",
        ctakes_name="generic",
        silver_key="generic",
        supported=False,
        measurable_against_ctakes=False,
        corpus_positives=7,
        note="Corpus-blocked: 7 positives. Deliberately not implemented -- an "
             "unmeasurable False in a clinical record reads as an assessment. "
             "The reason is not the count: there is no external cue lexicon to "
             "adopt, so rules could only be phrases invented to fit the corpus.",
        question="Is this a generic reference rather than a specific instance?",
        positive_label="generic",
        negative_label="specific",
        gold=lambda m: bool(m.get("generic", False)),
    ),
    AttributeSpec(
        name="conditional",
        ctakes_name="conditional",
        silver_key="conditional",
        supported=False,
        measurable_against_ctakes=False,
        corpus_positives=3,
        note="Prototype, off by default: ConText HYPOTHETICAL cues in "
             "umlsmatch.assertion.conditional. 3 corpus positives cannot "
             "score it; adjudication can, and has not been run yet.",
        question="Is this conditional -- 'call if you develop chest pain'?",
        positive_label="conditional",
        negative_label="unconditional",
        gold=lambda m: bool(m.get("conditional", False)),
        prototype=True,
        pipeline_option="conditional",
    ),
)

ATTRIBUTES_BY_NAME: dict[str, AttributeSpec] = {a.name: a for a in ATTRIBUTES}


def attribute_names(*, supported_only: bool = False) -> tuple[str, ...]:
    """Attribute names, in registry order. Used to build CLI choices."""
    return tuple(a.name for a in ATTRIBUTES if a.supported or not supported_only)


def get_attribute(name: str) -> AttributeSpec:
    """Look up one spec, naming the alternatives when the name is wrong.

    Raises rather than returning ``None``: every caller is resolving a
    command-line argument, and a silent miss there would score the wrong
    attribute under the right heading.
    """
    try:
        return ATTRIBUTES_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown assertion attribute {name!r}; expected one of "
            f"{', '.join(attribute_names())}"
        ) from None

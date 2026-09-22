"""End-to-end clinical text analysis: text in, annotations out.

This is the front door. It wires together the stages that already exist
independently -- sentence/token/POS from :mod:`umlsmatch.pipeline.tokenizer`,
UMLS concept lookup from :mod:`umlsmatch.dictionary.matcher`, and the assertion
rules in :mod:`umlsmatch.assertion` -- so callers don't have to.

    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline() as nlp:
        for a in nlp.analyze("Patient denies chest pain. History of CHF."):
            print(a.cui, a.text, a.group, a.negated, a.subject, a.history_of)

Scope, stated plainly. This covers the core of cTAKES' default pipeline:
concept extraction plus four of the six assertion attributes -- ``negated``
(:mod:`~umlsmatch.assertion.negation`), ``subject``
(:mod:`~umlsmatch.assertion.subject`), ``history_of``
(:mod:`~umlsmatch.assertion.history`) and ``uncertain``
(:mod:`~umlsmatch.assertion.uncertainty`).

``generic`` is **deliberately not implemented** and reports ``None``;
:mod:`umlsmatch.assertion.attributes` records why, and it is a decision about
what this corpus can measure rather than a backlog item. ``conditional`` also
reports ``None`` by default, but for a different reason: rules for it exist
(:mod:`umlsmatch.assertion.conditional`) and are off until asked for, because
nothing has measured them. ``ClinicalPipeline(conditional=True)`` turns them on.

Relation extraction, temporal reasoning and coreference are not implemented
either -- they are a different problem and not simply a later one.

Each attribute has its own switch on the constructor, and turning one off
yields ``None`` rather than ``False``: "we did not assess this" and "we
assessed this and it is absent" are different claims, and the type keeps them
different. See :class:`Annotation`.

**Two named profiles**, because one question here has two defensible answers
and neither should require knowing a keyword argument::

    ClinicalPipeline()                            # strict, the default
    ClinicalPipeline(profile="clinical_recall")   # optimized for the chart

``strict`` reproduces every published figure. ``clinical_recall`` turns
on section-aware history, which raises ``history_of`` recall against adjudicated
verdicts (.198 -> .670), and drops chart furniture from the output. See :data:`PROFILES` and
docs/ADJUDICATION_RESULTS.md.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from umlsmatch.assertion.attributes import (
    ATTRIBUTES,
    SUBJECT_FAMILY_MEMBER,
    SUBJECT_PATIENT,
)
from umlsmatch.assertion.conditional import conditional_matches
from umlsmatch.assertion.history import CANDIDATE_HISTORY_SECTIONS, history_matches
from umlsmatch.assertion.negation import MAX_SCOPE_TOKENS, negated_matches
from umlsmatch.assertion.sections import (
    NEGATIVE_FINDINGS_SECTIONS,
    section_header_span,
    track_sections,
)
from umlsmatch.assertion.subject import family_member_matches
from umlsmatch.assertion.uncertainty import uncertain_matches
from umlsmatch.dictionary.matcher import Match, RareWordMatcher, longest_non_overlapping

__all__ = [
    "DEFAULT_DB",
    "DEFAULT_PROFILE",
    "PROFILES",
    "Annotation",
    "ClinicalPipeline",
    "find_dictionary",
]

#: Default dictionary location, relative to the working directory.
DEFAULT_DB = Path("data/umls_sno_rx.sqlite")

#: Environment variable that overrides the dictionary path.
DB_ENV_VAR = "UMLSMATCH_DB"

#: Characters per document, checked by :meth:`ClinicalPipeline.analyze`.
#:
#: spaCy's own ``nlp.max_length`` is the same number and is the reason for it:
#: the parser wants roughly 1 GB of working memory per 100,000 characters, so
#: beyond this a document is a memory-allocation risk rather than a slow
#: analysis. Matching the value means this check replaces spaCy's error rather
#: than shadowing it at a different threshold, so raising one and not the other
#: cannot produce a document this accepts and the tokenizer then rejects.
DEFAULT_MAX_CHARS = 1_000_000


class _Unset:
    """Sentinel: this argument was not supplied.

    ``None`` is taken in this package -- on :class:`Annotation` it means "not
    assessed", a claim about clinical data -- so it must not double as "use the
    profile's value" on the constructor. Different question, different token.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET = _Unset()

#: Named configurations, each a set of constructor arguments.
#:
#: These exist because the pipeline's defaults answer a question that has two
#: defensible answers, and the choice was previously reachable only by knowing
#: which keyword argument to pass. ``history_sections`` in particular is the
#: single switch that moves ``history_of`` adjudicated F1 from .277 to .704
#: (recall .198 -> .670), and it ships off because the evidence rewarding it is
#: a model's pre-annotation whose labelling convention is the rule itself. See
#: docs/ADJUDICATION_RESULTS.md.
#:
#: A profile covers only the two switches that decision turns on. Everything
#: else -- ``coordination``, ``clause_bounding``, ``sections``, the attribute
#: toggles -- is the same in both, because nothing measured argues for splitting
#: them. Naming a profile is a shorthand for a documented trade-off, not a
#: preset that quietly changes six things.
PROFILES: dict[str, dict[str, bool]] = {
    # Takes the note literally: no attribute is inferred from section context,
    # and nothing is dropped for sitting in a heading. Every published
    # validation figure is measured here. Do not change it to improve a number;
    # a profile that drifts stops reproducing the table it exists to reproduce.
    "strict": {
        "history_sections": False,
        "drop_header_mentions": False,
    },
    # Optimizes for agreement with adjudicated verdicts instead. Section-aware
    # history raises recall sharply, and dropping header mentions removes 1.9%
    # of annotations at a cost of 0.008 concept-extraction F1 (0.971 -> 0.963).
    "clinical_recall": {
        "history_sections": True,
        "drop_header_mentions": True,
    },
}

#: Used when no profile is named. Deliberately the strict one: it is what this
#: package already did, and a default that silently changed on upgrade would
#: invalidate every number a user had measured against it.
DEFAULT_PROFILE = "strict"


def resolve_profile(name: str | None) -> dict[str, bool]:
    """Settings for `name`, or for :data:`DEFAULT_PROFILE` when it is ``None``.

    Raises ``ValueError`` naming the valid profiles rather than ``KeyError``,
    because this is reached from a CLI flag and an environment variable where
    the input is a typo far more often than a bug.
    """
    if name is None:
        name = DEFAULT_PROFILE
    try:
        return dict(PROFILES[name])
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose one of {', '.join(sorted(PROFILES))}"
        ) from None


def find_dictionary(explicit: str | Path | None = None) -> Path:
    """Resolve which dictionary to use.

    Order: explicit argument, then ``$UMLSMATCH_DB``, then ``data/umls_sno_rx.sqlite``
    relative to the working directory, then the same path relative to the
    repository root (so the package works when imported from elsewhere).
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(DB_ENV_VAR)
    if env:
        return Path(env)
    if DEFAULT_DB.is_file():
        return DEFAULT_DB
    repo_relative = Path(__file__).resolve().parent.parent.parent / DEFAULT_DB
    if repo_relative.is_file():
        return repo_relative
    return DEFAULT_DB  # let the matcher raise a useful FileNotFoundError


@dataclass(frozen=True, slots=True)
class Annotation:
    """One clinical concept found in a document, with its assertion attributes.

    **Unassessed attributes are ``None``, not ``False``.** cTAKES writes six
    assertion attributes and this pipeline does not implement all of them (see
    :mod:`umlsmatch.assertion.attributes` for which, and why). The honest
    spelling of "we did not assess this" is ``None``: it is a different claim
    from "we assessed this and it is absent", and in a clinical record the
    difference is the whole point. A confidently wrong ``False`` on
    ``conditional`` is indistinguishable downstream from a genuine assessment,
    and nobody reading it can tell which they got.

    The cost is real and is accepted deliberately: every consumer now handles
    three states rather than two. ``negated`` is exempt -- it is always
    assessed, so it stays a plain ``bool`` and existing code that treats it as
    one is unaffected.
    """

    cui: str
    #: Exact document text that matched.
    text: str
    #: Character offsets into the document.
    start: int
    end: int
    #: cTAKES semantic group, e.g. "DISORDER", "DRUG", "PROCEDURE".
    group: str
    #: True when a negation trigger scopes over this mention. Always assessed.
    negated: bool
    #: The concept's canonical UMLS label.
    preferred_text: str = ""
    #: Dictionary form that matched, in tokenizer spelling.
    term: str = ""
    #: Whose problem this is: ``"patient"``, ``"family_member"``, or ``None``
    #: when not assessed. cTAKES also emits ``"other"``, which this pipeline
    #: never predicts -- it does not occur at all in the 1,724 reference
    #: mentions, and occurred once in 12,592 on a real-note corpus not included here.
    subject: str | None = None
    #: True when the mention sits in a history-taking context. ``None`` when
    #: not assessed.
    history_of: bool | None = None
    #: True when the assertion is hedged ("possible", "cannot rule out").
    #: ``None`` when not assessed. Do not gate on the cTAKES figure (F1 0.162)
    #: -- 97% of what cTAKES calls uncertain is not hedged, so it measures the
    #: reference. Adjudicated sampling puts precision at 0.724 and leaves
    #: recall unestimable; both are provisional. See
    #: :mod:`umlsmatch.assertion.uncertainty`.
    uncertain: bool | None = None
    #: True when the mention is asserted under a condition ("call if you
    #: develop chest pain"). ``None`` unless the pipeline was built with
    #: ``conditional=True``, which is a prototype: 27 reference positives
    #: cannot score it, so it is off until adjudication says otherwise.
    conditional: bool | None = None
    #: Deliberately never assessed: 54 positives, and unlike ``conditional``
    #: there is no external cue lexicon to adopt -- see
    #: :mod:`umlsmatch.assertion.attributes`.
    generic: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ClinicalPipeline:
    """Text -> :class:`Annotation` list.

    Construction loads a spaCy model and opens the dictionary, so build one and
    reuse it; ``analyze()`` afterwards is cheap by comparison.

    **Not thread-safe** -- it inherits
    :class:`~umlsmatch.dictionary.matcher.RareWordMatcher`'s unsynchronized
    caches. Scale with processes, one pipeline per worker; see
    ``examples/parse_to_jsonl_batch.py``.

    Args:
        db_path: dictionary to use; see :func:`find_dictionary` for resolution.
        profile: a name from :data:`PROFILES`, supplying defaults for the
            settings that trade a literal reading of the note against clinical
            recall. ``"strict"`` (the default, and what this class has always
            done) infers nothing from section context and reproduces every
            published validation figure; ``"clinical_recall"`` turns on
            section-aware history and drops header mentions, which is the
            better configuration when the question is what is in the chart.
            Any keyword passed explicitly wins over the profile, so
            ``profile="clinical_recall", drop_header_mentions=False`` means
            what it reads as.
        max_scope: negation scope cap in tokens. ``None`` disables the cap
            (reaches to the terminator or sentence edge, which over-negates).
            Must be non-negative; a negative cap would empty every scope and
            look like negation silently not working.
        coordination: let a negation trigger follow a coordinate list past
            ``max_scope`` ("denies chest pain, shortness of breath, or fever").
            Additive -- it can only negate more, never less. This is the one
            knob here that lowers measured agreement with cTAKES while raising
            recall; ``coordination=False`` with ``clause_bounding=False`` gives
            the lexical-only row (precision 0.515). See
            :mod:`umlsmatch.assertion.negation` for the table and for why the
            default is nonetheless on.
        clause_bounding: stop a trigger at a coordinated clause ("Patient
            denies chest pain, and reports pneumonia and diabetes mellitus."
            must not negate either reported diagnosis). Raises
            both precision and accuracy against cTAKES; on unless you are
            reproducing older numbers.
        sections: uncap ``max_scope`` inside negative-findings sections such as
            Review of Systems, where the lists outrun any fixed window and the
            parse is least reliable. A trigger is still required; this never
            negates a section wholesale. See
            :mod:`umlsmatch.assertion.sections`.
        resolve_overlaps: keep only the longest of overlapping matches. Off by
            default because cTAKES emits overlapping annotations too, and
            enabling it measurably *lowers* recall. Co-extensive matches are
            kept -- two CUIs on one span are two concepts, not two opinions
            about a span -- so "CHF" still yields both of its CUIs while
            "chest" and "pain" are suppressed by "chest pain". See
            :func:`~umlsmatch.dictionary.matcher.longest_non_overlapping`.
        subject: assess whose problem each mention is -- the patient's or a
            family member's (:mod:`umlsmatch.assertion.subject`). Turning it
            off leaves ``Annotation.subject`` ``None``, which reads as "not
            assessed" rather than "the patient"; that distinction is why this
            is a switch and not a filter applied afterwards.
        history: assess whether each mention sits in a history-taking context
            (:mod:`umlsmatch.assertion.history`). Off leaves
            ``Annotation.history_of`` ``None``, as above.
        history_sections: treat everything under a Past Medical History header
            as history, not only mentions carrying their own lexical cue.
            **Profile-controlled**; defaults to the profile's value. This is
            the switch the two references disagree about in opposite
            directions -- see :data:`PROFILES`.
        drop_header_mentions: discard concepts lying inside a section heading
            ("History" in "Past Medical History:"). **Profile-controlled.**
        uncertainty: assess whether each mention is hedged
            (:mod:`umlsmatch.assertion.uncertainty`). **Usable precision,
            unestimable recall** -- adjudicated sampling puts precision at
            0.724 while recall's interval spans 0.228-0.955. The published
            cTAKES agreement (F1 0.162) is not a capability figure and should
            not be gated on: 97% of what cTAKES marks uncertain is not hedged,
            so that number is mostly a report on the reference. Both
            adjudicated figures come from a model pre-annotation rather than a
            clinician and are provisional.
        conditional: assess whether each mention is asserted under a condition
            -- "call if you develop chest pain"
            (:mod:`umlsmatch.assertion.conditional`). **A prototype, and the
            only attribute switch that is off by default.** The rules are
            ConText's HYPOTHETICAL cues; nothing has measured them, because 27
            corpus positives cannot. Off means ``Annotation.conditional`` stays
            ``None``, which is the claim the README makes about this attribute
            and remains true for anyone who does not ask for it. Turn it on to
            adjudicate it -- see that module on what would promote it.
        groups: if given, keep only annotations in these semantic groups,
            e.g. ``{"DISORDER", "DRUG"}``.
        max_chars: reject a document longer than this (default
            :data:`DEFAULT_MAX_CHARS`). ``None`` removes the check and hands
            the document to spaCy, which has its own limit and a less helpful
            way of saying so. Raise both together or neither.
        model: spaCy model name, or a preloaded ``Language``.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        profile: str | None = None,
        max_scope: int | None = MAX_SCOPE_TOKENS,
        coordination: bool = True,
        clause_bounding: bool = True,
        sections: bool = True,
        resolve_overlaps: bool = False,
        subject: bool = True,
        history: bool = True,
        history_sections: bool | _Unset = UNSET,
        uncertainty: bool = True,
        conditional: bool = False,
        drop_header_mentions: bool | _Unset = UNSET,
        groups: Iterable[str] | None = None,
        max_chars: int | None = DEFAULT_MAX_CHARS,
        model: str | object | None = None,
    ) -> None:
        # Imported here so `import umlsmatch` stays usable without the optional
        # `nlp` extra installed -- the error should name the missing extra.
        try:
            from umlsmatch.pipeline.tokenizer import DEFAULT_MODEL, annotate_sentences
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "ClinicalPipeline needs the optional 'nlp' extra (spaCy). "
                "Install it with:  pip install -e \".[nlp]\""
            ) from exc

        if max_scope is not None and max_scope < 0:
            raise ValueError(f"max_scope must be >= 0 or None, got {max_scope}")

        # Resolved before anything else so an unknown profile name fails
        # before a 589 MB dictionary is opened and a model is loaded.
        settings = resolve_profile(profile)
        if not isinstance(history_sections, _Unset):
            settings["history_sections"] = history_sections
        if not isinstance(drop_header_mentions, _Unset):
            settings["drop_header_mentions"] = drop_header_mentions

        self._annotate_sentences = annotate_sentences
        self._model = model or DEFAULT_MODEL
        self.db_path = find_dictionary(db_path)
        self.matcher = RareWordMatcher(self.db_path)
        #: Which profile supplied the defaults, for introspection and for
        #: ``/info``. An explicit keyword argument can still have overridden
        #: any individual setting, so this names the starting point rather
        #: than a guarantee about the final state.
        self.profile = profile or DEFAULT_PROFILE
        self.max_scope = max_scope
        self.coordination = coordination
        self.clause_bounding = clause_bounding
        self.sections = sections
        self.resolve_overlaps = resolve_overlaps
        self.subject = subject
        self.history = history
        # Off under `strict`, on under `clinical_recall`. The rule it
        # enables scored precision 0.116 measured in isolation against cTAKES,
        # which is why it is not the default -- but cTAKES misses the same
        # mentions, so that number cannot settle it. See
        # docs/ADJUDICATION_RESULTS.md and umlsmatch.assertion.history.
        self.history_sections = settings["history_sections"]
        self._history_sections = (
            CANDIDATE_HISTORY_SECTIONS if self.history_sections else None
        )
        # Off under `strict`: it changes what is *extracted*, and the
        # F1 0.971 concept-extraction figure is measured on the unfiltered stream.
        self.drop_header_mentions = settings["drop_header_mentions"]
        self.uncertainty = uncertainty
        # Off by default, and the default is the claim: every other attribute
        # switch ships on because a number stands behind it. This one has no
        # number and cannot get one from this corpus, so asking for it is an
        # explicit act. See umlsmatch.assertion.conditional.
        self.conditional = conditional
        self.groups = frozenset(groups) if groups else None
        if max_chars is not None and max_chars < 1:
            raise ValueError(f"max_chars must be >= 1 or None, got {max_chars}")
        self.max_chars = max_chars

    @property
    def assessed_attributes(self) -> frozenset[str]:
        """Assertion attributes *this* pipeline populates, by :class:`Annotation`
        field name.

        Every other name in :func:`~umlsmatch.assertion.attributes.attribute_names`
        is left ``None`` on every annotation this pipeline produces -- not
        because the mention lacks the property, but because nothing looked. A
        consumer writing a fixed schema (a CSV header, a Parquet schema) needs
        to know that before the first row exists, and the answer depends on how
        this pipeline was configured rather than on what the data turned out to
        contain: a column of empty cells is indistinguishable from an attribute
        that happened not to occur.

        Derived from the registry and this instance's own switches, so enabling
        ``conditional=True`` brings the attribute back without any consumer
        being told separately. ``generic`` has no switch and no rules, so it is
        never in this set; ``negated`` has no switch because it is always
        assessed, so it always is.
        """
        return frozenset(
            spec.name
            for spec in ATTRIBUTES
            if spec.assessable
            and (spec.pipeline_option is None or getattr(self, spec.pipeline_option))
        )

    def _to_annotation(
        self,
        m: Match,
        text: str,
        *,
        negated: bool,
        subject: str | None = None,
        history_of: bool | None = None,
        uncertain: bool | None = None,
        conditional: bool | None = None,
    ) -> Annotation:
        """Build one annotation. Attribute arguments default to "not assessed".

        Keyword-only past `text`: five booleans in a row is exactly the
        signature where a transposed pair produces a plausible result and no
        error, and here a transposition would silently report one attribute's
        value under another's name.
        """
        return Annotation(
            cui=m.cui,
            text=text[m.start : m.end],
            start=m.start,
            end=m.end,
            group=m.group,
            negated=negated,
            # The matcher memoizes this, so repeated CUIs cost one dict hit.
            preferred_text=self.matcher.preferred_text(m.cui),
            term=m.term,
            subject=subject,
            history_of=history_of,
            uncertain=uncertain,
            conditional=conditional,
        )

    def analyze(self, text: str) -> list[Annotation]:
        """Find concepts in `text`, each flagged for negation.

        Returns annotations in document order. Overlapping annotations are
        returned unless ``resolve_overlaps`` was set.

        Raises:
            ValueError: `text` is longer than :attr:`max_chars`.
        """
        if not text or not text.strip():
            return []
        self._check_length(text)

        out: list[Annotation] = []
        # Section state spans windows, so `track_sections` holds it. It reports
        # the canonical section *name* rather than the old "is this a
        # negative-findings section" boolean: negation only ever asked that
        # question, but `subject` asks about a different section, and booleans
        # tracked in parallel would be one chance each to leave one stuck on.
        windows = self._annotate_sentences(text, model=self._model)
        for tokens, header in track_sections(windows):
            # `sections=False` keeps the windows and discards the state, so the
            # switch disables the section *rules* without changing what gets
            # extracted.
            section = header if self.sections else None
            in_negative_section = section in NEGATIVE_FINDINGS_SECTIONS

            matches: Sequence[Match] = self.matcher.match(tokens)
            if self.drop_header_mentions:
                # A concept lying inside the heading itself -- the "History" in
                # "Past Medical History:" -- is part of the chart's furniture,
                # not of the note. Narrow on purpose: measured against
                # adjudicated verdicts it removes 1.9% of annotations, at a
                # cost of 0.008 concept-extraction F1. See
                # docs/ADJUDICATION_RESULTS.md.
                found = section_header_span(tokens)
                if found is not None:
                    matches = [m for m in matches if m.end > found[1]]
            if not matches:
                continue
            if self.resolve_overlaps:
                matches = longest_non_overlapping(matches)
            negated = negated_matches(
                tokens,
                matches,
                # Inside a negative-findings section the cap is the wrong
                # instrument -- see umlsmatch.assertion.sections.
                max_scope=None if in_negative_section else self.max_scope,
                coordination=self.coordination,
                clause_bounding=self.clause_bounding,
            )
            family = (
                family_member_matches(tokens, matches, section=section)
                if self.subject
                else frozenset()
            )
            past = (
                history_matches(
                    tokens,
                    matches,
                    section=section,
                    history_sections=self._history_sections,
                )
                if self.history
                else frozenset()
            )
            hedged = (
                uncertain_matches(tokens, matches) if self.uncertainty else frozenset()
            )
            hypothetical = (
                conditional_matches(tokens, matches)
                if self.conditional
                else frozenset()
            )
            for m in matches:
                if self.groups is not None and m.group not in self.groups:
                    continue
                out.append(
                    self._to_annotation(
                        m,
                        text,
                        negated=m in negated,
                        subject=self._subject(m in family),
                        history_of=(m in past) if self.history else None,
                        uncertain=(m in hedged) if self.uncertainty else None,
                        conditional=(m in hypothetical) if self.conditional else None,
                    )
                )

        out.sort(key=lambda a: (a.start, a.end, a.cui))
        return out

    def _check_length(self, text: str) -> None:
        """Reject an over-long document here rather than inside spaCy.

        Without this the failure is ``ValueError: [E088] Text of length N
        exceeds maximum of 1000000``, raised from the tokenizer several frames
        down. It is accurate and it names nothing the caller recognizes: not
        this pipeline, not the document, and not what to do -- the remedy it
        suggests (raise ``nlp.max_length``) is advice about a spaCy object the
        caller never constructed.

        The service has always checked this at its own edge (``MAX_CHARS``),
        which is exactly why the gap was easy to miss: the HTTP path was clean
        and every script using the library directly got the raw spaCy error.
        """
        limit = self.max_chars
        if limit is not None and len(text) > limit:
            raise ValueError(
                f"document is {len(text):,} characters; the limit is {limit:,}. "
                "Split it into sections or notes and analyze each -- annotation "
                "offsets are per-document, so a caller that splits must add the "
                "section's offset back. Raise ClinicalPipeline(max_chars=...) if "
                "the model on this machine can take it; spaCy's parser wants "
                "roughly 1 GB of working memory per 100,000 characters."
            )

    def _subject(self, is_family: bool) -> str | None:
        """``None`` when the rule is off -- "not assessed", not "the patient"."""
        if not self.subject:
            return None
        return SUBJECT_FAMILY_MEMBER if is_family else SUBJECT_PATIENT

    def analyze_documents(self, texts: Iterable[str]) -> Iterator[list[Annotation]]:
        """Analyze many documents, yielding one annotation list per document."""
        for text in texts:
            yield self.analyze(text)

    def to_json(self, text: str, *, indent: int | None = 2) -> str:
        """Analyze `text` and return annotations as a JSON array."""
        return json.dumps([a.to_dict() for a in self.analyze(text)], indent=indent)

    def close(self) -> None:
        self.matcher.close()

    def __enter__(self) -> ClinicalPipeline:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

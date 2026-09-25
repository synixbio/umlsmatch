"""Whose problem is this -- the patient's, or a family member's.

cTAKES' ``subject`` attribute. It *was* the best-supported of the non-polarity
attributes, on a real-note corpus not included here carrying 559 ``family_member``
positives in 12,592 mentions (4.4%); the signal is overwhelmingly lexical and
sectional, which is what this codebase already does well. **The 20-note
synthetic corpus shipped here carries 2 positives in 1,724 mentions
(0.12%)**, so none of that evidence can be reproduced from anything on disk --
see the measurement note below.

**Two rules, in this order.**

1. **Section.** A mention inside a Family History section is a family member's
   unless the sentence points back at the patient. The section state machine
   ``ClinicalPipeline.analyze`` already maintains for negation is reused
   wholesale -- the header is the one signal that survives a bare comma list of
   diagnoses with no verb and no sentence-final punctuation, exactly as in
   :mod:`umlsmatch.assertion.sections`.
2. **Cue.** A kinship term scopes over nearby mentions, using the plain window
   rule from :mod:`umlsmatch.assertion.scope`.

**The section rule is the one that carries the attribute, and it is not a
blanket.** "Family history: mother with breast cancer. Patient has never been
screened." must not make the screening a family member's. The patient cues in
:data:`PATIENT_CUES` terminate a family scope and, inside a family-history
section, revert a sentence to the patient outright when no kinship term is
present. That restraint is the same one
:mod:`umlsmatch.assertion.sections` makes about negation: a section changes
how a cue is scoped, it is not itself an assertion.

**The section rule inherits section tracking's failure mode, and it is worth
knowing before reading a surprising result.** A Family History section runs
until :mod:`umlsmatch.assertion.sections` recognizes the *next* header. In a
real note one always follows; in a fragment that does not have one -- a
synthetic test case, a note truncated mid-way, a header spelling
``OTHER_SECTIONS`` does not carry -- the section runs to the end of the text
and claims every mention in it that lacks a patient cue. The same leak is
documented for negation in that module, where it was measured as a real cause
of wrong flags. Its cost here is bounded the same way: a leaked section still
needs a window with no reference to the patient in it, and the rule measures at
precision 0.84 on this corpus because real notes have headers.

**Direction matters and the two directions are not symmetric.** "Mother with
diabetes" puts the kinship term first; "MI in her father" puts it last. Both
occur, so kinship terms are matched in both directions -- but the backward
window is tighter (:data:`BACKWARD_SCOPE_TOKENS` against
:data:`FORWARD_SCOPE_TOKENS`), because a trailing "in her father" attaches to
the noun phrase immediately before it, while a leading "Mother:" heads a list
that can run on.

**``other`` is never predicted.** cTAKES' third value does not occur at all in
the 1,724 reference mentions (it occurs once in a real-note corpus not included
here). A rule for it could not be measured, and an unmeasurable
label in a clinical record reads as an assessment -- the same argument that
makes for not implementing ``generic`` at all, and for keeping
:mod:`umlsmatch.assertion.conditional` off by default.

**This attribute currently has no usable score, and the reason is the corpus.**
The 20-note synthetic corpus holds **2** ``family_member`` positives in 1,708
aligned mentions, because the notes are single-patient prose with no EHR
family-history tables -- which is the construction this module exists for.
Agreement with cTAKES is P 0.111 / R 0.500 / F1 0.182, accuracy 0.995, and that
is a statement about a 2-positive reference rather than about the rules;
``AttributeSpec.measurable_against_ctakes`` is ``False`` for exactly this
reason, so the scorer prints it with a "do not gate on it" caveat. The
stratified adjudication cannot rescue it either: 10 non-``both_negative`` cases.

On a real-note corpus not included here, this module measures **P 0.693 /
R 0.973 / F1 0.809** vs cTAKES over 556 ``family_member`` positives, and **F1
0.956** against adjudicated verdicts -- its best-evidenced result. Those numbers
are kept in docs/ADJUDICATION_RESULTS.md and are not reproducible from anything
on disk. **The rules are unchanged; only the corpus is.**

The caps sat on a **broad plateau** on that corpus: a sweep over the three
windows moved F1 by about 0.02 either way, and the values below are mid-plateau
rather than the single best measured point. That sweep cannot be repeated here
-- 2 positives will not resolve a 0.02 difference -- so the values stand on the
earlier evidence, the same way
:data:`umlsmatch.assertion.negation.MAX_SCOPE_TOKENS` does.

**A list separator is not a scope terminator here**, which is the opposite of
the obvious call. A bullet, middle dot or semicolon looks like it should end a
cue's reach in the table layout above -- one bullet per row, so a cue in one
row should not reach the next -- and treating it that way costs recall badly.
cTAKES carries a family-history reading *across* the bullets in these tables,
and so should we: the rows share one header, and the bullet separates entries
rather than ending the section.

**Read the precision honestly: 0.693 is not 307 mistakes.** Attributing every
disagreement by the rule that caused it gives section 189 right / 35 wrong,
self-reference 162 / 82, cue scope 198 / 192. But cTAKES itself labels 33
distinct mention texts both ways, including "family history" (67 family, 46
patient) and "diabetes" (38 / 70), and a large share of the remaining
disagreements sit in EHR family-history **tables** --
``Arthritis Brother RA / Arthritis Sister RA / Cancer Brother Prostate`` --
which cTAKES marks ``family_member`` in some notes and ``patient`` in others
that are near-identical. This is the ``subject`` counterpart of the
Review-of-Systems finding in :mod:`umlsmatch.assertion.negation`: the
disagreement is consistent with this pipeline being right, and it is **not**
proof of it. Only human adjudication settles it, which is what
:mod:`umlsmatch.eval.adjudication` exists to make affordable.

Usage::

    from umlsmatch.assertion.subject import family_member_matches

    family = family_member_matches(tokens, matches, section=section)
"""

from __future__ import annotations

from collections.abc import Sequence

from umlsmatch.assertion.scope import (
    Phrases,
    TriggerScope,
    TriggerSpan,
    find_cues,
    find_phrase_spans,
    overlaps,
    phrase_set,
    resolve_scopes,
    scope_causes,
)
from umlsmatch.dictionary.matcher import Match, Token

__all__ = [
    "FAMILY_CUES",
    "FAMILY_HEADER_CUES",
    "FAMILY_HISTORY_SECTIONS",
    "HEADER_SCOPE_TOKENS",
    "KINSHIP_BACKWARD_SCOPE_TOKENS",
    "KINSHIP_CUES",
    "KINSHIP_FORWARD_SCOPE_TOKENS",
    "PATIENT_CUES",
    "PSEUDO_FAMILY_CUES",
    "family_member_matches",
    "subject_evidence",
]

#: How far a "family history" *header* phrase reaches forward. Wide, because
#: what follows it is a list -- "Family history: diabetes, hypertension, breast
#: cancer" -- and the terminators below, not the cap, are what should stop it.
HEADER_SCOPE_TOKENS = 10

#: How far a bare kinship term reaches forward. Much tighter than the header
#: cap, and the difference is load-bearing. "Mother with diabetes" is a short
#: noun phrase; the wide cap was measured at precision 0.41 on this corpus
#: because these notes carry EHR family-history *tables* laid out
#: ``Problem Relation Age`` -- "Arthritis Brother RA / Arthritis Sister RA" --
#: where a forward window from "Brother" reaches into the **next row**.
KINSHIP_FORWARD_SCOPE_TOKENS = 4

#: How far a trailing kinship term reaches back. "MI in her father" modifies
#: the noun phrase directly before it, and the table layout above puts the
#: relation after the problem, so this direction carries the tables.
KINSHIP_BACKWARD_SCOPE_TOKENS = 4

#: Phrases that head a family-history list. **Forward only**: a trailing
#: "family history" scoped backward measured 7 correct against 33 wrong on this
#: corpus (precision 0.17), because the construction it fires on is "Past
#: Medical History ... Family History", where the text before the phrase is the
#: previous section, not its contents.
_FAMILY_HEADER_PHRASES = (
    "family history", "family hx", "fh", "fhx", "famhx",
    "family history of", "history in the family", "runs in the family",
    "family member", "family members",
)

#: Kinship terms, scoped in both directions. See the two caps above.
#:
#: **Spouses are deliberately absent.** "wife", "husband" and "spouse" scored
#: 0 correct against 11 wrong: cTAKES does not treat a spouse as
#: ``family_member``, and these notes mention a spouse mostly as the person who
#: called EMS or gave the history -- "Wife found patient in the bathroom" --
#: where the following diagnosis is the patient's. Including them on the
#: intuition that a spouse is family cost precision and bought nothing.
_KINSHIP_PHRASES = (
    "mother", "mothers", "father", "fathers", "mom", "dad", "parent", "parents",
    "sister", "sisters", "brother", "brothers", "sibling", "siblings",
    "son", "sons", "daughter", "daughters", "child", "children",
    "aunt", "aunts", "uncle", "uncles", "cousin", "cousins",
    "nephew", "niece", "twin", "twins",
    "grandmother", "grandfather", "grandparent", "grandparents",
    "grandmothers", "grandfathers", "grandson", "granddaughter",
    "maternal", "paternal",
    "relative", "relatives",
)

#: Phrases containing a cue word that are not about a relative.
#:
#: "Family planning", "family practice" and "family physician" are the ones
#: that actually occur; "child" is the awkward one, since "child" as a cue
#: ("his child has asthma") and "child" as the patient ("the child presents
#: with fever") are the same token, and pediatric notes use the second freely.
#: The pseudo entries cover the determiner spellings that mark the second
#: reading.
_PSEUDO_FAMILY_PHRASES = (
    "family planning", "family practice", "family physician", "family medicine",
    "family doctor", "family practitioner",
    "the child", "this child", "the patient's child",
    "child abuse", "children's hospital",
    # "relative" as a comparative, not a person: "relative to baseline",
    # "relative risk", "relative hypotension".
    "relative to", "relative risk", "relative hypotension", "relative rest",
)

#: Terms that point the assertion back at the patient. These end a family
#: scope, and inside a family-history section they revert a sentence with no
#: kinship term in it. Kept narrow on purpose: a bare pronoun would fire on
#: "her mother", which is the opposite of what it means here.
_PATIENT_PHRASES = (
    "patient", "patients", "pt", "pts", "the patient", "this patient",
    "he", "she", "himself", "herself", "his own", "her own",
    "denies", "reports", "complains", "presents", "admitted", "admits",
)

#: Phrases that end a family cue's scope early, beyond the patient cues. The
#: lexical ones are the same class negation uses.
_TERMINATOR_PHRASES = (
    "but", "however", "except", "although", "though", "whereas", "otherwise",
)


FAMILY_HEADER_CUES: Phrases = phrase_set(_FAMILY_HEADER_PHRASES)
KINSHIP_CUES: Phrases = phrase_set(_KINSHIP_PHRASES)
#: Every family cue, whatever its scoping rule. Used by the self-reference rule,
#: which is about the cue's own span and so does not care about direction.
FAMILY_CUES: Phrases = phrase_set(_FAMILY_HEADER_PHRASES + _KINSHIP_PHRASES)
PSEUDO_FAMILY_CUES: Phrases = phrase_set(_PSEUDO_FAMILY_PHRASES)
PATIENT_CUES: Phrases = phrase_set(_PATIENT_PHRASES)
_TERMINATORS: Phrases = phrase_set(_TERMINATOR_PHRASES)

#: Canonical section names whose body is about relatives.
#: :mod:`umlsmatch.assertion.sections` already recognizes ``family_history``;
#: this names it rather than spelling the string at each use.
FAMILY_HISTORY_SECTIONS = frozenset({"family_history"})


def _terminator_spans(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """Where a family cue's scope must stop: a lexical terminator or the patient.

    A bullet is **not** a terminator here, which is counter-intuitive and was
    measured -- see "what did not work" in the module docstring.
    """
    norms = [t.norm for t in tokens]
    return find_phrase_spans(norms, _TERMINATORS) + find_phrase_spans(norms, PATIENT_CUES)


def _family_scopes(tokens: Sequence[Token]) -> tuple[list[TriggerScope], list[TriggerSpan]]:
    """Every kinship cue's scope, and the cue spans themselves.

    Forward and backward windows are resolved in separate passes because the
    caps differ; see the module docstring.
    :func:`~umlsmatch.assertion.scope.resolve_scopes` takes a single cap by
    design -- one cap per call is what makes a scope explainable.
    """
    terminators = _terminator_spans(tokens)
    n = len(tokens)
    headers = find_cues(tokens, forward=FAMILY_HEADER_CUES, pseudo=PSEUDO_FAMILY_CUES)
    kin_fwd = find_cues(tokens, forward=KINSHIP_CUES, pseudo=PSEUDO_FAMILY_CUES)
    kin_back = find_cues(tokens, backward=KINSHIP_CUES, pseudo=PSEUDO_FAMILY_CUES)
    scopes = [
        *resolve_scopes(headers, terminators, n, max_scope=HEADER_SCOPE_TOKENS),
        *resolve_scopes(
            kin_fwd, terminators, n, max_scope=KINSHIP_FORWARD_SCOPE_TOKENS
        ),
        *resolve_scopes(
            kin_back, terminators, n, max_scope=KINSHIP_BACKWARD_SCOPE_TOKENS
        ),
    ]
    # `kin_back` is dropped from the cue list: it holds the same phrase spans as
    # `kin_fwd` and differs only in direction, so including both would report
    # every kinship term twice in the evidence.
    return scopes, headers + kin_fwd


def _cue_self_matches(
    matches: Sequence[Match], cues: Sequence[TriggerSpan]
) -> set[Match]:
    """Matches that overlap a family cue span -- the cue phrase *as* a concept.

    "Family history", "no family history" and a bare "History" under a family
    header all resolve to UMLS concepts in their own right, and cTAKES labels
    those concepts ``family_member``. That is coherent rather than a quirk: the
    concept is *about* the family, so its subject is the family.

    It is also, on this corpus, **half the attribute**: 277 of 556 gold
    positives are the cue phrase itself, and turning this rule off drops recall
    from 0.973 to 0.773. That is worth knowing before reading the headline F1,
    which is why ``cue_self_reference=False`` exists -- the difference between
    the two rows is the honest size of the rest of the work. See the table in
    the module docstring.
    """
    return {
        m
        for m in matches
        if any(overlaps((m.token_start, m.token_end), (c.start, c.end)) for c in cues)
    }


def _section_claims_everything(tokens: Sequence[Token]) -> bool:
    """True when a family-history sentence has no patient cue pulling it back.

    The section rule applies to the whole window in that case. When a patient
    cue *is* present -- "Patient has never been screened" under a Family
    History header -- the window falls back to the cue rule, which will claim
    only what a kinship term actually scopes over.
    """
    return not find_phrase_spans([t.norm for t in tokens], PATIENT_CUES)


def subject_evidence(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    section: str | None = None,
    cue_self_reference: bool = True,
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which cue(s) make each match a family member's -- the reasons behind the call.

    Same inputs and the same decision as :func:`family_member_matches`; the keys
    are exactly the set that returns. Diagnostic rather than hot-path: error
    analysis needs to know which cue fired, since "our family-history precision
    is 0.7" only becomes actionable once it decomposes by cue.

    A match claimed by the *section* rule rather than by any cue maps to an
    empty tuple -- it is in the set, and there is no cue to name. Callers
    rendering evidence should say "family history section" for those rather
    than print nothing.
    """
    scopes, cues = _family_scopes(tokens)
    causes = scope_causes(matches, scopes)
    if cue_self_reference:
        for m in _cue_self_matches(matches, cues):
            # The cue that *is* the match explains it better than a window the
            # match happens to also sit in, so it is prepended rather than
            # appended -- error listings read the first cause.
            overlapping = tuple(
                c
                for c in cues
                if overlaps((m.token_start, m.token_end), (c.start, c.end))
            )
            causes[m] = overlapping + tuple(
                c for c in causes.get(m, ()) if c not in overlapping
            )
    if section in FAMILY_HISTORY_SECTIONS and _section_claims_everything(tokens):
        for m in matches:
            causes.setdefault(m, ())
    return causes


def family_member_matches(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    section: str | None = None,
    cue_self_reference: bool = True,
) -> frozenset[Match]:
    """The subset of `matches` that belong to a family member, not the patient.

    `tokens` and `matches` must come from the same sentence --
    ``Match.token_start``/``token_end`` are indices into `tokens`, exactly what
    ``RareWordMatcher.match(tokens)`` returns.

    `section` is the canonical section name the window sits in, as
    :func:`~umlsmatch.assertion.sections.section_header` spells it. ``None``
    disables the section rule, leaving the cue rule alone -- which is what a
    caller that does not track sections gets, and it degrades to fewer
    family-member calls rather than to wrong ones.

    `cue_self_reference` claims a mention that *is* the cue phrase, e.g. the
    concept behind "no family history". It is on by default because it is
    correct, and it is a switch because it accounts for half the reference
    positives and a headline F1 that does not disclose that is misleading --
    see :func:`_cue_self_matches`.

    Everything not in the returned set is the patient's. There is no third
    answer: ``other`` is never predicted, see the module docstring.
    """
    scopes, cues = _family_scopes(tokens)
    if section in FAMILY_HISTORY_SECTIONS and _section_claims_everything(tokens):
        return frozenset(matches)
    family = set(scope_causes(matches, scopes))
    if cue_self_reference:
        family |= _cue_self_matches(matches, cues)
    return frozenset(family)

"""Is this mention part of the history, or of the present illness.

cTAKES' ``historyOf`` attribute, which is narrower than its name suggests. It
does **not** mark every past-tense finding; it marks mentions in
history-*taking* contexts -- the Past Medical History list, "history of severe
COPD", "status post cholecystectomy" -- and leaves the events of this admission
alone even when they are already over. 17 positives in 1,724 reference
mentions (1.0%) -- thin, though still the best class balance of the five
non-polarity attributes. (A real-note corpus carries 644 in 12,592, or 5.1%.)

**This attribute does not meet its target, and ships saying so.**
The target is >=0.80 F1 against the silver standard. The
rules below reach **0.564** (P 0.645, R 0.502). That is reported in this
docstring, in ``tools/score_attributes.py``, and in the README rather than
being quietly dropped, because a weak attribute with a published number is a
different thing from a weak attribute without one. When rules plateau the
answer is to improve them, widen the corpus, or **accept the ceiling and
document it**. This is the third.

**Only one of the three rules** :mod:`umlsmatch.assertion.subject` **uses
survives here**, which is the useful finding: ``subject`` is the same shape of
problem on the same machinery, and it is carried *mostly* by the two rules that
fail for ``history_of``.

1. **Cue.** "history of", "h/o", "s/p", "status post" scope forward;
   "resolved", "in remission", "N years ago" scope backward, because they trail
   the finding they qualify. **Kept: this is the whole of the attribute.**
2. **Section.** Everything under a Past Medical History header is history.
   Obviously true, and measurably wrong against cTAKES: precision 0.116.
   **Dropped** -- see :data:`HISTORY_SECTIONS`.
3. **Self-reference.** The concept behind "past medical history" is itself a
   history concept. True for ``subject``, where it is half the positives; for
   ``history_of``, cTAKES marks only 54 of 450 bare "history" mentions, and the
   rule scores 0.189. **Dropped** -- ``cue_self_reference`` defaults to
   ``False`` here and ``True`` there, and that asymmetry is a measurement, not
   an inconsistency.

**``resolved`` belongs here, and that is not a coincidence.**
:data:`umlsmatch.assertion.negation.NEGEX_EXCLUSIONS` holds the NegEx triggers
that this project refused to adopt as *negation*, and the reason given there was
that they encode temporality rather than polarity -- "X resolved" asserts that X
happened and has since stopped, which is history, not absence. That list is
therefore a list of ``historyOf`` candidates, and it is consumed here rather
than re-typed: see :data:`_FROM_NEGEX_EXCLUSIONS`. Cross-referencing it from
both sides is what keeps the two modules' accounts of these phrases from
drifting apart.

**What it measures.** Per-mention agreement with cTAKES' own ``historyOf``,
20 notes, 1,708 aligned mentions, 17 positives: **P 0.500 / R 1.000 / F1
0.667** (``tools/score_attributes.py --attribute history_of``). The forward cap
sits on a flat plateau (see :data:`FORWARD_SCOPE_TOKENS`), so nothing here is
finely tuned. Every caveat in :mod:`umlsmatch.eval.attributes` applies on top,
and **17 reference positives make this figure thin** -- a recall of 1.000 means
the rules caught all 17, not that they are complete. The reference is also
self-inconsistent: cTAKES labels 6 distinct mention texts both ways, including
"illness" (9 yes, 2 no) and "hypertension" (1 yes, 5 no). Adjudicated verdicts
put it far lower -- P 0.460 / R 0.198 -- since most of what it misses cTAKES
misses too. See docs/ADJUDICATION_RESULTS.md, which also records what the same
rules reach on a real-note corpus not included here (P 0.645 / R 0.502 vs
cTAKES, P 0.881 / R 0.128 adjudicated).

**A lexicon entry that reads correctly and matches nothing is this module's
main hazard.** "h/o" and "s/p" are the two most common history abbreviations in
the corpus, and both tokenizers in this project split the slashed spelling into
three tokens -- so a lexicon carrying only ``h/o`` matches neither.
:mod:`umlsmatch.assertion.sections` has the same hazard for list markers and
:mod:`umlsmatch.assertion.uncertainty` for the "cannot ..." family. Hand-writing
a second spelling fixes one entry and protects no other, so the respelling
happens once, in :func:`~umlsmatch.assertion.scope.normalize_phrase`, and
``tests/test_pipeline_cues.py`` fails if any lexicon carries an entry the
tokenizer cannot produce.

Usage::

    from umlsmatch.assertion.history import history_matches

    past = history_matches(tokens, matches, section=section)
"""

from __future__ import annotations

from collections.abc import Sequence

from umlsmatch.assertion.negation import NEGEX_EXCLUSIONS
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
    "BACKWARD_SCOPE_TOKENS",
    "CANDIDATE_HISTORY_SECTIONS",
    "FORWARD_SCOPE_TOKENS",
    "HISTORY_BACKWARD_CUES",
    "HISTORY_FORWARD_CUES",
    "HISTORY_SECTIONS",
    "PRESENT_CUES",
    "PSEUDO_HISTORY_CUES",
    "history_evidence",
    "history_matches",
]

#: How far a "history of" cue reaches forward. Short, and measured: F1 across
#: 3-12 tokens is a broad, flat plateau, so this is not a tuned value.
#:
#: It sits on the **precision side** of that plateau rather than at the F1
#: optimum. A current finding wrongly filed as history is the expensive error
#: here: downstream it reads as a resolved problem, the same way a missed
#: negation reads as an asserted diagnosis in
#: :mod:`umlsmatch.assertion.negation`. A hundredth of F1 is cheap for a tenth
#: of precision on that trade.
#:
#: "History of severe COPD" is a noun phrase, not a list. The list forms would
#: need the section header, which on this corpus turned out not to work at all
#: -- see :data:`HISTORY_SECTIONS`.
FORWARD_SCOPE_TOKENS = 4

#: How far a trailing temporal cue reaches back. Tighter: "resolved" and "in
#: 2014" qualify the noun phrase immediately before them.
BACKWARD_SCOPE_TOKENS = 4

#: Phrases that put what *follows* them in the past.
#:
#: "status post" and "s/p" are included even though they are strictly about a
#: procedure having happened rather than about history-taking -- cTAKES marks
#: their objects ``historyOf``, and the surgical-history lists in these notes
#: are written almost entirely in that form.
#: The slashed abbreviations are written once, in the spelling a clinician uses.
#: :func:`~umlsmatch.assertion.scope.normalize_phrase` respells them as the
#: tokenizers emit them -- ``h | / | o`` -- which is what makes them reachable.
#:
#: Do not write the split spelling out by hand alongside it. *Both* tokenizers
#: in this project split ``h/o``, so a hand-written twin hides the fact that the
#: whole spelling never fires on its own, and the next person to add a cue
#: copies a rule that does not exist. Same caveat
#: :mod:`umlsmatch.assertion.sections` records for list markers.
_FORWARD_PHRASES = (
    "history of", "histories of", "hx of", "h/o",
    "past medical history", "past surgical history", "family history",
    "social history", "past history", "prior history", "remote history",
    "status post", "s/p",
)

#: Phrases that put what *precedes* them in the past. These trail the finding.
#:
#: "resolved", "declined" and "declines" arrive from
#: :data:`~umlsmatch.assertion.negation.NEGEX_EXCLUSIONS` rather than being
#: typed again; see :data:`_FROM_NEGEX_EXCLUSIONS`.
_BACKWARD_PHRASES = (
    "years ago", "year ago", "months ago", "month ago", "weeks ago",
    "days ago", "ago", "in the past", "in the remote past",
    "in remission", "since resolved", "has resolved", "have resolved",
)

#: The NegEx phrases :mod:`umlsmatch.assertion.negation` refused as negation
#: *because* they are temporality markers. Adopted here, which is the other
#: half of that decision: they were not dropped for being useless, they were
#: dropped for answering a different question -- this one.
#:
#: Read as a tuple rather than a frozenset so the ordering of the lexicon stays
#: deterministic; :func:`~umlsmatch.assertion.scope.phrase_set` sorts anyway.
_FROM_NEGEX_EXCLUSIONS: tuple[str, ...] = tuple(
    " ".join(phrase) for phrase in sorted(NEGEX_EXCLUSIONS)
)

#: Phrases containing a cue word that are not about the past.
#:
#: **"history of present illness" is deliberately not here**, which is the
#: opposite of the obvious call. It contains "history of" and heads the section
#: that is by definition the *present* illness, so suppressing it looks
#: self-evidently right. Measured, suppressing it costs F1:
#: cTAKES marks the concepts inside an HPI narrative ``historyOf`` freely
#: ("65 year old male with history of severe COPD who presents intubated"),
#: because a history *taken* in the HPI is still history-taking. The heading is
#: about where the text sits; the attribute is about what the sentence is
#: doing.
_PSEUDO_PHRASES = (
    "history and physical",
    "admission history", "history obtained", "history taken",
    "history limited", "poor historian", "history is limited",
    "past the", "prior to admission", "prior to arrival", "prior to bed",
    "prior to presentation",
)

#: Phrases that pull a sentence back to the present, ending a history scope.
#: "Now", "today" and "currently" are the ones that actually occur.
_PRESENT_PHRASES = (
    "now", "today", "currently", "current", "presents", "presenting",
    "on admission", "this admission", "at present", "new onset", "acutely",
    "but", "however", "except", "although", "though", "whereas",
)

HISTORY_FORWARD_CUES: Phrases = phrase_set(_FORWARD_PHRASES)
HISTORY_BACKWARD_CUES: Phrases = phrase_set(
    _BACKWARD_PHRASES + _FROM_NEGEX_EXCLUSIONS
)
PSEUDO_HISTORY_CUES: Phrases = phrase_set(_PSEUDO_PHRASES)
PRESENT_CUES: Phrases = phrase_set(_PRESENT_PHRASES)

#: All cues, for the self-reference rule, which cares about a cue's own span
#: and not about direction.
_ALL_CUES: Phrases = phrase_set(
    _FORWARD_PHRASES + _BACKWARD_PHRASES + _FROM_NEGEX_EXCLUSIONS
)

#: The sections a history rule would obviously use. Public because
#: :func:`history_matches` takes it as an opt-in override -- see
#: :data:`HISTORY_SECTIONS` for what happened when it was the default.
CANDIDATE_HISTORY_SECTIONS = frozenset(
    {
        "past_medical_history",
        "past_surgical_history",
        "family_history",
        "social_history",
    }
)

#: Sections whose body this module treats as history-taking by default:
#: **none**. Pass ``history_sections=`` to :func:`history_matches`, or
#: ``ClinicalPipeline(history_sections=True)``, to turn the rule on.
#:
#: This is a measurement, not an oversight -- but *which* measurement turns out
#: to be the whole question. Against cTAKES the rule is a clear loss. Against
#: adjudicated verdicts it is a clear gain. Same rule, same corpus, same run:
#:
#: ===================  ==============  ==============
#: reference            rule off        rule on
#: ===================  ==============  ==============
#: cTAKES agreement     P .500 R 1.000  P .304 R 1.000
#:                      **F1 .667**     **F1 .466**
#: adjudicated          P .460 R .198   P .742 R .670
#:                      **F1 .277**     **F1 .704**
#: ===================  ==============  ==============
#:
#: cTAKES does not mark a PMH problem list wholesale -- it marks the entries it
#: reads as history-taking and leaves the rest. The original conclusion was that
#: our section spans are wider than whatever it is doing. The adjudication says
#: the opposite: those entries *are* history, cTAKES misses them, and agreeing
#: with cTAKES about them is the error. 90% of this attribute's adjudicated
#: misses are bare PMH entries with no per-item cue, which is exactly what this
#: rule claims.
#:
#: **It still ships off.** The verdicts that reverse the conclusion are a model
#: pre-annotation, and one of their conventions -- a bare undated problem-list
#: entry reads as history -- is most of what the rule is being rewarded for.
#: Turning it on for that reason would be fitting the rule to the labeller.
#: docs/ADJUDICATION_RESULTS.md has the caveats; the flag is there so the
#: experiment is one argument away once a clinician has ruled.
HISTORY_SECTIONS: frozenset[str] = frozenset()


def _terminator_spans(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """Where a history cue's scope must stop: a return to the present."""
    return find_phrase_spans([t.norm for t in tokens], PRESENT_CUES)


def _history_scopes(
    tokens: Sequence[Token],
) -> tuple[list[TriggerScope], list[TriggerSpan]]:
    """Every temporal cue's scope, and the cue spans themselves."""
    terminators = _terminator_spans(tokens)
    n = len(tokens)
    forward = find_cues(tokens, forward=HISTORY_FORWARD_CUES, pseudo=PSEUDO_HISTORY_CUES)
    backward = find_cues(
        tokens, backward=HISTORY_BACKWARD_CUES, pseudo=PSEUDO_HISTORY_CUES
    )
    scopes = [
        *resolve_scopes(forward, terminators, n, max_scope=FORWARD_SCOPE_TOKENS),
        *resolve_scopes(backward, terminators, n, max_scope=BACKWARD_SCOPE_TOKENS),
    ]
    return scopes, forward + backward


def _cue_self_matches(
    tokens: Sequence[Token], matches: Sequence[Match]
) -> set[Match]:
    """Matches that overlap a cue span -- "past medical history" as a concept.

    Uses :data:`_ALL_CUES` rather than the direction-split lexicons, since a
    cue's own span is the same either way, and drops nothing to pseudo
    suppression for the same reason the ``subject`` rule does not: a mention
    that *is* "history of present illness" is a history concept even though
    that phrase suppresses the forward scope.
    """
    spans = find_phrase_spans([t.norm for t in tokens], _ALL_CUES)
    return {
        m
        for m in matches
        if any(overlaps((m.token_start, m.token_end), s) for s in spans)
    }


def _section_claims_everything(tokens: Sequence[Token]) -> bool:
    """True when a history-section window has no present-tense cue pulling it back.

    "Past Medical History: ... Now presents with chest pain" must not make the
    chest pain historical. Same restraint the ``subject`` section rule makes.
    """
    return not find_phrase_spans([t.norm for t in tokens], PRESENT_CUES)


def history_evidence(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    section: str | None = None,
    cue_self_reference: bool = False,
    history_sections: frozenset[str] | None = None,
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which cue(s) put each match in the past -- the reasons behind the call.

    Same inputs and the same decision as :func:`history_matches`; the keys are
    exactly the set that returns. A match claimed by the *section* rule maps to
    an empty tuple: it is in the set and there is no cue to name.

    That invariant is why `history_sections` is here and means the same thing it
    does in :func:`history_matches`. This function used to read
    :data:`HISTORY_SECTIONS` directly, which held only because the shipped value
    is empty and both section branches are therefore dead: pass the override to
    one function and not the other and the key set silently stops being the
    decision. Keep the two resolutions identical.
    """
    claimed = HISTORY_SECTIONS if history_sections is None else history_sections
    scopes, cues = _history_scopes(tokens)
    causes = scope_causes(matches, scopes)
    if cue_self_reference:
        for m in _cue_self_matches(tokens, matches):
            overlapping = tuple(
                c for c in cues if overlaps((m.token_start, m.token_end), (c.start, c.end))
            )
            causes[m] = overlapping + tuple(
                c for c in causes.get(m, ()) if c not in overlapping
            )
    if section in claimed and _section_claims_everything(tokens):
        for m in matches:
            causes.setdefault(m, ())
    return causes


def history_matches(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    section: str | None = None,
    cue_self_reference: bool = False,
    history_sections: frozenset[str] | None = None,
) -> frozenset[Match]:
    """The subset of `matches` that sit in a history-taking context.

    `tokens` and `matches` must come from the same sentence --
    ``Match.token_start``/``token_end`` are indices into `tokens`.

    `section` is the canonical section name the window sits in, as
    :func:`~umlsmatch.assertion.sections.section_header` spells it. ``None``
    disables the section rule, which degrades to fewer history calls rather
    than to wrong ones.

    `history_sections` overrides :data:`HISTORY_SECTIONS` -- the sections whose
    body is claimed wholesale. ``None`` uses the shipped value, which is empty;
    pass :data:`CANDIDATE_HISTORY_SECTIONS` to turn the rule on. It is a
    parameter rather than a constant because the measurement that switched it
    off used a reference that is wrong on exactly these mentions; see
    docs/ADJUDICATION_RESULTS.md. Off remains the default until a human says
    otherwise.

    `cue_self_reference` claims a mention that *is* a history phrase, e.g. the
    concept behind "past medical history". It defaults to ``False`` here and
    ``True`` in :func:`~umlsmatch.assertion.subject.family_member_matches`;
    that asymmetry is measured, see the module docstring.
    """
    claimed = HISTORY_SECTIONS if history_sections is None else history_sections
    scopes, _ = _history_scopes(tokens)
    if section in claimed and _section_claims_everything(tokens):
        return frozenset(matches)
    past = set(scope_causes(matches, scopes))
    if cue_self_reference:
        past |= _cue_self_matches(tokens, matches)
    return frozenset(past)

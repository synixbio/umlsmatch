"""Is the mention asserted under a condition -- "call if you develop chest pain".

cTAKES' ``conditional`` attribute. **This is a prototype behind an off-by-default
switch, not a shipped capability.** ``ClinicalPipeline`` leaves
``Annotation.conditional`` at ``None`` unless built with ``conditional=True``,
:class:`~umlsmatch.assertion.attributes.AttributeSpec` still reports
``supported=False`` for it, and no number from this module belongs in a results
table. See "What would make this shippable" below for the one thing that would
change that.

**Why it exists at all, when a handful of positives cannot score it.** The
corpus carries 3 ``conditional`` positives in 1,724 mentions (0.17%; it was 27
in 12,592 on the real-note corpus, the same order), so agreement with
cTAKES cannot distinguish a working rule set from a broken one. That is a fact
about the corpus and no rule set changes it. What it does not block is
*implementation*: docs/ADJUDICATION_RESULTS.md establishes that a stratified
adjudication samples *this pipeline's* positives, so precision is estimable
however few positives cTAKES found. The sampler cannot run on an attribute the
pipeline does not assess -- two of its four strata would be empty -- so
prototyping behind a flag is what breaks the circle.

The same document is the reason to expect the corpus number to be bad and to
not treat that as a verdict: ``uncertain`` scores F1 0.162 against cTAKES and
P 0.724 against adjudicated labels, because 97% of the mentions cTAKES calls
uncertain are not hedged. Nothing about ``conditional`` suggests cTAKES is more
reliable on it.

**The lexicon is adopted, not invented.** This is the condition
:mod:`umlsmatch.assertion.uncertainty` sets for implementing a rare attribute:
adopting an external cue list is a different act from writing down phrases that
fit the handful of positives in one corpus, and only the first survives contact
with another corpus. The cues below are ConText's ``HYPOTHETICAL`` modifier
category -- the same ConText that supplies the hedging cues, carried into
pyConTextNLP and medspaCy -- rather than anything derived from the 27. They
were not chosen by looking at what cTAKES marked.

**Four deliberate omissions**, each of which would have been easy and none of
which has evidence behind it:

*No backward cues.* Every hypothetical cue in ConText is a left marker of its
clause: "if", "in case", "should you". Negation and hedging both need a
backward form ("pneumonia cannot be excluded"); conditionals, as written in
these notes, do not. ``"pain medication if needed"`` is the case that argues
otherwise and it is one construction, so it waits for a count.

*No "watch for" / "monitor for".* They read as hypothetical, and
:mod:`umlsmatch.assertion.uncertainty` already claims the identical
construction ("evaluate for", "assess for", "workup for"). Splitting one
construction across two attributes on no evidence makes both unmeasurable, and
the tie goes to the attribute that was there first.

*No section rule.* "Discharge Instructions" and "Patient Instructions" are
dense with conditionals and a section rule would obviously raise recall.
:mod:`umlsmatch.assertion.history` is the argument against doing it anyway: its
section rule looked obvious, was measured, and cost F1 0.564 -> 0.430. There is
no measurement available here that could catch the same mistake.

*No apodosis terminators.* An earlier draft ended the scope at the imperative
that opens a main clause -- "if you develop chest pain, **call** the office" --
and it was wrong twice over. It fired inside the protasis it was meant to bound
("should the patient **report** chest pain", "if you **start** having chest
pain"), and the construction it was meant to catch does not need catching: the
apodosis of a clinical conditional either carries no concept at all ("call the
office") or carries a treatment that genuinely *is* conditional ("if you
develop a fever, take Tylenol"). Removing it left this module with no invented
lexicon at all, which is the property worth having. The 8-token cap is the only
bound, exactly as in NegEx.

**The one thing 27 positives buy.** A corpus run cannot say whether these rules
are right, but it can say how *often* they fire. That check is what caught the
only structural error in this module -- a trailing ``PRN`` at the end of a
medication row reaching forward into the next row, which made the rules fire
several times over the reference volume. :func:`_indication_scopes` bounds it.
That is the whole of what this corpus has contributed here, and nothing below
was chosen to make a score move.

**The liveness numbers, and what they are worth.** 20 notes,
``data/umls_ctakes_16ab.sqlite``, 1,708 aligned mentions, 3 reference
positives, 4 predicted: P 0.000, R 0.000, F1 0.000, accuracy 0.996. Both
questions a liveness check can answer come back clean -- the rules are neither
dead nor runaway -- and the F1 is the one number here that should not be read
as a result at all. A rule that never fired would score accuracy 0.998.
(A real-note corpus gives the same three zeros over 27 reference
positives and 84 predicted, at accuracy 0.991.)

The zero is worth a second look rather than a shrug, because it is exact: not
one of this module's predictions coincides with one of cTAKES'. What each
side marks is visible in the error listing and the two lists do not describe the
same attribute. This module's are PRN indications -- ``wheezing``,
``discomfort``, ``sinus``, ``weight`` -- which is what "conditional" means in a
medication list. cTAKES' are ``infusion``, ``education`` and
``pancreatic adenocarcinoma``. (On the real-note corpus the same split was far
better populated: ``pain`` (29), ``wheezing`` (13), ``sleep`` (8),
``muscle spasms``, ``nausea``, ``constipation`` here against ``sedation``,
``blood glucose``, ``colonoscopy``, ``lung transplant`` there.) Given what
docs/ADJUDICATION_RESULTS.md found when the same
comparison was adjudicated for ``uncertain`` (F1 0.162 against cTAKES, P 0.724
against human-style labels, because 97% of cTAKES' positives were not hedges),
disagreement this total is a reason to adjudicate and not a reason to conclude
anything. It is equally consistent with this module being wrong.

**Known residual.** A trailing ``PRN`` running directly into a section header
with no punctuation between them -- "LORazepam 1 mg Intravenous Q6H PRN FAMILY
HISTORY:" -- still crosses the boundary, because every token in between is a
word. It is a handful of the predictions. Fixing it means giving this module
the section state
that :mod:`umlsmatch.assertion.sections` already computes, which is a real
option and not one to take on four examples: a rule that cannot be evaluated is
how an unmeasurable attribute acquires the appearance of one. Revisit it with
the adjudication.

**The failure mode this module is actually built around** is ``if`` as a
complementizer -- "asked if he had chest pain", "unclear if the pneumonia
resolved", "to see if it recurs". That ``if`` means *whether*, asserts nothing
conditionally, and is common in a history. It is handled the way NegEx handles
"gram negative": a pseudo-cue list whose spans suppress any cue overlapping
them. :data:`PSEUDO_CONDITIONAL_CUES` is the largest lexicon here for that
reason, and it is the first place to look when precision disappoints.

**What would make this shippable.** One stratified adjudication::

    python tools/make_adjudication_set.py --attribute conditional \\
        --jsonl free_texts/json/silver.jsonl

which now builds the pipeline with this rule on, samples its positives, and
yields an estimable precision. If that precision holds up, ``supported`` and
``prototype`` flip together in
:mod:`umlsmatch.assertion.attributes` and the constructor default changes. If it
does not, this module is deleted -- which is the point of shipping it off by
default rather than on.

Usage::

    from umlsmatch.assertion.conditional import conditional_matches

    hypothetical = conditional_matches(tokens, matches)
"""

from __future__ import annotations

from collections.abc import Sequence

from umlsmatch.assertion.scope import (
    Phrases,
    TriggerSpan,
    find_cues,
    find_phrase_spans,
    phrase_set,
    resolve_scopes,
    scope_causes,
)
from umlsmatch.dictionary.matcher import Match, Token

__all__ = [
    "CLAUSE_CUES",
    "CLAUSE_SCOPE_TOKENS",
    "CONDITIONAL_TERMINATORS",
    "INDICATION_CUES",
    "INDICATION_SCOPE_TOKENS",
    "PSEUDO_CONDITIONAL_CUES",
    "conditional_evidence",
    "conditional_matches",
]

#: How far a clause cue reaches forward.
#:
#: Borrowed from :mod:`umlsmatch.assertion.negation`, not swept here, and the
#: distinction matters: a cap is a per-attribute tuning decision and 27
#: positives cannot make one. Negation's 8 is the defensible thing to borrow
#: because what these cues govern is a *clause* ("if you develop chest pain"),
#: the same unit negation scopes over, rather than the noun phrase
#: :mod:`umlsmatch.assertion.uncertainty` caps at 6.
CLAUSE_SCOPE_TOKENS = 8

#: How far an indication cue reaches forward. "PRN pain", "as needed for
#: nausea" -- a bare noun phrase, and a short one.
INDICATION_SCOPE_TOKENS = 4

#: ConText's ``HYPOTHETICAL`` cues that open a subordinate clause.
#:
#: ``if`` alone subsumes ``if you``, ``if he``, ``if the patient`` and the rest
#: of that family -- :func:`~umlsmatch.assertion.scope.find_phrase_spans` matches
#: at every offset, so a one-token cue fires inside every longer phrase
#: containing it. Listing the longer forms would add entries that can never be
#: the span that fired and would make the pseudo-cue list below look shorter
#: than it is relative to what it must suppress.
#:
#: Bare ``should`` is deliberately absent where ``should you`` is present.
#: "Should you develop a fever" is conditional; "the patient should take
#: aspirin" is a recommendation and by far the commoner use in these notes.
#: ConText lists the pronoun forms for exactly this reason.
_CLAUSE_PHRASES = (
    "if",
    "in case",
    "in the event",
    "unless",
    "whenever",
    "should you", "should he", "should she", "should they",
    "should the patient", "should there be", "should symptoms", "should any",
    "contingent on", "provided that", "hypothetically",
)

#: The PRN family: a drug's indication, which is conditional by definition.
#:
#: Split from the clause cues because it governs a different unit and has to be
#: bounded differently -- see :func:`_indication_scopes`. This is the family
#: that produces most of what this module marks in a real note, because a
#: medication list is mostly what a real note is.
_INDICATION_PHRASES = (
    "as needed", "as needed for", "prn", "prn for",
)

#: Phrases containing a cue word that assert nothing conditionally.
#:
#: Almost all of this is one problem: ``if`` used as a complementizer, where it
#: means *whether*. "Asked if he had chest pain" reports a question, "unclear if
#: the pneumonia resolved" reports uncertainty -- neither makes the mention
#: hypothetical, and both are ordinary history-taking prose. Suppression is by
#: span overlap, so a two-token entry here beats the one-token ``if`` inside it.
#:
#: ``even if`` and ``as if`` are the other two: concessive and comparative, not
#: conditional.
_PSEUDO_PHRASES = (
    "as if", "even if", "if any", "if at all",
    "ask if", "asks if", "asked if", "asking if",
    "see if", "to see if", "seeing if",
    "check if", "checks if", "checked if", "checking if",
    "determine if", "determines if", "determined if", "determining if",
    "know if", "knows if", "knew if", "knowing if",
    "tell if", "tells if", "told if", "telling if",
    "unclear if", "unsure if", "uncertain if", "unknown if", "not sure if",
    "wonder if", "wonders if", "wondered if", "wondering if",
    "assess if", "evaluate if", "verify if", "confirm if",
    "question if", "questions if", "questioned if",
    "inquire if", "inquires if", "inquired if",
    "doubt if", "doubts if", "find out if", "found out if",
    "note if", "noted if", "document if", "documented if",
    # "as needed" adjusting a plan is not a conditional assertion about a
    # finding: "titrate as needed", "adjust dose as needed".
    "titrate as needed", "adjust as needed", "repeat as needed",
)

#: Phrases that end the conditional clause. ConText's own termination set,
#: unchanged and unextended -- see the module docstring on why there is no
#: apodosis list here.
_TERMINATOR_PHRASES = (
    "but", "however", "though", "although", "except", "otherwise",
    "yet", "still", "aside from", "apart from",
)

CLAUSE_CUES: Phrases = phrase_set(_CLAUSE_PHRASES)
INDICATION_CUES: Phrases = phrase_set(_INDICATION_PHRASES)
PSEUDO_CONDITIONAL_CUES: Phrases = phrase_set(_PSEUDO_PHRASES)
CONDITIONAL_TERMINATORS: Phrases = phrase_set(_TERMINATOR_PHRASES)


def _clause_scopes(
    tokens: Sequence[Token], terminators: list[tuple[int, int]]
) -> list:
    """Scopes for the cues that open a subordinate clause."""
    cues = find_cues(tokens, forward=CLAUSE_CUES, pseudo=PSEUDO_CONDITIONAL_CUES)
    return resolve_scopes(
        cues, terminators, len(tokens), max_scope=CLAUSE_SCOPE_TOKENS
    )


def _indication_scopes(
    tokens: Sequence[Token], terminators: list[tuple[int, int]]
) -> list:
    """Scopes for the PRN family, stopped by the first non-word token.

    **This is the one rule here that a corpus run produced, and it is a fix for
    a structural error rather than a tuning choice.** With the clause bound, a
    trailing ``PRN`` at the end of a medication row reached forward into the
    *next* row: "bisacodyl 10 mg Rectal Daily PRN / dextrose in water 50%" made
    the dextrose conditional, and 180 corpus predictions against 21 reference
    positives were mostly that. The row separator is a bullet glyph in one
    corpus and something else in the next, so terminating on the glyph would
    have been fitting one export's encoding.

    What is not corpus-specific: an indication is a bare noun phrase, and a
    bare noun phrase contains no punctuation, no digits and no bullets. So the
    scope ends at the first token that is not a word. "PRN pain", "as needed
    for chest pain" and "prn nausea" all survive; "PRN" followed by a row
    break governs nothing, which is correct -- a trailing PRN has no
    indication to give.

    :attr:`Token.is_word` is set by both tokenizers in this project, so the
    rule means the same thing in a unit test and in the pipeline.
    """
    cues = find_cues(tokens, forward=INDICATION_CUES, pseudo=PSEUDO_CONDITIONAL_CUES)
    bounds = [*terminators, *((i, i + 1) for i, t in enumerate(tokens) if not t.is_word)]
    return resolve_scopes(
        cues, bounds, len(tokens), max_scope=INDICATION_SCOPE_TOKENS
    )


def _conditional_scopes(tokens: Sequence[Token]) -> list:
    """Every conditional cue's forward scope, the two families resolved apart.

    No backward pass -- see the module docstring on why there are no backward
    cues to resolve.
    """
    terminators = find_phrase_spans([t.norm for t in tokens], CONDITIONAL_TERMINATORS)
    return [
        *_clause_scopes(tokens, terminators),
        *_indication_scopes(tokens, terminators),
    ]


def conditional_evidence(
    tokens: Sequence[Token], matches: Sequence[Match]
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which cue(s) make each match conditional. Keys are exactly what
    :func:`conditional_matches` returns.

    Evidence matters more here than for the shipped attributes, not less. There
    is no corpus number to argue with, so the only way to judge this rule set is
    to read what fired and why -- which is what ``tools/diff_attributes.py`` and
    the adjudication review file both print.
    """
    return scope_causes(matches, _conditional_scopes(tokens))


def conditional_matches(
    tokens: Sequence[Token], matches: Sequence[Match]
) -> frozenset[Match]:
    """The subset of `matches` a conditional cue scopes over.

    `tokens` and `matches` must come from the same sentence --
    ``Match.token_start``/``token_end`` are indices into `tokens`.
    """
    return frozenset(conditional_evidence(tokens, matches))

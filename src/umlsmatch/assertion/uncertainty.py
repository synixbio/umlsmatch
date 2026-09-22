"""Is the assertion hedged -- possible, suspected, to be ruled out.

cTAKES' ``uncertainty`` attribute. **Do not gate on the cTAKES score (F1 0.162)
-- it measures the reference, not these rules.** 97% of the mentions cTAKES
calls uncertain are not hedged, so agreement with it is mostly a report on
cTAKES.

**The precision figure is usable; the recall figure is not.** That asymmetry is
the accurate claim, and a blunter one would be wrong. It would be easy to say
ships "as a diagnostic, not as a measured capability", on the grounds that 120
positives are too few to validate a rule set. docs/ADJUDICATION_RESULTS.md
disproved the grounds: too-few-positives is a fact about *cTAKES' positives*,
and a stratified sample of the *pipeline's* positives estimates precision
regardless of how few cTAKES found. It does, at **P 0.724** -- 72% of the solo
calls this module makes are genuinely hedged. Recall stays unusable at any
framing: 0.462 with an interval spanning 0.228-0.955, because one adjudicated
positive in the huge "neither" stratum extrapolates to ~73 corpus positives.

**Both adjudicated figures are provisional.** The verdicts are a model
pre-annotation, not a clinician's, and the blind review file is deliberately
untouched so a clinician can still judge it unanchored. Usable precision from a
model reviewer is a stronger claim than "diagnostic only" and a weaker one than
"validated"; it is the one the evidence supports, and it should be restated
once real verdicts exist.

**16 and 15 are both this attribute's positive count**, and the difference is
not a discrepancy. 16 is the count in the corpus; 15 is the count among the
1,708 mentions that *align* on CUI and overlapping span and are therefore
scorable. Published P/R/F1 are computed over the 15. Every figure below says
which it means -- see :attr:`~umlsmatch.assertion.attributes.AttributeSpec.corpus_positives`.

**Why implement it at all, when ``generic`` is declined.** The deciding test
is the lexicon, not the positive count. ConText and NegEx both carry hedging
cues, so this module adopts an external source rather than inventing phrases to
fit the positives in one corpus; ``generic`` has no cue list to adopt at all,
which is why it is declined and stays declined.

The count argument that used to lead this section has not survived. It ran:
``generic`` has 54 positives and ``conditional`` 27, too few to tell a working
rule set from a broken one, while 120 can at least do that much. The premise is
true only of *cTAKES'* positives -- a stratified adjudication samples the
*pipeline's* positives, so precision is estimable however few cTAKES found. That
is what promoted ``uncertain`` from "diagnostic" to "usable precision", and it
is the same reasoning under which :mod:`umlsmatch.assertion.conditional` is a
prototype that can be measured rather than a rule set that cannot. A rare
attribute is not unmeasurable; it is unmeasurable *against cTAKES*.

**The NegEx pseudo-negation class is the hedging lexicon.** NegEx classifies
"rule out", "r/o", "cannot be ruled out" and their 18 variants as ``pnega``:
phrases that look like negation and are not. :mod:`umlsmatch.assertion.negation`
uses that class only to *suppress* -- a negation trigger overlapping one is
dropped, which is why the bare "rule out" in its own forward list has never
fired. Those phrases are not noise; they are a different attribute, and this is
that attribute. The relationship is the same one
:mod:`umlsmatch.assertion.history` has with
:data:`~umlsmatch.assertion.negation.NEGEX_EXCLUSIONS`: what polarity threw
away, another axis catches.

**Read the reference before reading the score.** What cTAKES marks
``uncertainty`` on the 20-note corpus is ``pain`` (3), ``consolidation``,
``screening mammogram``, ``secretion clearance``, ``paroxysmal atrial
fibrillation``, ``stridor`` and ``abdominal pain`` -- confirmed findings and an
ordered test, not hedges. (On the real-note corpus it was "Taking?" in a
medication table header, lab values flagged with ``*``, and "unspecified" in ICD
boilerplate, alongside genuine hedges; the character of the noise differs, the
fact of it does not.) A score against that reference is not measuring what the
attribute name suggests. Reported anyway, with this paragraph attached, because
a hidden number is worse than a qualified one.

**The cTAKES number, and what it is worth.** 20 notes, 1,708 aligned mentions,
15 aligned reference positives: P 0.000, R 0.000, F1 0.000, accuracy 0.985. The
accuracy is the honest illustration of why the *agreement* figure carries no
capability claim -- a rule that never fired at all would score 0.991.

**The zero is exact, and it is not a regression.** Not one of this module's 10
predictions coincides with one of cTAKES' 15. What each side marks does not
describe the same attribute: this module's are hedges in an assessment --
``cutaneous melanoma`` ("concerning for"), ``ovarian torsion`` and ``acute
tubular necrosis`` and ``infective endocarditis`` (all "consistent with") --
against cTAKES' confirmed findings above. The lexicon fires 10 times, so it is
neither silently dead nor running away; that liveness check is now the whole of
what this corpus can say about it, because the two systems' zero overlap leaves
the stratified design with no ``both_positive`` stratum and it has not been
adjudicated here.

**On a real-note corpus not included here, adjudicated verdicts showed the agreement figure was
measuring the reference.** There the same rules scored F1 0.162 against cTAKES
and estimated **P 0.724 / R 0.462 / F1 0.564** against sampled verdicts, because
97% of the mentions cTAKES called uncertain were not hedged: precision usable,
recall not. That is the strongest evidence this module has, it is recorded in
docs/ADJUDICATION_RESULTS.md, and **it cannot be reproduced from anything on
disk** -- the corpus it came from is gone. Repeating it needs real notes;
``tools/make_adjudication_set.py --attribute uncertain`` builds the set, and on
the synthetic corpus it will report an empty ``both_positive`` stratum.

Usage::

    from umlsmatch.assertion.uncertainty import uncertain_matches

    hedged = uncertain_matches(tokens, matches)
"""

from __future__ import annotations

from collections.abc import Sequence

from umlsmatch.assertion.negex_triggers import NEGATION_PHRASE_TYPES
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
    "BACKWARD_SCOPE_TOKENS",
    "CERTAINTY_CUES",
    "FORWARD_SCOPE_TOKENS",
    "HEDGE_BACKWARD_CUES",
    "HEDGE_FORWARD_CUES",
    "PSEUDO_HEDGE_CUES",
    "uncertain_matches",
    "uncertainty_evidence",
]

#: How far a hedge reaches forward. "Possible pneumonia" is a noun phrase;
#: "differential includes pneumonia, PE and pleural effusion" is a short list.
#: Not tuned -- see the module docstring on why tuning this against 120
#: positives would be fitting noise.
FORWARD_SCOPE_TOKENS = 6

#: How far a trailing hedge reaches back. "Pneumonia cannot be excluded."
BACKWARD_SCOPE_TOKENS = 4

#: Hedges that qualify what follows them. Standard ConText/NegEx hedging cues.
_FORWARD_PHRASES = (
    "possible", "possibly", "probable", "probably", "likely", "unlikely",
    "suspect", "suspected", "suspicion for", "suspicious for",
    "concerning for", "concern for", "worrisome for",
    "question of", "questionable", "query",
    "cannot rule out", "can not rule out", "cannot exclude",
    "unable to exclude", "unable to rule out",
    "differential includes", "differential diagnosis includes",
    "may represent", "may be", "may have", "might be", "might have",
    "could be", "could represent", "c/w",
    "suggestive of", "suggests", "suggesting",
    "consistent with", "compatible with", "concerning",
    "appears to be", "appear to be", "presumed", "presumptive",
    "apparent", "equivocal", "borderline", "versus", "vs",
    "evaluate for", "assess for", "workup for", "work up for",
)

#: Hedges that qualify what precedes them.
#:
#: The passive "to be ruled out" forms are here as well as in NegEx's forward
#: ``pnega`` class, and the duplication is deliberate. NegEx files them as
#: *pre*-negation because its examples read "rule out pneumonia"; the passive
#: reads the other way round ("pneumonia to be ruled out"), and both spellings
#: occur. A phrase adopted in one direction only fires on half its occurrences.
_BACKWARD_PHRASES = (
    "cannot be excluded", "cannot be ruled out", "can not be ruled out",
    "to be ruled out", "is to be ruled out", "to be excluded",
    "must be ruled out", "should be ruled out",
    "is possible", "was possible", "are possible", "is suspected",
    "was suspected", "is questionable", "remains possible",
    "versus", "vs", "is likely", "was likely", "is probable",
)

#: NegEx's pseudo-negation class, which is the "rule out" hedging family.
#: Adopted whole; see the module docstring on why it belongs here rather than
#: only as a suppression list in :mod:`umlsmatch.assertion.negation`.
_FROM_NEGEX_PSEUDO: tuple[str, ...] = tuple(
    sorted(
        " ".join(phrase)
        for phrase, kind in NEGATION_PHRASE_TYPES.items()
        if kind == "pnega"
    )
)

#: Phrases containing a hedge word that are not hedges.
#:
#: "Not ruled out" is the awkward one: it is in NegEx's ``pnega`` class, so it
#: arrives via :data:`_FROM_NEGEX_PSEUDO`, and it *is* a hedge. What is not a
#: hedge is "consistent with" used to mean "matches", which is most of its
#: occurrences in a radiology impression ("findings consistent with prior
#: study") -- but removing it entirely loses the genuine diagnostic use, so the
#: specific reporting idioms are suppressed instead.
_PSEUDO_PHRASES = (
    "consistent with prior", "consistent with previous", "consistent with age",
    "consistent with the", "compatible with prior",
    "vs.", "possible side effects", "possible causes",
    "as possible", "if possible", "when possible", "as soon as possible",
    "evaluate for the", "no suspicion",
)

#: Phrases that pull a sentence back to a definite assertion, ending a hedge.
_CERTAINTY_PHRASES = (
    "confirmed", "definite", "definitely", "definitive", "known",
    "diagnosed with", "established", "proven", "documented",
    "but", "however", "except", "although", "though",
)

HEDGE_FORWARD_CUES: Phrases = phrase_set(_FORWARD_PHRASES + _FROM_NEGEX_PSEUDO)
HEDGE_BACKWARD_CUES: Phrases = phrase_set(_BACKWARD_PHRASES)
PSEUDO_HEDGE_CUES: Phrases = phrase_set(_PSEUDO_PHRASES)
CERTAINTY_CUES: Phrases = phrase_set(_CERTAINTY_PHRASES)


def _hedge_scopes(tokens: Sequence[Token]) -> list:
    """Every hedge's scope, forward and backward windows resolved apart."""
    terminators = find_phrase_spans([t.norm for t in tokens], CERTAINTY_CUES)
    n = len(tokens)
    forward = find_cues(tokens, forward=HEDGE_FORWARD_CUES, pseudo=PSEUDO_HEDGE_CUES)
    backward = find_cues(tokens, backward=HEDGE_BACKWARD_CUES, pseudo=PSEUDO_HEDGE_CUES)
    return [
        *resolve_scopes(forward, terminators, n, max_scope=FORWARD_SCOPE_TOKENS),
        *resolve_scopes(backward, terminators, n, max_scope=BACKWARD_SCOPE_TOKENS),
    ]


def uncertainty_evidence(
    tokens: Sequence[Token], matches: Sequence[Match]
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which hedge(s) qualify each match. Keys are exactly what
    :func:`uncertain_matches` returns.

    No section rule and no self-reference rule, unlike ``subject`` and
    ``history_of``. Not because they were measured and rejected -- because
    agreement with cTAKES could not have told the difference, and adding a rule
    on evidence that cannot separate it from its absence is how an attribute
    acquires the appearance of having been tuned.

    That reasoning is now weaker than it was, and the module docstring says why:
    a stratified adjudication *can* estimate precision for a rule this corpus
    barely covers. Either rule could be tried and scored that way. Neither has
    been, so neither is here.
    """
    return scope_causes(matches, _hedge_scopes(tokens))


def uncertain_matches(
    tokens: Sequence[Token], matches: Sequence[Match]
) -> frozenset[Match]:
    """The subset of `matches` an uncertainty cue scopes over.

    `tokens` and `matches` must come from the same sentence --
    ``Match.token_start``/``token_end`` are indices into `tokens`.
    """
    return frozenset(uncertainty_evidence(tokens, matches))

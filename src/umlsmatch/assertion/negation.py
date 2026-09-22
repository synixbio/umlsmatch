"""Rule-based negation detection (ConText-lite), scoped to polarity only.

cTAKES computes polarity with a trained ClearTK classifier; a classifier
cannot live in a zero-dependency core, so this is rules. medspaCy would have supplied
them and will not install in this environment -- its QuickUMLS/sentence extras
(``pysimstring``, ``quicksectx``, ``PyRuSH``) need a C++ compiler that isn't
available here (same class of problem as the scispaCy note in
``pipeline/tokenizer.py``). What follows is a from-scratch, dependency-free
reimplementation of the ConText algorithm (Harkema et al. 2009, itself an
extension of Chapman et al. 2001's NegEx) over a vendored NegEx trigger
lexicon (see provenance below), scoped to polarity only.

**Honesty about provenance.** The lexicon has two sources, and they are kept
separate so neither is mistaken for the other.

The bulk of it is NegEx as MetaMapLite ships it -- 262 phrases transcribed from
``NegExKeyMap`` into :mod:`umlsmatch.assertion.negex_triggers`, which carries
NLM's required notice. That part is a verified transcription, like
``umlsmatch.dictionary.exclusions`` and unlike anything else in this module.

The rest is locally curated: 44 phrases NegEx does not carry, unioned with it
here. They are not padding. NegEx has seven post-negation triggers and this
corpus needs sixteen ("unremarkable", "was negative"), and NegEx has no reason
to know that "not elsewhere classified" is ICD billing boilerplate rather than
clinical negation -- the single largest source of wrong flags from the bare
"not" trigger on this corpus.

Only the *lexicon* is adopted. MetaMapLite's matching -- meta-token merging,
longest-phrase-at-a-position, a fixed 6-token window -- is not reproduced; scope
is resolved by the rules below.

Algorithm, applied per sentence (the same lookup window the dictionary
matcher uses):

  1. Find trigger-phrase matches (case-insensitive, over tokens, not raw
     text) from ``FORWARD_TRIGGERS`` and ``BACKWARD_TRIGGERS``, dropping any
     that overlap a ``PSEUDO_TRIGGERS`` match -- e.g. "gram negative" and
     "not ruled out" are not negation despite containing "negative"/"not".
  2. A ``forward`` trigger ("no", "denies") negates what follows it; a
     ``backward`` trigger ("ruled out", "was negative") negates what
     precedes it.
  3. A trigger's scope extends to the next ``TERMINATORS`` match (e.g.
     "but", "except"), the sentence boundary, or ``MAX_SCOPE_TOKENS`` tokens
     from the trigger -- whichever comes first.
  4. A scope additionally claims any *coordinate list* it already touches
     (:data:`COORDINATION_DEPS`), however far that list runs -- "denies chest
     pain, shortness of breath, or fever" negates "fever" even though it sits
     past ``MAX_SCOPE_TOKENS``. This step needs a parse and is skipped
     silently without one, so the rules degrade to plain NegEx rather than
     failing.
  5. A dictionary match is negated if its token span falls *entirely*
     within at least one active trigger's scope.

Step 4 is strictly additive to step 3: it can only widen a scope, never
narrow one. The window is a precision guard against a trigger running away
down a sentence, and nothing here relaxes it -- a coordinate list is a
structure the parse can actually see, which is why it is allowed to escape
a cap that exists to bound the cases it cannot.

**Steps 1-3 and 5 live in** :mod:`umlsmatch.assertion.scope`, which
``subject``, ``history_of`` and ``uncertain`` also use. Step 4 and the
span-head fallback stay here: their trade-offs were settled on polarity, so
lending them to three attributes that never made that decision would
misattribute both the gains and the regressions.

**The two parse-driven rules are two switches, because they answer
differently.** ``clause_bounding`` raises precision against cTAKES and needs no
defending. ``coordination`` trades precision for recall *against cTAKES*, and
cTAKES is measurably wrong on exactly the construction coordination exists to
handle -- comma lists under a negation header. Of the 94 mentions where
enabling it newly disagrees with cTAKES, 82 have a comma between the trigger
and the mention, and all 94 sit downstream of a real trigger within 300
characters. That is consistent with cTAKES being wrong, and it is *not* proof
of it: only a human-adjudicated gold standard would settle it.

So ``coordination=True`` is a judgement, not a measurement: a missed negation
reads downstream as an asserted diagnosis, which is the more expensive error in
a clinical record. Pass ``coordination=False`` to prefer agreement with cTAKES
instead.

**On the lexicon's long tail.** NegEx's 262 phrases buy little over the
triggers that actually fire in clinical text ("no", "not", "denies", "negative
for"); its tail ("adequate to rule the patient out against") essentially never
occurs. Adopted wholesale it is a net *loss*, which is what
:data:`NEGEX_EXCLUSIONS` exists for: ``resolved`` is a temporality marker that
NegEx classifies as forward negation. The value is in the long tail of genuine
negation phrases; the cost is in a handful of terms that answer a different
question than polarity.

Usage::

    from umlsmatch.assertion.negation import negated_matches

    for tokens in annotate_sentences(text):
        matches = matcher.match(tokens)
        negated = negated_matches(tokens, matches)  # frozenset[Match]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from umlsmatch.assertion.negex_triggers import NEGATION_PHRASE_TYPES
from umlsmatch.assertion.scope import (
    Direction,
    Phrases,
    TriggerScope,
    TriggerSpan,
    find_cues,
    find_phrase_spans,
    normalize_phrase,
    resolve_scopes,
    tokenize_phrase,
)
from umlsmatch.dictionary.matcher import Match, Token

__all__ = [
    "BACKWARD_TRIGGERS",
    "COORDINATION_DEPS",
    "FORWARD_TRIGGERS",
    "MAX_SCOPE_TOKENS",
    "MODIFIER_DEPS",
    "NEGEX_EXCLUSIONS",
    "PSEUDO_TRIGGERS",
    "TERMINATORS",
    "Direction",
    "TriggerSpan",
    "find_triggers",
    "negated_matches",
    "negation_evidence",
    "trigger_text",
]

#: Maximum distance, in tokens, that a trigger's scope may reach.
#:
#: Without a cap a trigger negates everything from itself to the next
#: terminator or the end of the sentence, which over-negates: on the 20-note
#: corpus uncapped scope predicts 207 negated occurrences against a gold of 106
#: (P=0.502). NegEx caps scope at a small token window for exactly this reason.
#:
#: **The value is 8 because a real-note corpus not included here put the F1 optimum in a broad
#: plateau around 5-10 tokens. The 20-note corpus does not reproduce that**, and
#: the disagreement is left standing rather than resolved by retuning:
#:
#: ======  =====  =====  =====
#: scope   P      R      F1
#: ======  =====  =====  =====
#: 3       0.646  0.877  0.744
#: 4       0.581  0.915  0.711
#: 6       0.543  0.943  0.690
#: 8       0.520  0.972  0.678
#: 10      0.520  0.981  0.680
#: ======  =====  =====  =====
#:
#: Here F1 falls monotonically as scope widens and peaks at 3, so 8 is near the
#: worst end of the range rather than mid-plateau. Retuning to 3 on this
#: evidence would be fitting a shipped default to 20 synthetic notes -- the
#: overfitting this comment has warned against from the start -- and the
#: synthetic corpus is short prose with few of the run-on EHR table dumps a
#: scope cap exists to contain, so its optimum is not the one that generalizes.
#: Recall costs 0.10 at scope 3, which is the trade an implementer may want
#: differently; pass ``max_scope`` to ``ClinicalPipeline`` rather than editing
#: this. A real-note corpus is what would settle it.
MAX_SCOPE_TOKENS = 8

# --- Trigger lexicon ---------------------------------------------------------
# Two sources, unioned. The bulk is NegEx as MetaMapLite ships it
# (:mod:`umlsmatch.assertion.negex_triggers`); the phrases below are local
# additions NegEx does not carry, kept because each was measured to matter on
# this corpus. Neither set is a subset of the other -- NegEx has 126 forward
# triggers to these 10, while these carry 16 backward triggers to NegEx's 7.
#
# Forward (pre-) triggers: negate what comes AFTER them in their scope.
_LOCAL_FORWARD_PHRASES = (
    "no", "not", "denies", "denied", "deny", "without",
    "without evidence of", "without any evidence of",
    "no evidence of", "no sign of", "no signs of", "no indication of",
    "no suggestion of", "no history of", "no complaints of", "no complaint of",
    "no findings of", "no findings to suggest",
    "negative for", "absence of", "free of", "not associated with",
    "not demonstrate", "not appear", "not reveal", "not show",
    "rule out", "ruled out", "r/o", "unremarkable for",
    "never had", "never developed", "not experienced",
)

# Backward (post-) triggers: negate what comes BEFORE them in their scope.
_LOCAL_BACKWARD_PHRASES = (
    "not seen", "not appreciated", "not identified", "not noted",
    "not observed", "not present", "not detected",
    "negative", "is negative", "was negative", "were negative", "are negative",
    "is ruled out", "was ruled out", "were ruled out",
    "unremarkable", "was denied", "were denied",
)

# Phrases that superficially match a trigger substring but are NOT negation --
# checked first; any trigger match overlapping one of these is dropped.
_LOCAL_PSEUDO_PHRASES = (
    "gram negative", "not rule out", "not ruled out", "not been ruled out",
    # The `exclude` half of NegEx's hedging family. "Cannot exclude pneumonia"
    # asserts that pneumonia is still on the table; read as negation it records
    # the opposite, which is the expensive direction in a clinical record.
    #
    # The `rule out` spellings above were already here and the `exclude` ones
    # were not, so the two halves of one family behaved oppositely. That went
    # unnoticed because the hedge cues that should have claimed these sentences
    # were themselves unreachable -- see scope.normalize_phrase -- so nothing
    # ever contradicted the negation. `unable to exclude` is listed because
    # NegEx files ("to", "exclude") as a forward trigger in its own right.
    #
    # These suppress two mentions on a real-note corpus not included here, both of which cTAKES
    # called negated, so agreement falls by a hair. Kept anyway: cTAKES
    # is not a sound reference for assertion (see the README), and a reference
    # that reads "cannot exclude pneumonia" as the patient not having pneumonia
    # is wrong on exactly the construction this suppresses. Two mentions cannot
    # settle that either way -- the adjudication set can, and this is the kind
    # of disagreement `tools/make_adjudication_set.py --attribute negated`
    # exists to put in front of a human.
    "not exclude", "not excluded", "not been excluded", "unable to exclude",
    "no increase", "no interval change", "no significant change",
    "not cause", "not certain", "not necessarily", "without difficulty",
    "without further", "not only", "no further increase",
    # ICD/billing boilerplate, not clinical negation. These arrive in Past
    # Medical History lists copied from coded problem lists -- "Chronic airway
    # obstruction, not elsewhere classified" asserts the diagnosis, and reading
    # its "not" as negation negates the diagnosis and everything after it.
    # Measured on a real-note corpus not included here: the bare "not" trigger ran at precision
    # 0.18, and this boilerplate was its single largest source of wrong flags.
    "not elsewhere classified", "not otherwise specified",
    "not elsewhere specified", "not otherwise classified",
)

# --- Dependency relations ----------------------------------------------------
#: Relations joining coordinate siblings in a list. A trigger that reaches any
#: member of such a list reaches all of them, however long the list runs.
#:
#: This is what the flat ``MAX_SCOPE_TOKENS`` window cannot express. In
#: "Patient denies chest pain, shortness of breath, or fever" the window stops
#: one token short of "fever", flipping a denied symptom to asserted -- the
#: expensive direction of error. Review-of-Systems lists routinely run past any
#: fixed window, and they are exactly the construction where cTAKES' own
#: polarity is wrong and this pipeline's is right; covering only the head of the
#: list makes that claim only half true.
#:
#: ``appos`` is included because spaCy labels the second element of a comma list
#: as an apposition rather than a conjunct ("No fever, chills, nausea" gives
#: chills=appos, nausea=conj), so taking ``conj`` alone breaks the chain at the
#: first comma and the closure never starts.
COORDINATION_DEPS = frozenset({"conj", "appos"})

#: Relations pulled in alongside a coordinate sibling, so a multi-word mention
#: is covered whole. "weight loss" and "night sweats" reach the list through
#: their head noun; without the ``compound`` child the match span is only
#: partly covered and the mention is not negated.
#:
#: Deliberately narrow -- modifiers, not clauses. Taking the full subtree of a
#: coordinate sibling instead would be simpler and much too broad: spaCy makes
#: the list head the sentence ROOT often enough ("Negative for chills, fever,
#: night sweats" roots at "fever") that its subtree is the entire sentence.
MODIFIER_DEPS = frozenset({"compound", "amod", "nmod", "nummod", "det", "poss", "punct"})


# Phrases that end an active trigger's scope early (within a sentence).
_LOCAL_TERMINATOR_PHRASES = (
    "but", "however", "except", "apart from", "aside from", "though",
    "yet", "still", "which", "otherwise than",
)


#: NegEx trigger phrases deliberately *not* adopted.
#:
#: NegEx is a negation detector; this module is a polarity detector, and the
#: two disagree at the edges. ConText separates negation from temporality,
#: and these phrases belong to the latter -- NegEx folds them in because it
#: has no temporality axis to put them on.
#:
#: ``resolved`` is the expensive one, and it is classified ``nega``, i.e.
#: *pre*-negation, so it negates what comes after it. In a problem list --
#: "Bowel obstruction (HCC) resolved | CAD (coronary artery disease)" -- that
#: negates the *next, unrelated* diagnosis. Even read charitably as post-
#: negation, "X resolved" asserts that X happened and has since stopped; it is
#: history, not absence.
#:
#: Measured on a real-note corpus not included here: adopting NegEx wholesale flips 39 mentions,
#: 26 of them away from cTAKES, and ``resolved`` alone accounts for 19. This
#: list is what makes the vendored lexicon a net gain rather than a net loss.
#:
#: Not excluded, but worth knowing about: NegEx also carries a bare ``free``
#: as a forward trigger, which would negate the following concept in "free T4"
#: or "free fluid". It never fires that way on this corpus -- excluding it
#: changes nothing measurable -- so it is left in rather than removed on
#: speculation. Revisit if a corpus shows it firing.
NEGEX_EXCLUSIONS: frozenset[tuple[str, ...]] = frozenset(
    {("resolved",), ("declined",), ("declines",)}
)


def _negex(*types: str) -> set[tuple[str, ...]]:
    """NegEx phrases of the given types, minus :data:`NEGEX_EXCLUSIONS`."""
    return {
        p
        for p, kind in NEGATION_PHRASE_TYPES.items()
        if kind in types and p not in NEGEX_EXCLUSIONS
    }


def _combine(local: tuple[str, ...], *negex_types: str) -> Phrases:
    """Union the local phrases with NegEx's, in a deterministic order.

    Sorted rather than left in declaration order: matching collects *every*
    phrase span and a trigger is dropped if it overlaps any pseudo span, so
    order cannot change which mentions are negated -- only the order triggers
    appear in :func:`negation_evidence`, which should not depend on how two
    lexicons happened to be concatenated.

    Both sides go through :func:`~umlsmatch.assertion.scope.normalize_phrase`,
    which is what makes NegEx's ``("cannot",)`` reachable: the tokenizer emits
    ``can | not`` and the phrase as NegEx spells it can never match. The
    exclusion check above runs on NegEx's own spelling, before this, so
    :data:`NEGEX_EXCLUSIONS` is still written the way NegEx writes it.
    """
    phrases = {tokenize_phrase(p) for p in local} | _negex(*negex_types)
    return tuple(sorted(normalize_phrase(p) for p in phrases))


#: Negate what follows. NegEx ``nega`` plus the local forward phrases.
FORWARD_TRIGGERS: Phrases = _combine(_LOCAL_FORWARD_PHRASES, "nega")
#: Negate what precedes. NegEx ``negb`` plus the local backward phrases, which
#: dominate here -- NegEx carries only seven.
BACKWARD_TRIGGERS: Phrases = _combine(_LOCAL_BACKWARD_PHRASES, "negb")
#: Look like triggers but are not. Both NegEx pseudo classes plus the local
#: ones, which include the ICD boilerplate NegEx has no reason to know about.
PSEUDO_TRIGGERS: Phrases = _combine(_LOCAL_PSEUDO_PHRASES, "pnega", "pnegb")
#: End a scope early. NegEx ``conj`` plus the local terminators.
TERMINATORS: Phrases = _combine(_LOCAL_TERMINATOR_PHRASES, "conj")


class ScopedTokens(NamedTuple):
    """Every token index a trigger negates, and the bound the parse may not cross.

    ``covered`` is the window rule's range plus whatever coordination
    propagation added; ``outer`` is the trigger's uncapped scope, or ``None``
    when the parse-driven rules are inactive.
    """

    trigger: TriggerSpan
    covered: set[int]
    outer: range | None


def find_triggers(tokens: Sequence[Token]) -> tuple[list[TriggerSpan], list[tuple[int, int]]]:
    """Real (non-pseudo) trigger spans and terminator spans for one sentence."""
    triggers = find_cues(
        tokens,
        forward=FORWARD_TRIGGERS,
        backward=BACKWARD_TRIGGERS,
        pseudo=PSEUDO_TRIGGERS,
    )
    terminator_spans = find_phrase_spans([t.norm for t in tokens], TERMINATORS)
    return triggers, terminator_spans


def _trigger_scopes(
    tokens: Sequence[Token], max_scope: int | None, *, clause_bounding: bool = False
) -> list[TriggerScope]:
    """A :class:`TriggerScope` for every active trigger in the sentence.

    The *outer* pair is the same scope computed with no token cap: it stops
    only at a terminator or the sentence edge. It is the hard boundary that
    dependency propagation may not cross, and it is what keeps that step from
    quietly undoing the two rules the cap was never responsible for --
    direction ("reports pain, no fever" must not negate the pain) and
    terminators ("no fever but pain present" must not negate the pain). Both
    are fundamental to NegEx; only the cap is a tuning knob, so only the cap is
    negotiable.

    Shared by :func:`negated_matches` and :func:`negation_evidence` so the two
    can never disagree about what a trigger covers -- one of them deciding
    polarity while the other explains it would make the explanation worthless.
    """
    triggers, terminator_spans = find_triggers(tokens)
    if clause_bounding:
        # A coordinated clause bounds a trigger the way "but" does, and for the
        # same reason -- it is where the assertion stops being about the same
        # thing. Lexical terminators cannot see it; the parse can.
        terminator_spans = terminator_spans + _clause_boundaries(tokens)
    return resolve_scopes(
        triggers, terminator_spans, len(tokens), max_scope=max_scope
    )


def _clause_boundaries(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    """Coordinated *clauses*, as terminator spans. Needs a parse; empty without one.

    "Patient denies chest pain, and reports pneumonia and diabetes mellitus."
    coordinates two verbs ("denies", "reports"), and the second opens a clause
    the trigger has no business reaching into --
    but "and" is not a lexical terminator, so the window runs straight through
    it and negates the reported diagnoses. The token cap is all that limits the
    damage, which is the wrong instrument: it makes the error depend on how
    many words the first clause happens to contain.

    The test is narrow on purpose -- a ``conj`` verb whose head is *also* a
    verb. Requiring both ends to be verbal is what keeps mistagged nouns out:
    the tagger calls "diabetes" a VBZ in "pneumonia and diabetes mellitus", and
    reading that as a clause boundary would cut a genuine coordinate list in
    half. Its head ("pneumonia") is a noun, so it is correctly left alone.
    """
    return [
        (i, i + 1)
        for i, tok in enumerate(tokens)
        if tok.dep == "conj"
        and (tok.pos or "").startswith("VB")
        and 0 <= tok.head < len(tokens)
        and (tokens[tok.head].pos or "").startswith("VB")
    ]


def _coordination_components(tokens: Sequence[Token]) -> list[set[int]]:
    """Connected groups of coordinate siblings, as token-index sets.

    Undirected on purpose. A forward trigger reaches the head of a list and
    needs to reach down to its conjuncts; a backward trigger ("fever, chills
    and pneumonia were ruled out") reaches the *last* conjunct and needs to
    reach back up to the head. One component serves both.
    """
    adjacency: dict[int, list[int]] = {}
    for i, tok in enumerate(tokens):
        if tok.dep in COORDINATION_DEPS and 0 <= tok.head < len(tokens):
            adjacency.setdefault(i, []).append(tok.head)
            adjacency.setdefault(tok.head, []).append(i)

    components: list[set[int]] = []
    seen: set[int] = set()
    for start in adjacency:
        if start in seen:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        seen |= component
        components.append(component)
    return components


def _modifier_children(tokens: Sequence[Token]) -> dict[int, list[int]]:
    """Index -> the modifier children that must travel with it. See MODIFIER_DEPS."""
    children: dict[int, list[int]] = {}
    for i, tok in enumerate(tokens):
        if tok.dep in MODIFIER_DEPS and 0 <= tok.head < len(tokens):
            children.setdefault(tok.head, []).append(i)
    return children


def _scope_index_sets(
    tokens: Sequence[Token],
    scopes: list[TriggerScope],
    *,
    coordination: bool,
) -> list[ScopedTokens]:
    """Each trigger paired with every token index it negates, and its outer bound.

    With `coordination` off -- or on tokens carrying no parse, which is the
    same thing -- a scope is just its token range and this reduces exactly to
    the window rule. With a parse, a scope additionally claims any coordination
    component it already touches, **clipped to the trigger's uncapped scope**
    so the extension can cross the token cap and nothing else.

    Strictly additive with respect to the cap: a token in the window is never
    removed, so enabling this can only ever negate *more*. That matters because
    the window's failure mode is a missed negation, and a missed negation reads
    downstream as an asserted diagnosis.

    The third element is the trigger's uncapped scope, or ``None`` when the
    parse-driven rules are inactive. :func:`_causes` needs it for the span-head
    rule and must not apply that rule without a parse, because the promise that
    ``coordination=False`` and unparsed tokens both reproduce the plain window
    rule is what lets callers reproduce older measurements exactly.
    """
    windows = [
        ScopedTokens(
            scope.trigger,
            set(range(scope.start, scope.end)),
            range(scope.outer_start, scope.outer_end),
        )
        for scope in scopes
    ]
    # `dep` is empty for every token when the producer did no parsing.
    if not coordination or not any(t.dep for t in tokens):
        return [ScopedTokens(window.trigger, window.covered, None) for window in windows]

    components = _coordination_components(tokens)
    modifiers = _modifier_children(tokens)
    out: list[ScopedTokens] = []
    for trigger, covered, outer in windows:
        extra: set[int] = set()
        for component in components:
            if component & covered:
                extra |= component
                for member in component:
                    extra.update(modifiers.get(member, ()))
        out.append(ScopedTokens(trigger, covered | {i for i in extra if i in outer}, outer))
    return out


def _span_head(tokens: Sequence[Token], start: int, end: int) -> int:
    """The token in ``[start, end)`` whose syntactic head lies outside the span.

    A dictionary match is a constituent, so exactly one of its tokens normally
    attaches upward and the rest hang off it. Falls back to `start` when the
    span has no such token -- an unparsed window, or a parse with a cycle.
    """
    for i in range(start, end):
        head = tokens[i].head
        if not (start <= head < end):
            return i
    return start


def _causes(
    tokens: Sequence[Token],
    index_sets: list[ScopedTokens],
    match: Match,
) -> tuple[TriggerSpan, ...]:
    """Triggers that negate `match`: full span coverage, or its head plus the outer bound.

    The second rule exists because enumerating dependency labels does not scale.
    ``MODIFIER_DEPS`` lists the relations that must travel with a coordinate
    sibling, and it omitted ``prep``/``pobj`` -- so in "denies fever, chills,
    ..., and shortness of breath", `shortness` joined the coordinate component
    but `of` and `breath` did not, the span was only partly covered, and one of
    the most common complaints in medicine stayed affirmed past the token cap.

    Adding the two labels does not fix it: :func:`_scope_index_sets` reads
    modifier children one level down from a component *member*, and `breath`
    hangs off `of`, which is not a member. Rather than make that walk transitive
    and depend on the label list being complete forever, a match is also negated
    when its **head** is covered and the whole span sits inside the trigger's
    uncapped scope. Negating "shortness" while leaving "shortness of breath"
    affirmed is incoherent whatever the labels say.

    The head rule never crosses a terminator, a clause boundary, or the
    trigger's direction: `outer` is bounded by all three, and it is ``None``
    whenever the parse-driven rules are off.
    """
    span = range(match.token_start, match.token_end)
    head = _span_head(tokens, match.token_start, match.token_end)
    causes: list[TriggerSpan] = []
    for trigger, covered, outer in index_sets:
        covers_whole_span = all(i in covered for i in span)
        covers_head_only = (
            outer is not None and head in covered and all(i in outer for i in span)
        )
        if covers_whole_span or covers_head_only:
            causes.append(trigger)
    return tuple(causes)


def negated_matches(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    max_scope: int | None = MAX_SCOPE_TOKENS,
    coordination: bool = True,
    clause_bounding: bool = True,
) -> frozenset[Match]:
    """The subset of `matches` that fall inside an active negation trigger's scope.

    `tokens` and `matches` must come from the same sentence -- `Match.token_start`/
    `token_end` are indices into `tokens` (exactly what
    ``RareWordMatcher.match(tokens)`` returns).

    `max_scope` caps how far a trigger reaches, in tokens; pass ``None`` for the
    uncapped behavior (to the terminator or sentence edge).

    `coordination` lets a trigger follow a coordinate list past that cap
    (:data:`COORDINATION_DEPS`); `clause_bounding` stops it at a coordinated
    clause (:func:`_clause_boundaries`). Both are no-ops on tokens without a
    parse, so callers holding hand-built or
    :func:`~umlsmatch.dictionary.matcher.tokenize` tokens get the window rule
    unchanged.

    They are separate switches because they do not agree: against cTAKES
    polarity, clause bounding raises precision while coordination lowers it and
    raises recall. See the module docstring for why the default keeps both.

    `coordination` also gates the span-head rule in :func:`_causes`, so a
    multi-word mention is covered whole or not at all. Both are off without a
    parse.
    """
    scopes = _trigger_scopes(tokens, max_scope, clause_bounding=clause_bounding)
    if not scopes:
        return frozenset()
    index_sets = _scope_index_sets(tokens, scopes, coordination=coordination)
    return frozenset(m for m in matches if _causes(tokens, index_sets, m))


def negation_evidence(
    tokens: Sequence[Token],
    matches: Sequence[Match],
    *,
    max_scope: int | None = MAX_SCOPE_TOKENS,
    coordination: bool = True,
    clause_bounding: bool = True,
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which trigger(s) scope each negated match -- :func:`negated_matches` with reasons.

    Same inputs and the same decision; the keys are exactly the set
    :func:`negated_matches` returns. Diagnostic rather than hot-path: error
    analysis needs to know *which* phrase fired, since "half our negation flags
    are wrong" is only actionable once it decomposes into which triggers are
    responsible.

    A match can be scoped by more than one trigger, so the value is a tuple in
    the order :func:`find_triggers` produced (forward triggers first, then
    backward).
    """
    scopes = _trigger_scopes(tokens, max_scope, clause_bounding=clause_bounding)
    if not scopes:
        return {}
    index_sets = _scope_index_sets(tokens, scopes, coordination=coordination)
    evidence: dict[Match, tuple[TriggerSpan, ...]] = {}
    for m in matches:
        causes = _causes(tokens, index_sets, m)
        if causes:
            evidence[m] = causes
    return evidence


def trigger_text(tokens: Sequence[Token], trigger: TriggerSpan) -> str:
    """The normalized phrase a :class:`TriggerSpan` matched, for reporting."""
    return " ".join(t.norm for t in tokens[trigger.start : trigger.end])

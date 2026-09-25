"""Lexical cue matching and scope resolution, shared by the assertion attributes.

:mod:`umlsmatch.assertion.negation` grew this machinery first and owned it
alone: find the spans where a trigger phrase matches, drop the ones a
pseudo-trigger explains away, run a scope out to the next terminator or a token
cap, and ask which dictionary matches fall inside. None of that is specific to
polarity. ``subject``, ``history_of`` and ``uncertain`` are the same shape --
a cue lexicon plus a window -- and they differ from negation only in which
phrases they carry and how far a cue reaches.

**Extracted when it had a second consumer, not before.** A shared module with
one consumer is speculation. This one exists because ``subject`` needed exactly
these four functions, and ``history_of`` and ``uncertain`` then needed them
again -- the refactor was called as part of that work rather than ahead of it.

**What stayed in negation.py.** The parse-driven rules -- coordination
propagation, clause bounding, the span-head fallback -- did not move. They were
measured on polarity and their trade-offs are polarity's
(``coordination`` buys recall and costs precision *against cTAKES*, on a
construction where cTAKES is itself wrong). Lifting them here would hand three
attributes a tuning decision that was never made for them, and would make a
regression in one attribute look like a regression in all four.

What this module provides is therefore the *plain window rule*: phrase
matching, pseudo-cue suppression, terminators, a token cap, and full-span
containment. That is the whole of NegEx, and it is what the three lexical
attributes actually need.

Usage::

    from umlsmatch.assertion import scope

    cues = scope.find_cues(tokens, forward=FORWARD, backward=(), pseudo=PSEUDO)
    scopes = scope.resolve_scopes(cues, terminators, len(tokens), max_scope=6)
    hits = scope.matches_in_scope(matches, scopes)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

from umlsmatch.dictionary.matcher import Match, Token

__all__ = [
    "Direction",
    "Phrases",
    "TriggerScope",
    "TriggerSpan",
    "find_cues",
    "find_phrase_spans",
    "matches_in_scope",
    "normalize_phrase",
    "overlaps",
    "phrase_set",
    "resolve_scopes",
    "scope_causes",
    "scope_for_trigger",
    "split_word",
    "tokenize_phrase",
]

#: A lexicon: phrases already split into token tuples, so matching is a tuple
#: comparison rather than a string search. Spelled as a type because every
#: function here takes one and a bare ``tuple[tuple[str, ...], ...]`` in six
#: signatures reads as noise.
Phrases = tuple[tuple[str, ...], ...]

#: Which side of a cue its scope runs to. "no fever" is forward, "fever was
#: ruled out" is backward; nothing else is a valid direction, so it is spelled
#: as a type rather than left to a comment on a bare ``str``.
Direction = Literal["forward", "backward"]


@dataclass(frozen=True)
class TriggerSpan:
    """One cue phrase's token span, and the side of it that it governs."""

    #: Token index, inclusive.
    start: int
    #: Token index, exclusive.
    end: int
    direction: Direction


class TriggerScope(NamedTuple):
    """A cue paired with the token range it governs, capped and uncapped.

    ``start``/``end`` honour the ``max_scope`` cap; ``outer_start``/``outer_end``
    are the same scope computed without it. A tuple rather than a dataclass so
    it stays cheap to build per cue per sentence, and named rather than a bare
    5-tuple because the two pairs are easy to transpose and a transposition
    would silently widen every scope.
    """

    trigger: TriggerSpan
    start: int
    end: int
    outer_start: int
    outer_end: int


def tokenize_phrase(phrase: str) -> tuple[str, ...]:
    """Split a lexicon entry the way :attr:`Token.norm` spells a token."""
    return tuple(phrase.casefold().split())


#: Whole words the pipeline's tokenizer never emits, mapped to what it emits
#: instead. Consulted before the rules in :func:`split_word`.
#:
#: Only ``cannot`` is listed because only ``cannot`` occurs: no lexicon in this
#: package carries an ``n't`` form. Add here rather than respelling a lexicon
#: if one ever does.
_IRREGULAR_SPLITS: dict[str, tuple[str, ...]] = {
    "cannot": ("can", "not"),
}


def split_word(word: str) -> tuple[str, ...]:
    """Split one lexicon word the way the pipeline's tokenizer would.

    **This is a model of spaCy, not spaCy.** :mod:`umlsmatch.assertion.scope`
    is part of the zero-dependency core and cannot import it, so the three
    patterns below are written out and
    ``test_every_lexicon_phrase_survives_the_tokenizer`` pins them against the
    real tokenizer. When that test fails, spaCy has changed and this function is
    what to correct -- do not respell the lexicon to match a stale model.

    The patterns, all of them observed in lexicons in this package:

    ``cannot``      ``can`` + ``not``, via :data:`_IRREGULAR_SPLITS`
    ``h/o``         ``h`` + ``/`` + ``o`` -- the slash is its own token
    ``patient's``   ``patient`` + ``'s`` -- the clitic is its own token
    """
    if word in _IRREGULAR_SPLITS:
        return _IRREGULAR_SPLITS[word]
    if len(word) > 2 and word.endswith("'s"):
        return (word[:-2], "'s")
    if "/" in word and word != "/":
        parts: list[str] = []
        for i, piece in enumerate(word.split("/")):
            if i:
                parts.append("/")
            if piece:
                parts.append(piece)
        return tuple(parts)
    return (word,)


def normalize_phrase(phrase: tuple[str, ...]) -> tuple[str, ...]:
    """Respell a tokenized phrase the way the pipeline's tokenizer would.

    A phrase the tokenizer can never produce is a lexicon entry that silently
    never fires, and nothing about it looks wrong: ``("cannot", "exclude")``
    reads correctly and matches nothing, because spaCy emits
    ``can | not | exclude``.

    This is the third time that failure mode has cost this project something.
    :mod:`umlsmatch.assertion.sections` documented it for list markers;
    :mod:`umlsmatch.assertion.history` hit it with ``h/o`` and ``s/p``, where
    fixing it moved F1 from 0.508 to 0.564 -- more than every lexicon and cap
    change in that module combined. Both were fixed by writing the split
    spelling into the lexicon by hand, next to the closed one. That works and
    does not generalize: it leaves the next entry unprotected, and a
    hand-written twin makes the dead sibling beside it look alive. Sixteen
    entries across all four attributes were unreachable when this function was
    added, ten of them with no twin.

    Applied to lexicons, not to sentences -- sentence tokens already come from
    the tokenizer and are by definition in its spelling.
    """
    return tuple(part for word in phrase for part in split_word(word))


def phrase_set(phrases: Iterable[str]) -> Phrases:
    """Build a lexicon from plain strings, deduplicated and ordered.

    Entries are respelled by :func:`normalize_phrase` so that a phrase written
    ``"cannot exclude"`` matches the ``can | not | exclude`` the tokenizer
    actually emits. Deduplication happens after that, so a lexicon carrying both
    spellings -- as ``history`` does for ``h/o`` -- collapses to one entry
    instead of matching twice.

    Sorted rather than left in declaration order: matching collects *every*
    phrase span and a cue is dropped if it overlaps any pseudo span, so order
    cannot change which mentions are in scope -- only the order cues are
    reported in, which should not depend on how a lexicon was written down.
    """
    return tuple(sorted({normalize_phrase(tokenize_phrase(p)) for p in phrases}))


def find_phrase_spans(norms: Sequence[str], phrases: Phrases) -> list[tuple[int, int]]:
    """All (start, end) token spans where one of `phrases` matches exactly."""
    spans: list[tuple[int, int]] = []
    n = len(norms)
    for phrase in phrases:
        length = len(phrase)
        for i in range(n - length + 1):
            if tuple(norms[i : i + length]) == phrase:
                spans.append((i, i + length))
    return spans


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when two half-open token spans share at least one index."""
    return a[0] < b[1] and b[0] < a[1]


def find_cues(
    tokens: Sequence[Token],
    *,
    forward: Phrases = (),
    backward: Phrases = (),
    pseudo: Phrases = (),
) -> list[TriggerSpan]:
    """Real (non-pseudo) cue spans in one sentence, forward ones first.

    A cue overlapping any `pseudo` match is dropped entirely -- "gram negative"
    is not negation and "family planning" is not family history, despite each
    containing a cue word. Suppression is by overlap rather than by exact span
    so a longer pseudo phrase beats a shorter cue inside it, which is the only
    way round that works: the cue is by construction the shorter of the two.
    """
    norms = [t.norm for t in tokens]
    pseudo_spans = find_phrase_spans(norms, pseudo)

    def _live(spans: list[tuple[int, int]], direction: Direction) -> list[TriggerSpan]:
        return [
            TriggerSpan(start, end, direction)
            for start, end in spans
            if not any(overlaps((start, end), p) for p in pseudo_spans)
        ]

    cues = _live(find_phrase_spans(norms, forward), "forward")
    cues.extend(_live(find_phrase_spans(norms, backward), "backward"))
    return cues


def scope_for_trigger(
    trigger: TriggerSpan,
    terminator_spans: Sequence[tuple[int, int]],
    n_tokens: int,
    max_scope: int | None,
) -> tuple[int, int]:
    """Token span a cue governs: to the terminator, sentence edge, or cap.

    `max_scope` is required rather than defaulted. The cap is a per-attribute
    tuning decision -- negation's 8 was swept on this corpus and means nothing
    for ``history_of`` -- and a default here would quietly lend one attribute's
    measurement to another. ``None`` means uncapped.
    """
    if trigger.direction == "forward":
        bound = min((s for s, _ in terminator_spans if s >= trigger.end), default=n_tokens)
        if max_scope is not None:
            bound = min(bound, trigger.end + max_scope)
        return (trigger.end, bound)
    bound = max((e for _, e in terminator_spans if e <= trigger.start), default=0)
    if max_scope is not None:
        bound = max(bound, trigger.start - max_scope)
    return (bound, trigger.start)


def resolve_scopes(
    cues: Iterable[TriggerSpan],
    terminator_spans: Sequence[tuple[int, int]],
    n_tokens: int,
    *,
    max_scope: int | None,
) -> list[TriggerScope]:
    """A :class:`TriggerScope` for every cue: its capped and uncapped range.

    The uncapped pair is the hard boundary any later widening rule may not
    cross. It stops only at a terminator or the sentence edge, so it still
    honours direction ("reports pain, no fever" must not reach the pain) and
    terminators ("no fever but pain present" must not reach the pain). Both are
    fundamental to NegEx; only the cap is a tuning knob, so only the cap is
    negotiable.
    """
    return [
        TriggerScope(
            cue,
            *scope_for_trigger(cue, terminator_spans, n_tokens, max_scope),
            *scope_for_trigger(cue, terminator_spans, n_tokens, None),
        )
        for cue in cues
    ]


def scope_causes(
    matches: Sequence[Match], scopes: Sequence[TriggerScope]
) -> dict[Match, tuple[TriggerSpan, ...]]:
    """Which cue(s) govern each match, by plain full-span containment.

    A match counts only when **every** one of its tokens falls inside the
    scope. Partial coverage is rejected rather than rounded up: "shortness of
    breath" half inside a window is not a mention the window saw, and treating
    it as one is how a cap stops meaning anything.

    Returns only the matches with at least one cause, so an empty value is
    never stored and ``match in result`` answers the yes/no question directly.
    """
    causes: dict[Match, tuple[TriggerSpan, ...]] = {}
    for m in matches:
        span = range(m.token_start, m.token_end)
        hits = tuple(
            s.trigger for s in scopes if all(s.start <= i < s.end for i in span)
        )
        if hits:
            causes[m] = hits
    return causes


def matches_in_scope(
    matches: Sequence[Match], scopes: Sequence[TriggerScope]
) -> frozenset[Match]:
    """The subset of `matches` governed by at least one cue. See :func:`scope_causes`."""
    return frozenset(scope_causes(matches, scopes))

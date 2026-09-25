"""Do the assertion cue lexicons fire on sentences the *pipeline* produces?

Every other test of the assertion rules builds its tokens with
``umlsmatch.dictionary.matcher.tokenize``. ``ClinicalPipeline`` does not -- it
tokenizes with spaCy -- and the two disagree. A lexicon entry spelled
``"cannot exclude"`` was green in ``tests/test_uncertainty.py`` and unreachable
in the shipped pipeline, because spaCy emits ``can | not | exclude``. Sixteen
entries across all four attributes were dead the same way.

**This is the third time.** ``sections`` documented it for list markers;
``history`` hit it with ``h/o`` and ``s/p``, where the fix was worth more than
every lexicon and cap change in that module combined. Both were fixed by hand,
one lexicon at a time, and the fix did not generalize -- which is what these
tests are for.

Three kinds of test here, and they fail for different reasons:

``test_every_lexicon_is_already_in_the_tokenizer_s_spelling``
    The cheap invariant. No spaCy, no dictionary, so it runs on a fresh clone.
    Catches a lexicon that skipped ``normalize_phrase``.

``test_every_lexicon_phrase_survives_the_tokenizer``
    The authoritative one. Round-trips every phrase through spaCy, so it also
    catches ``split_word`` modelling the tokenizer *wrongly* -- including after
    a model upgrade changes what the tokenizer does.

The cue-family cases
    End to end through ``ClinicalPipeline``. They need the dictionary and the
    ``nlp`` extra, so they skip on a fresh clone. They catch a cue that is dead
    for a reason neither invariant models -- and they pin the behaviour the two
    invariants exist to protect.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"

needs_spacy = pytest.mark.skipif(
    importlib.util.find_spec("spacy") is None,
    reason="optional 'nlp' extra (spaCy) not installed",
)
needs_dictionary = pytest.mark.skipif(
    not DB.is_file(), reason=f"dictionary not built: {DB}"
)


# --- the invariant ------------------------------------------------------------


def _tokenized_phrases(value: object) -> tuple[tuple[str, ...], ...] | None:
    """`value` as a collection of token tuples, or None if it is not one.

    Two shapes occur: a tuple of token tuples (every ``phrase_set`` lexicon) and
    a dict keyed by token tuples (``sections._ALL_SPELLINGS``). Both are matched
    against token norms, so both are subject to the same reachability rule.
    """
    keys = tuple(value) if isinstance(value, (tuple, dict)) else ()
    if not keys or not all(isinstance(k, tuple) for k in keys):
        return None
    if not all(isinstance(w, str) for k in keys for w in k):
        return None
    return keys


#: Names the sweep must skip: source data, not a lexicon matched over tokens.
#:
#: ``NEGATION_PHRASE_TYPES`` is the vendored NegEx table. It is *correct* for it
#: to hold NegEx's own spelling -- ``('cannot',)``, ``('r/o',)`` -- because
#: ``negation._combine`` normalizes on the way into the lexicons that do the
#: matching. Checking it would demand the vendored table be rewritten to suit
#: our tokenizer, which is the opposite of vendoring it.
NOT_A_LEXICON = frozenset({"NEGATION_PHRASE_TYPES"})


def _every_lexicon() -> dict[str, tuple[tuple[str, ...], ...]]:
    """Every token-matched lexicon in the package, by qualified name.

    ``sections`` is in here deliberately. It is the module that first documented
    this hazard, its header spellings are matched over token norms exactly like
    a cue lexicon, and it stores them as dict keys rather than via
    ``phrase_set`` -- so nothing else would check them.
    """
    from umlsmatch.assertion import (
        conditional,
        history,
        negation,
        sections,
        subject,
        uncertainty,
    )

    found: dict[str, tuple[tuple[str, ...], ...]] = {}
    for module in (negation, uncertainty, history, subject, sections, conditional):
        for name in dir(module):
            if name.startswith("__") or name in NOT_A_LEXICON:
                continue
            phrases = _tokenized_phrases(getattr(module, name))
            if phrases is not None:
                found[f"{module.__name__}.{name}"] = phrases
    return found


def test_the_lexicon_sweep_finds_something():
    """Guard the guard: a rename must not turn the invariant into a no-op."""
    lexicons = _every_lexicon()
    assert len(lexicons) >= 8, sorted(lexicons)
    assert sum(len(v) for v in lexicons.values()) > 100
    # Every module that owns a lexicon is represented, so dropping one from the
    # import list above fails here rather than quietly narrowing the invariant.
    modules = {name.rsplit(".", 2)[-2] for name in lexicons}
    assert modules == {
        "negation", "uncertainty", "history", "subject", "sections",
        # A prototype's lexicon is subject to the same invariant: being off by
        # default is not a reason for its entries to be unreachable, and an
        # unreachable entry is how a rule set looks broken for a reason that has
        # nothing to do with the rules.
        "conditional",
    }


def test_every_lexicon_is_already_in_the_tokenizer_s_spelling():
    """Every stored phrase is a fixed point of ``normalize_phrase``.

    The cheap half of the invariant: no spaCy, no dictionary, so it runs on a
    fresh clone. It catches a lexicon built some other way -- a raw tuple
    literal, a new helper that forgets to normalize -- which would leave entries
    that silently never match.

    It cannot catch a phrase that ``split_word`` models *wrongly*; that is what
    ``test_every_lexicon_phrase_survives_the_tokenizer`` is for.
    """
    from umlsmatch.assertion.scope import normalize_phrase

    unnormalized = [
        f"{name}: {' '.join(phrase)} -> {' '.join(normalize_phrase(phrase))}"
        for name, phrases in _every_lexicon().items()
        for phrase in phrases
        if normalize_phrase(phrase) != phrase
    ]
    assert not unnormalized, "lexicon entries not in tokenizer spelling:\n" + "\n".join(
        unnormalized
    )


@needs_spacy
def test_every_lexicon_phrase_survives_the_tokenizer():
    """Round-trip every phrase through spaCy: is this a sequence it can emit?

    The authoritative half, and the one that generalizes. ``split_word`` is a
    hand-written model of spaCy -- ``scope`` is in the zero-dependency core and
    cannot import it -- so the model needs pinning against the real thing.

    This is also the test that would have caught all three historical instances
    of the bug: ``h/o`` and ``s/p`` in ``history`` (worth F1 0.508 -> 0.564 when
    fixed), the list markers in ``sections``, and the ``cannot`` family. When it
    fails, fix ``split_word`` -- do not respell the lexicon to match a stale
    model.
    """
    from umlsmatch.pipeline.tokenizer import annotate_sentences

    unreachable = []
    for name, phrases in _every_lexicon().items():
        for phrase in phrases:
            emitted = tuple(
                t.norm
                for sentence in annotate_sentences(" ".join(phrase))
                for t in sentence
            )
            if emitted != phrase:
                unreachable.append(
                    f"{name}: stores {list(phrase)}, tokenizer emits {list(emitted)}"
                )
    assert not unreachable, (
        "lexicon entries the tokenizer can never produce:\n" + "\n".join(unreachable)
    )


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("cannot", ("can", "not")),
        ("h/o", ("h", "/", "o")),
        ("r/o", ("r", "/", "o")),
        ("patient's", ("patient", "'s")),
        ("children's", ("children", "'s")),
        # Left alone: nothing to split, and the degenerate spellings must not
        # produce empty tokens or lose the word.
        ("pneumonia", ("pneumonia",)),
        ("/", ("/",)),
        ("'s", ("'s",)),
        ("", ("",)),
    ],
)
def test_split_word(word, expected):
    from umlsmatch.assertion.scope import split_word

    assert split_word(word) == expected


# --- end to end ---------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB) as p:
        yield p


def _flags(nlp, text: str, term: str) -> tuple[bool, bool]:
    """``(negated, uncertain)`` for `term` in `text`. Asserts it was found.

    A term can match several CUIs at one span; they must agree, or the caller's
    single expected value would be checking whichever one came first.
    """
    hits = [a for a in nlp.analyze(text) if a.text.casefold() == term.casefold()]
    assert hits, f"{term!r} not matched in {text!r}"
    negated = {a.negated for a in hits}
    uncertain = {a.uncertain for a in hits}
    assert len(negated) == 1 and len(uncertain) == 1, (
        f"{term!r} in {text!r} got inconsistent flags across CUIs: "
        f"negated={negated} uncertain={uncertain}"
    )
    return negated.pop(), uncertain.pop()


#: The NegEx hedging family, in the spellings a clinician writes. Every one of
#: these means "still on the table", so every one must come back hedged and
#: none of them negated. Before ``scope.normalize_phrase`` the four ``exclude``
#: spellings came back ``negated=True, uncertain=False`` -- the opposite claim.
HEDGED_NOT_NEGATED = (
    "Cannot exclude pneumonia.",
    "We cannot exclude pneumonia.",
    "Can not exclude pneumonia.",
    "Cannot rule out pneumonia.",
    "Unable to exclude pneumonia.",
    "Unable to rule out pneumonia.",
    "Pneumonia cannot be excluded.",
    "Pneumonia cannot be ruled out.",
    "Rule out pneumonia.",
    "Possible pneumonia.",
    # Abbreviated spellings. Both were dead with no twin to cover for them.
    "R/O pneumonia.",
    "Findings c/w pneumonia.",
)


@needs_dictionary
@needs_spacy
@pytest.mark.parametrize("text", HEDGED_NOT_NEGATED)
def test_the_hedging_family_is_hedged_and_not_negated(nlp, text):
    negated, uncertain = _flags(nlp, text, "pneumonia")
    assert uncertain, f"{text!r}: not hedged"
    assert not negated, f"{text!r}: negated, which asserts the opposite"


#: ``subject``'s two possessive pseudo-cues, which were both dead. A kinship
#: word inside one of these phrases does not make the mention a relative's:
#: "the patient's child has asthma" is a fact about the child, but the EHR
#: sentence it stands in for is about the patient. Both cases returned
#: ``family_member`` before the fix.
@needs_dictionary
@needs_spacy
@pytest.mark.parametrize(
    ("text", "expected_subject"),
    [
        ("The patient's child has asthma.", "patient"),
        ("Seen at children's hospital for asthma.", "patient"),
        # The cue these two must not suppress.
        ("The patient's brother has asthma.", "family_member"),
    ],
)
def test_possessive_pseudo_cues_fire_through_the_pipeline(nlp, text, expected_subject):
    hits = [a for a in nlp.analyze(text) if a.text.casefold() == "asthma"]
    assert hits, f"asthma not matched in {text!r}"
    assert {a.subject for a in hits} == {expected_subject}, text


@pytest.fixture(scope="module")
def conditional_nlp():
    """The prototype's switch, which is the only one that is off by default.

    A separate pipeline rather than a parameter on ``nlp``: the default
    configuration is itself under test above, and building the prototype into
    it would stop these files from noticing if ``conditional=True`` ever became
    the default by accident.
    """
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, conditional=True) as p:
        yield p


@needs_dictionary
@needs_spacy
def test_conditional_is_not_assessed_by_default(nlp):
    """The README's claim about this attribute, pinned at the pipeline."""
    hits = nlp.analyze("Call the office if you develop chest pain.")
    assert hits
    assert all(a.conditional is None for a in hits)


@needs_dictionary
@needs_spacy
@pytest.mark.parametrize(
    ("text", "term", "expect_conditional"),
    [
        ("Call the office if you develop chest pain.", "chest pain", True),
        ("Return to the ED in case of fever.", "fever", True),
        ("Tylenol as needed for pain.", "pain", True),
        # The complementizer, which is the failure mode the pseudo list exists
        # for: this reports a question, it does not assert conditionally.
        ("We asked if he had chest pain.", "chest pain", False),
        ("Patient has chest pain.", "chest pain", False),
    ],
)
def test_conditional_cues_fire_through_the_pipeline(
    conditional_nlp, text, term, expect_conditional
):
    hits = [
        a for a in conditional_nlp.analyze(text) if a.text.casefold() == term.casefold()
    ]
    assert hits, f"{term!r} not matched in {text!r}"
    assert {a.conditional for a in hits} == {expect_conditional}, text


@needs_dictionary
@needs_spacy
@pytest.mark.parametrize(
    ("text", "term", "expect_negated"),
    [
        ("Patient denies chest pain.", "chest pain", True),
        ("No evidence of pneumonia.", "pneumonia", True),
        ("Patient has pneumonia.", "pneumonia", False),
        # The pseudo-negation added alongside the hedge fix: "does not exclude"
        # asserts the finding is still possible, so it must not negate.
        ("This does not exclude pneumonia.", "pneumonia", False),
    ],
)
def test_negation_cues_fire_through_the_pipeline(nlp, text, term, expect_negated):
    negated, _ = _flags(nlp, text, term)
    assert negated is expect_negated, f"{text!r}: negated={negated}"

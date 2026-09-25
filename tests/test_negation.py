"""Tests for the ConText-lite negation module.

Uses synthetic Token/Match objects (via matcher.tokenize()) so trigger/scope
logic is tested independently of the built UMLS dictionary and spaCy --
this module has neither dependency, and these tests should always run.
"""

from __future__ import annotations

from umlsmatch.assertion.negation import find_triggers, negated_matches
from umlsmatch.dictionary.matcher import Match, tokenize


def _match_for(tokens, word: str) -> Match:
    """Build a single-token Match for the first occurrence of `word` in `tokens`."""
    for i, t in enumerate(tokens):
        if t.text.casefold() == word.casefold():
            return Match(
                cui="CTEST", term=word.casefold(), text=t.text,
                start=t.start, end=t.end, token_start=i, token_end=i + 1,
            )
    raise AssertionError(f"{word!r} not found in tokenized text")


def test_forward_trigger_negates_what_follows():
    tokens = tokenize("Patient denies fever and chills.")
    fever = _match_for(tokens, "fever")
    chills = _match_for(tokens, "chills")
    neg = negated_matches(tokens, [fever, chills])
    assert fever in neg
    assert chills in neg


def test_forward_trigger_does_not_negate_what_precedes():
    tokens = tokenize("Patient reports pain, no fever.")
    pain = _match_for(tokens, "pain")
    fever = _match_for(tokens, "fever")
    neg = negated_matches(tokens, [pain, fever])
    assert pain not in neg
    assert fever in neg


def test_backward_trigger_negates_what_precedes():
    tokens = tokenize("Pneumonia was ruled out.")
    pneumonia = _match_for(tokens, "Pneumonia")
    neg = negated_matches(tokens, [pneumonia])
    assert pneumonia in neg


def test_terminator_stops_forward_scope():
    tokens = tokenize("No fever but pain present.")
    fever = _match_for(tokens, "fever")
    pain = _match_for(tokens, "pain")
    neg = negated_matches(tokens, [fever, pain])
    assert fever in neg
    assert pain not in neg


def test_pseudo_trigger_blocks_gram_negative():
    tokens = tokenize("Culture showed gram negative rods.")
    triggers, _ = find_triggers(tokens)
    assert triggers == []


def test_pseudo_trigger_not_been_ruled_out_blocks_negation():
    tokens = tokenize("Pneumonia has not been ruled out.")
    pneumonia = _match_for(tokens, "Pneumonia")
    neg = negated_matches(tokens, [pneumonia])
    assert pneumonia not in neg


def test_icd_boilerplate_is_not_clinical_negation():
    """"not elsewhere classified" asserts a diagnosis -- it does not negate it.

    Past Medical History sections are routinely pasted from coded problem
    lists, so this phrasing is common. Reading its "not" as negation flipped
    the diagnosis *and* everything within scope after it; on an earlier
    corpus this is the largest single source of wrong flags from the bare
    "not" trigger, which on its own runs at precision 0.18.
    """
    tokens = tokenize("Chronic airway obstruction, not elsewhere classified, since 2009.")
    obstruction = _match_for(tokens, "obstruction")
    assert negated_matches(tokens, [obstruction]) == frozenset()


def test_icd_boilerplate_not_otherwise_specified_is_also_blocked():
    tokens = tokenize("Anemia, not otherwise specified, treated with iron.")
    anemia = _match_for(tokens, "Anemia")
    assert negated_matches(tokens, [anemia]) == frozenset()


def test_ordinary_not_still_negates():
    """The boilerplate guard must not disarm "not" generally."""
    tokens = tokenize("The patient does not have pneumonia.")
    pneumonia = _match_for(tokens, "pneumonia")
    assert pneumonia in negated_matches(tokens, [pneumonia])


def test_no_triggers_means_no_negation():
    tokens = tokenize("Patient has fever and chills.")
    fever = _match_for(tokens, "fever")
    neg = negated_matches(tokens, [fever])
    assert fever not in neg


def test_bare_negative_backward_trigger():
    tokens = tokenize("Urine culture negative.")
    culture = _match_for(tokens, "culture")
    neg = negated_matches(tokens, [culture])
    assert culture in neg


# --- Dependency-driven scope -------------------------------------------------
# These use spaCy-parsed tokens, so they are skipped without the `nlp` extra.
# The lexical tests above deliberately stay parse-free: `negated_matches` must
# keep working on tokens that carry no dependencies at all, and that contract
# is what `test_coordination_is_a_noop_without_a_parse` pins.

import importlib.util  # noqa: E402

import pytest  # noqa: E402

needs_spacy = pytest.mark.skipif(
    importlib.util.find_spec("spacy") is None,
    reason="optional 'nlp' extra (spaCy) not installed",
)


def _parsed(text: str):
    from umlsmatch.pipeline.tokenizer import annotate_sentences

    windows = annotate_sentences(text)
    assert len(windows) == 1, f"expected one window, got {len(windows)}"
    return windows[0]


def _negated_words(text: str, words, **kwargs) -> dict[str, bool]:
    tokens = _parsed(text)
    matches = [_match_for(tokens, w) for w in words]
    neg = negated_matches(tokens, matches, **kwargs)
    return {w: m in neg for w, m in zip(words, matches, strict=True)}


@needs_spacy
def test_coordination_reaches_past_the_scope_cap():
    """"fever" is 10 tokens from "denies" -- two past MAX_SCOPE_TOKENS.

    The window stops one token short and reports a denied symptom as
    asserted, which is the expensive direction of error.
    """
    text = "Patient denies chest pain, shortness of breath, or fever."
    assert _negated_words(text, ["fever"], coordination=False)["fever"] is False
    assert _negated_words(text, ["fever"])["fever"] is True


@needs_spacy
def test_coordination_chains_through_a_long_list():
    """Reachability is transitive: "constipation" links to scope via two hops.

    It is a conjunct of "diarrhea", itself a conjunct of "vomiting", which is
    the last token the window covers. Propagating only one edge would stop at
    "diarrhea" and leave the tail of every long list asserted.
    """
    text = "No fever, chills, nausea, vomiting, diarrhea, or constipation."
    got = _negated_words(text, ["diarrhea", "constipation"])
    assert got == {"diarrhea": True, "constipation": True}


@needs_spacy
def test_coordination_covers_a_multiword_sibling_whole():
    """"weight loss" needs its compound child, or the match is only half covered."""
    text = "Negative for chills, fever, night sweats, weight loss."
    assert _negated_words(text, ["loss"])["loss"] is True


# --- Prepositional mentions in coordinate lists ------------------------------


def _phrase_match(tokens, phrase: str) -> Match:
    """Build a multi-token Match for `phrase`, matched over token norms."""
    want = phrase.casefold().split()
    norms = [t.norm for t in tokens]
    for i in range(len(norms) - len(want) + 1):
        if norms[i : i + len(want)] == want:
            return Match(
                cui="CTEST", term=phrase.casefold(),
                text=phrase, start=tokens[i].start, end=tokens[i + len(want) - 1].end,
                token_start=i, token_end=i + len(want),
            )
    raise AssertionError(f"{phrase!r} not found in {norms}")


def _negated_phrase(text: str, phrase: str, **kwargs) -> bool:
    tokens = _parsed(text)
    match = _phrase_match(tokens, phrase)
    return match in negated_matches(tokens, [match], **kwargs)


@needs_spacy
def test_prepositional_mention_is_negated_past_the_cap():
    """"shortness of breath" is the canonical case MODIFIER_DEPS could not reach.

    spaCy gives `shortness` <- `of` (prep) <- `breath` (pobj). Only `shortness`
    joins the coordinate component, so the span was half-covered and one of the
    most common complaints in medicine stayed affirmed inside a denial.

    Adding prep/pobj to MODIFIER_DEPS does *not* fix this -- the modifier walk
    reads one level down from a component member, and `breath` hangs off `of`,
    which is not a member. The span-head rule is what covers it.
    """
    text = (
        "Patient denies fever, chills, nausea, vomiting, dizziness, rash, "
        "and shortness of breath."
    )
    assert _negated_phrase(text, "shortness of breath", coordination=False) is False
    assert _negated_phrase(text, "shortness of breath") is True


@needs_spacy
@pytest.mark.parametrize(
    "text",
    [
        # clause bounding: the second clause is not the trigger's business
        "Patient denies chest pain, and reports shortness of breath.",
        # terminator: "but" ends the scope
        "No fever but shortness of breath is present.",
        # direction: a forward trigger does not reach backwards
        "He reports shortness of breath, no fever.",
    ],
)
def test_span_head_rule_respects_the_guards_it_must_not_cross(text):
    """The head rule widens past the token cap and past nothing else.

    It is clipped to the trigger's *uncapped* scope, which is already bounded
    by direction, terminators and clause boundaries -- so widening coverage of
    a span cannot resurrect a negation those three rules refused.
    """
    assert _negated_phrase(text, "shortness of breath") is False


@needs_spacy
def test_span_head_rule_is_a_noop_without_coordination():
    """`coordination=False` must still reproduce the published numbers exactly."""
    text = (
        "Patient denies fever, chills, nausea, vomiting, dizziness, rash, "
        "and shortness of breath."
    )
    assert _negated_phrase(text, "shortness of breath", coordination=False) is False


def test_span_head_rule_is_a_noop_without_a_parse():
    """Parse-free tokens get the plain window rule, head rule included.

    Without this gate the head rule reads every `head == -1` as "outside the
    span", picks the first token, and silently widens scope for callers holding
    `matcher.tokenize` output.
    """
    tokens = tokenize(
        "Patient denies fever, chills, nausea, vomiting, dizziness, rash, "
        "and shortness of breath."
    )
    assert all(t.dep == "" and t.head == -1 for t in tokens)
    match = _phrase_match(tokens, "shortness of breath")
    assert negated_matches(tokens, [match]) == negated_matches(
        tokens, [match], coordination=False
    )


@needs_spacy
def test_coordination_does_not_cross_a_terminator():
    """"but" ends the scope, and coordination must not reopen it.

    "pain" is a conjunct of "fever", so an unclipped closure would drag it in
    and silently undo a rule that predates this feature.
    """
    assert _negated_words("No fever but pain present.", ["pain"])["pain"] is False


@needs_spacy
def test_coordination_does_not_run_backwards_from_a_forward_trigger():
    """A forward trigger negates what follows it -- coordination is not a loophole.

    Coordination components are undirected (a backward trigger needs to reach
    up a list), so without clipping to the trigger's own direction this negates
    the reported pain.
    """
    text = "Patient reports pain, no fever."
    got = _negated_words(text, ["pain", "fever"])
    assert got == {"pain": False, "fever": True}


@needs_spacy
def test_clause_bounding_stops_at_a_coordinated_verb():
    """"reports pneumonia" is an assertion; "and" is not a lexical terminator.

    The full sentence is load-bearing. Shortened to "...and reports
    pneumonia.", spaCy tags "reports" NNS and makes it a *compound* of
    pneumonia -- there is then no coordinated verb, the rule correctly does
    not fire, and a test built on it would be asserting the parser's mood
    rather than the rule. The parse is the weakest input here; tests that
    depend on it should use text it actually handles.
    """
    text = "Patient denies chest pain, and reports pneumonia and diabetes mellitus."
    assert _negated_words(text, ["pneumonia"], clause_bounding=False)["pneumonia"] is True
    assert _negated_words(text, ["pneumonia"])["pneumonia"] is False


@needs_spacy
def test_clause_bounding_ignores_a_mistagged_noun():
    """The tagger calls "diabetes" a VBZ here; that must not split the list.

    Requiring the conjunct's *head* to be a verb too is what protects this:
    "diabetes" hangs off "pneumonia", a noun, so it is not a clause boundary.
    Without that guard a single POS error silently truncates a coordinate list
    -- and the POS scorer reports 0.700 tag agreement on this corpus, so these
    are common, not hypothetical.

    Asserted against ``_clause_boundaries`` directly: the scope window already
    covers this short sentence, so a polarity assertion would pass whether the
    guard worked or not.
    """
    from umlsmatch.assertion.negation import _clause_boundaries

    tokens = _parsed("No pneumonia and diabetes mellitus.")
    assert [t.pos for t in tokens if t.norm == "diabetes"] == ["VBZ"], "parse changed"
    assert _clause_boundaries(tokens) == []


def test_coordination_is_a_noop_without_a_parse():
    """Parse-free tokens must behave exactly as they did before the feature.

    `matcher.tokenize` sets no dep/head, and callers pass hand-built Tokens.
    Both must get the plain window rule rather than a crash or a silent change.
    """
    tokens = tokenize("Patient denies chest pain, shortness of breath, or fever.")
    assert all(t.dep == "" and t.head == -1 for t in tokens)
    fever = _match_for(tokens, "fever")
    assert negated_matches(tokens, [fever]) == negated_matches(
        tokens, [fever], coordination=False
    )


# --- Lexicon provenance ------------------------------------------------------
# The lexicon is a union of a vendored NegEx transcription and locally curated
# phrases. Each source is easy to drop by accident -- rebuilding from NegEx
# alone would silently discard the ICD boilerplate, and reverting to the local
# tuples would discard 262 phrases -- so both ends are pinned here.


def test_negex_phrases_are_loaded():
    """The vendored NegEx table reaches every category it should.

    Compared in the *tokenizer's* spelling, not NegEx's: NegEx writes
    ``cannot`` and the tokenizer emits ``can | not``, so the lexicons store the
    split form. Asserting on NegEx's spelling would require the buckets to hold
    a phrase no sentence can ever match, which is the bug this normalization
    exists to prevent -- see ``scope.normalize_phrase``.
    """
    from umlsmatch.assertion.negation import (
        BACKWARD_TRIGGERS,
        FORWARD_TRIGGERS,
        NEGEX_EXCLUSIONS,
        PSEUDO_TRIGGERS,
        TERMINATORS,
    )
    from umlsmatch.assertion.negex_triggers import NEGATION_PHRASE_TYPES
    from umlsmatch.assertion.scope import normalize_phrase

    for kinds, bucket in [
        (("nega",), FORWARD_TRIGGERS),
        (("negb",), BACKWARD_TRIGGERS),
        (("pnega", "pnegb"), PSEUDO_TRIGGERS),
        (("conj",), TERMINATORS),
    ]:
        expected = {
            normalize_phrase(p)
            for p, k in NEGATION_PHRASE_TYPES.items()
            if k in kinds and p not in NEGEX_EXCLUSIONS
        }
        assert expected <= set(bucket), f"{kinds}: {expected - set(bucket)} missing"


def test_local_additions_survive_the_union():
    """Phrases NegEx does not carry, each measured to matter on this corpus."""
    from umlsmatch.assertion.negation import (
        BACKWARD_TRIGGERS,
        PSEUDO_TRIGGERS,
    )

    # ICD billing boilerplate: the largest source of wrong flags from bare "not".
    assert ("not", "elsewhere", "classified") in PSEUDO_TRIGGERS
    assert ("not", "otherwise", "specified") in PSEUDO_TRIGGERS
    assert ("gram", "negative") in PSEUDO_TRIGGERS
    # NegEx ships seven post-negation triggers; clinical notes need these.
    assert ("unremarkable",) in BACKWARD_TRIGGERS
    assert ("was", "negative") in BACKWARD_TRIGGERS


def test_temporality_triggers_are_excluded():
    """NegEx folds temporality into negation; this module is polarity only.

    ``resolved`` is classified ``nega`` -- forward negation -- so in a problem
    list ("Bowel obstruction resolved | CAD") it negates the *next, unrelated*
    diagnosis. Adopting it cost 0.007 precision on a real-note corpus not included here.
    """
    from umlsmatch.assertion.negation import (
        BACKWARD_TRIGGERS,
        FORWARD_TRIGGERS,
        NEGEX_EXCLUSIONS,
    )
    from umlsmatch.assertion.negex_triggers import NEGATION_PHRASE_TYPES

    assert ("resolved",) in NEGATION_PHRASE_TYPES, "upstream table changed"
    for phrase in NEGEX_EXCLUSIONS:
        assert phrase not in FORWARD_TRIGGERS
        assert phrase not in BACKWARD_TRIGGERS


def test_resolved_does_not_negate_the_following_diagnosis():
    """The concrete failure the exclusion exists to prevent."""
    tokens = tokenize("Bowel obstruction resolved. CAD and diabetes mellitus.")
    cad = _match_for(tokens, "CAD")
    assert cad not in negated_matches(tokens, [cad])


def test_lexicon_order_is_deterministic():
    """Evidence order must not depend on how two lexicons were concatenated."""
    from umlsmatch.assertion.negation import FORWARD_TRIGGERS, TERMINATORS

    for bucket in (FORWARD_TRIGGERS, TERMINATORS):
        assert list(bucket) == sorted(bucket)
        assert len(set(bucket)) == len(bucket), "duplicate phrase in lexicon"

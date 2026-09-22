"""spaCy-backed sentence splitting, tokenization and POS tagging.

Supplies the real tokens ``RareWordMatcher`` needs -- see the "POS tags are
not optional in practice" finding: without POS,
``and`` matches a genuine but spurious UMLS concept (C1706368), and only the
25-tag exclusion list (``DEFAULT_EXCLUSION_TAGS``) suppresses it.

**A deliberate deviation.** Biomedical fidelity would argue for scispaCy.
This module defaults to plain spaCy's
``en_core_web_sm`` instead:

  * scispaCy's dependency chain (``nmslib`` for its optional entity linker)
    has a history of failing to build from source on Windows.
  * This project's venv runs Python 3.14 (see ``.venv``), newer than
    scispaCy's tested range; plain spaCy 3.8 already ships cp314 wheels
    (verified at install time), so it was the lower-risk starting point.

Swap in a biomedical model later by passing ``model="en_core_sci_sm"`` (once
installed) to :func:`load_model` / :func:`annotate_sentences` -- nothing
else in this module is spaCy-model-specific. cTAKES' own POS tagger
(``mayo-pos.zip``) is trained on clinical text and will still tag some
constructs differently; treat POS disagreement as a parity-harness metric,
not a bug in this bridge.

**Whitespace tokens are dropped from the window.** spaCy emits a token for
any run of whitespace it does not attach to a neighbour -- a newline between
two words, a double space after a colon. cTAKES does the opposite: its
``NewlineToken`` is skipped outright when the lookup window's token list is
assembled (``AbstractJCasTermAnnotator.getAnnotationsInWindow``, lines
317-320 of the ``ctakes-java`` clone: ``if (baseToken instanceof
NewlineToken) { continue; }``). Note this is *stronger* than the treatment of
punctuation/number/symbol tokens, which stay in the window and are merely
barred from anchoring a lookup.

Matching that matters, because :class:`~umlsmatch.dictionary.matcher.Match`
verification joins the window's token norms with single spaces: a retained
``"\\n"`` token turns ``congestive heart failure`` into ``congestive \\n heart
failure``, which compares equal to nothing. Clinical notes are hard-wrapped
constantly, so every multi-word term crossing a line break was being lost.

**Label-colon sentence splitting.** Measured against real cTAKES output
(``tools/score_boundaries.py``), token agreement is strong (F1 0.95) but
sentence agreement stays poor (F1 0.54; it was 0.22 with no rule at all).
Cause: these notes' inline
pseudo-headers ("History of Present Illness: <narrative>",
"Medications: Aspirin, Lisinopril") have no newline or other separator --
just a colon -- yet cTAKES' clinically-trained sentence model treats the
label as its own sentence. A general-domain model has no reason to have
learned that (a colon in ordinary prose continues the clause), so
:func:`_split_on_label_colons` applies it as an explicit post-process rule:
split after any ``:`` token immediately followed by a capitalized or
numeric token. This deliberately over-splits relative to ordinary English
(e.g. "The plan is as follows: continue antibiotics." would also split) --
the goal here is parity with cTAKES' specific behavior, not general-purpose
sentence segmentation correctness. The rule runs *after* whitespace tokens
are dropped, so a label ending the line ("Medications:\\nAspirin") splits the
same way as one followed by a space -- with the whitespace token retained it
silently did not, which is the commoner spelling of the two.

**Dependency edges come free.** ``load_model`` disables only ``ner`` and
``lemmatizer``, so the parser has always been running and its output was
simply discarded. :attr:`~umlsmatch.dictionary.matcher.Token.dep`/``head``
now carry it, which is what lets negation follow a coordinate list past a
fixed token window (``assertion.negation``). Because the parse was already
being computed, populating these costs one dict build per document, not a
second model pass -- measured at no change in throughput.

Treat the parse as *weaker* evidence than the tags. These notes are
telegraphic ("Alert/Orientedx3, not in acute distress"), and a
general-domain parser degrades on them faster than its tagger does; the
negation module consumes dependencies additively for that reason.

Usage::

    from umlsmatch.pipeline.tokenizer import annotate_sentences

    for sentence_tokens in annotate_sentences(note_text):
        matches = matcher.match(sentence_tokens)
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Token as SpacyToken

from umlsmatch.dictionary.matcher import Token

__all__ = ["DEFAULT_MODEL", "annotate_sentences", "load_model", "new_model"]

DEFAULT_MODEL = "en_core_web_sm"


def _build(model: str) -> Language:
    """Load a spaCy pipeline, disabling components we don't use.

    NER and the lemmatizer add load time and aren't consumed here -- entity
    typing comes from the UMLS dictionary lookup, not spaCy's general-domain
    NER.

    The missing-model case is re-raised with the download command because the
    model cannot be a dependency of the `nlp` extra -- spaCy models are not on
    PyPI, and PyPI rejects the direct URL reference that naming one requires.
    Installing the extra therefore leaves a working spaCy with no model, and
    spaCy's own E050 ("doesn't seem to be a Python package or a valid path")
    reads like a typo in `model` rather than the one missing install step.
    """
    try:
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except OSError as exc:
        raise OSError(
            f"spaCy model {model!r} is not installed. Install it with:\n"
            f"    python -m spacy download {model}\n"
            "It is a separate step from `pip install umlsmatch[nlp]` -- spaCy "
            "models are not distributed on PyPI."
        ) from exc


@lru_cache(maxsize=4)
def load_model(model: str = DEFAULT_MODEL) -> Language:
    """A *shared* spaCy pipeline for `model`, loaded once per process.

    The cache is right for the single-threaded callers -- a script building
    several ``ClinicalPipeline`` objects in a loop, the CLI, the evaluation
    tools -- where reloading a model per construction is pure waste.

    **It is wrong for concurrent callers**, which is not obvious from the call
    site and is why :func:`new_model` exists. See it for the argument; a
    threaded consumer wants that one.
    """
    return _build(model)


def new_model(model: str = DEFAULT_MODEL) -> Language:
    """An *unshared* spaCy pipeline, bypassing :func:`load_model`'s cache.

    A ``Language`` is safe to use from one thread at a time, not from several
    at once. The neural forward pass allocates per call and is not the problem;
    the mutable lookaside state is. The ``Vocab``/``StringStore`` interns every
    string it has not seen before, and the tokenizer keeps its own cache of
    segmentations -- both written during ordinary inference, both unsynchronized,
    and both living in Cython containers whose resize is not atomic against a
    concurrent reader. Clinical text is precisely the workload that keeps
    producing strings no model has seen: dictation artifacts, lab values,
    misspellings, identifiers.

    Nothing here has been observed corrupting, and that is the point -- the
    failure mode is a rare crash or a silently wrong tokenization under load,
    not an exception at the moment of the race. It is the same argument
    :class:`~umlsmatch.service.pool.PipelinePool` already makes for not sharing
    a ``RareWordMatcher``: this was simply the one piece of per-pipeline state
    that the pool did not own, because ``load_model``'s cache handed every
    pipeline the same object.

    The cost is one model per caller, so use it when pipelines run
    concurrently and :func:`load_model` otherwise.
    """
    return _build(model)


def _split_on_label_colons(tokens: list[Token]) -> list[list[Token]]:
    """Split a sentence's tokens at any ':' token followed by a capitalized/numeric token.

    See the module docstring's "Label-colon sentence splitting" note.
    """
    out: list[list[Token]] = []
    current: list[Token] = []
    for i, tok in enumerate(tokens):
        current.append(tok)
        has_next = i + 1 < len(tokens)
        if tok.text.endswith(":") and has_next:
            next_char = tokens[i + 1].text[:1]
            if next_char.isupper() or next_char.isdigit():
                out.append(current)
                current = []
    if current:
        out.append(current)
    return out


def _resolve_dependencies(
    window: list[Token], by_offset: dict[int, SpacyToken]
) -> list[Token]:
    """Fill in each token's ``dep``/``head``, with `head` indexed within `window`.

    Dependency edges arrive from spaCy numbered against the whole document,
    but two steps renumber the tokens before they reach a lookup window:
    whitespace tokens are dropped and sentences are re-split on label colons.
    Re-resolving through character offsets -- which neither step perturbs --
    keeps this independent of both, so the splitting rules stay free to change
    without silently corrupting the parse.

    A head outside `window` becomes -1 rather than an index into the wrong
    sentence. That happens for real: splitting "Medications: Aspirin" puts the
    label and its dependent in different windows.
    """
    local = {tok.start: i for i, tok in enumerate(window)}
    out: list[Token] = []
    for tok in window:
        spacy_token = by_offset.get(tok.start)
        if spacy_token is None:  # pragma: no cover - offsets come from the doc
            out.append(tok)
            continue
        head = spacy_token.head
        out.append(
            replace(
                tok,
                dep=spacy_token.dep_,
                # A spaCy root is its own head; -1 is this module's spelling.
                head=-1 if head.i == spacy_token.i else local.get(head.idx, -1),
            )
        )
    return out


def annotate_sentences(text: str, *, model: str | Language = DEFAULT_MODEL) -> list[list[Token]]:
    """Split `text` into sentences and tokenize + POS-tag each.

    Returns one :class:`~umlsmatch.dictionary.matcher.Token` list per
    sentence -- the unit cTAKES uses as the ``RareWordMatcher`` lookup
    window. Token offsets are absolute within `text`, so matches from
    different sentences are directly comparable/mergeable. Whitespace-only
    tokens are omitted and sentences are further split on label-terminating
    colons; see the module docstring for both.

    A sentence that holds nothing but whitespace yields no token list at all,
    so the result never contains an empty window.
    """
    nlp = model if isinstance(model, Language) else load_model(model)
    doc = nlp(text)
    # Character offset -> spaCy token, for re-resolving dependency edges after
    # the two renumbering steps; see :func:`_resolve_dependencies`.
    by_offset: dict[int, SpacyToken] = {t.idx: t for t in doc}
    sentences: list[list[Token]] = []
    for sent in doc.sents:
        tokens = [
            Token(
                text=t.text,
                start=t.idx,
                end=t.idx + len(t.text),
                pos=t.tag_,
                is_word=any(c.isalpha() for c in t.text),
            )
            # Whitespace never reaches the lookup window -- see the module
            # docstring's "Whitespace tokens are dropped" note.
            for t in sent
            if not t.is_space
        ]
        if tokens:
            sentences.extend(
                _resolve_dependencies(window, by_offset)
                for window in _split_on_label_colons(tokens)
            )
    return sentences

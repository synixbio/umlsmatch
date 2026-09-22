"""Token- and sentence-boundary agreement: Python tokenizer vs. Java cTAKES.

Measures the boundary targets README's Validation section sets --
"(greater-eq)95% token-boundary agreement and (greater-eq)90%
sentence-boundary agreement with cTAKES". Complements ``umlsmatch.eval.parity``
(CUI-level agreement): this isolates whether the CUI-level gap traces back
to tokenization/sentence-splitting disagreement itself, upstream of
dictionary matching, or is purely a matching/dictionary-coverage problem.

**The sentence half is a diagnostic, not a gate.**
Sentence agreement sits at F1 0.543, well under the 0.90 target, but
substituting cTAKES' *exact* sentence spans and re-scoring moved CUI parity
by -0.001 and negation by +0.010. The criterion was a proxy, and on this
corpus it was measured not to predict downstream quality; Python emits
0.91x cTAKES' sentence count at 1.11x mean length, so the boundaries are
misaligned rather than systematically wrong. Token agreement (F1 0.950)
still tracks something real and remains worth holding at >=0.95.

Exact-span-match scoring, mirroring ``eval.parity``'s exact-CUI-match
approach: a Python span counts as a hit only if it exactly matches a Java
span's ``(begin, end)`` -- a one-character offset difference counts as a
full miss on both sides. This is strict by design (it is what "boundary
agreement" means), so expect it to read harsher than a fuzzy-overlap metric
would.

Silver-standard records must include ``"sentences"`` and ``"tokens"`` fields
-- exported by ``tools/run_java_ctakes.py`` from ``Sentence`` /
``BaseToken``-subtype annotations in the XMI. Re-run that export (``--skip-run``
works against existing XMI, no need to re-invoke Java) if your ``silver.jsonl``
predates those fields.

**Newline tokens are excluded from both sides.** cTAKES emits a
``NewlineToken`` for a line break but drops it when assembling a lookup
window (``AbstractJCasTermAnnotator.getAnnotationsInWindow``), and
:func:`~umlsmatch.pipeline.tokenizer.annotate_sentences` mirrors that by
omitting whitespace tokens. Scoring them would measure a token class neither
side's matcher ever sees, and would count every line break in the corpus as a
Python miss. Gold newline spans are identified from the record's
``"pos_tokens"`` field; a record without it falls back to ``"tokens"``
unfiltered, which reads as a large recall penalty -- re-export to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from umlsmatch.eval.metrics import PrfAggregate, PrfScore, aggregate_prf
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "AggregateBoundaryScore",
    "BoundaryScore",
    "aggregate",
    "gold_token_spans",
    "python_sentences_and_tokens",
    "score_document",
    "score_jsonl",
]

#: cTAKES token type that exists in the CAS but never in a lookup window.
_NEWLINE_TOKEN_TYPE = "NewlineToken"


@dataclass(frozen=True)
class BoundaryScore(PrfScore):
    """One document's span-set agreement. P/R/F1 come from :class:`PrfScore`."""

    source_file: str
    gold: frozenset[tuple[int, int]]
    predicted: frozenset[tuple[int, int]]

    @property
    def gold_set(self) -> frozenset[tuple[int, int]]:
        return self.gold

    @property
    def predicted_set(self) -> frozenset[tuple[int, int]]:
        return self.predicted


@dataclass(frozen=True)
class AggregateBoundaryScore(PrfAggregate):
    """Corpus-level boundary aggregate; fields are :class:`PrfAggregate`'s."""


def python_sentences_and_tokens(
    text: str,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Sentence spans and token spans the Python tokenizer produces for `text`."""
    sentences: set[tuple[int, int]] = set()
    tokens: set[tuple[int, int]] = set()
    for sentence_tokens in annotate_sentences(text):
        if not sentence_tokens:
            continue
        sentences.add((sentence_tokens[0].start, sentence_tokens[-1].end))
        tokens.update((t.start, t.end) for t in sentence_tokens)
    return frozenset(sentences), frozenset(tokens)


def gold_token_spans(record: dict) -> frozenset[tuple[int, int]]:
    """cTAKES token spans for one record, minus the newline tokens.

    Prefers ``"pos_tokens"`` (which carries each token's type) so newlines can
    be identified; falls back to the untyped ``"tokens"`` field for records
    exported before that existed.
    """
    pos_tokens = record.get("pos_tokens")
    if pos_tokens:
        return frozenset(
            (int(b), int(e))
            for b, e, tok_type, _ in pos_tokens
            if tok_type != _NEWLINE_TOKEN_TYPE
        )
    return frozenset((int(b), int(e)) for b, e in record["tokens"])


def score_document(record: dict) -> tuple[BoundaryScore, BoundaryScore]:
    """Return ``(sentence_score, token_score)`` for one silver-standard record."""
    gold_sentences = frozenset((int(b), int(e)) for b, e in record["sentences"])
    gold_tokens = gold_token_spans(record)
    pred_sentences, pred_tokens = python_sentences_and_tokens(record["text"])
    source_file = record["source_file"]
    return (
        BoundaryScore(source_file, gold_sentences, pred_sentences),
        BoundaryScore(source_file, gold_tokens, pred_tokens),
    )


def score_jsonl(jsonl_path: str | Path) -> tuple[list[BoundaryScore], list[BoundaryScore]]:
    """Score every record in a silver-standard JSONL file.

    Returns ``(sentence_scores, token_scores)``, index-aligned to each other
    and to the input file's line order.
    """
    sentence_scores: list[BoundaryScore] = []
    token_scores: list[BoundaryScore] = []
    for record in iter_records(jsonl_path):
        s_score, t_score = score_document(record)
        sentence_scores.append(s_score)
        token_scores.append(t_score)
    return sentence_scores, token_scores


def aggregate(scores: list[BoundaryScore]) -> AggregateBoundaryScore:
    """Micro (pooled counts) and macro (mean of per-doc scores) aggregates."""
    return AggregateBoundaryScore(**aggregate_prf(scores))

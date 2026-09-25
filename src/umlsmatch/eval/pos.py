"""POS agreement: Python's tagger vs. cTAKES' clinically-trained tagger.

Scored two ways, because raw tag agreement is not the quantity that matters.

The dictionary matcher only consults POS for one decision: may this token
*anchor* a lookup? A token is eligible unless its tag is in
``DEFAULT_EXCLUSION_TAGS`` (25 Penn tags: verbs, conjunctions, determiners,
pronouns, prepositions, wh-words) or it is not a word token at all. So a
disagreement between ``NN`` and ``JJ`` costs nothing -- both anchor -- while a
disagreement between ``NN`` and ``IN`` flips eligibility and can add or remove
matches.

Hence:

  * **tag agreement** -- do the taggers assign the same Penn tag?
  * **anchor agreement** -- do they agree on *eligibility*? This is the number
    that predicts matcher behaviour.

Only tokens whose character spans align exactly are compared; tokenization
differences are a separate concern already measured by ``eval.boundaries``.

cTAKES' ``NewlineToken`` is skipped entirely rather than counted as
unaligned: neither side's lookup window contains one (see
``eval.boundaries``), so including them would only inflate the "tokenization
diff" figure with a class of token that is out of scope by construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.dictionary.matcher import DEFAULT_EXCLUSION_TAGS
from umlsmatch.eval.records import iter_records

__all__ = [
    "AggregatePosScore",
    "PosScore",
    "aggregate",
    "is_anchor_tag",
    "score_document",
    "score_jsonl",
]

#: cTAKES token types that can never anchor a lookup, whatever their POS.
NON_ANCHOR_TOKEN_TYPES = frozenset(
    {"PunctuationToken", "NumToken", "SymbolToken", "ContractionToken", "NewlineToken"}
)

#: cTAKES token type dropped from the lookup window altogether, so it has no
#: Python counterpart to compare against. See the module docstring.
NEWLINE_TOKEN_TYPE = "NewlineToken"


def is_anchor_tag(pos: str | None) -> bool:
    """Would a token with this tag be eligible to anchor a dictionary lookup?

    Mirrors ``RareWordMatcher._is_anchor``: an untagged token is eligible
    (cTAKES treats a null POS as non-excluded).
    """
    return pos is None or pos not in DEFAULT_EXCLUSION_TAGS


@dataclass(frozen=True)
class PosScore:
    source_file: str
    #: Tokens whose spans aligned between the two tokenizers.
    compared: int
    #: cTAKES tokens with no exactly-aligned Python token.
    unaligned: int
    tag_matches: int
    anchor_matches: int
    #: (gold_tag, python_tag) -> count, for disagreements only.
    confusions: Counter = field(default_factory=Counter)
    #: Disagreements that flip anchor eligibility, by direction.
    anchor_flips: Counter = field(default_factory=Counter)

    @property
    def tag_agreement(self) -> float:
        return self.tag_matches / self.compared if self.compared else 0.0

    @property
    def anchor_agreement(self) -> float:
        return self.anchor_matches / self.compared if self.compared else 0.0


@dataclass(frozen=True)
class AggregatePosScore:
    documents: int
    compared: int
    unaligned: int
    tag_matches: int
    anchor_matches: int
    confusions: Counter
    anchor_flips: Counter

    @property
    def tag_agreement(self) -> float:
        return self.tag_matches / self.compared if self.compared else 0.0

    @property
    def anchor_agreement(self) -> float:
        return self.anchor_matches / self.compared if self.compared else 0.0


def score_document(record: dict) -> PosScore:
    from umlsmatch.pipeline.tokenizer import annotate_sentences

    text = record["text"]
    gold = {
        (int(b), int(e)): (tok_type, pos)
        for b, e, tok_type, pos in record.get("pos_tokens", [])
        if tok_type != NEWLINE_TOKEN_TYPE
    }

    python_by_span = {
        (t.start, t.end): t for tokens in annotate_sentences(text) for t in tokens
    }

    compared = tag_matches = anchor_matches = 0
    confusions: Counter = Counter()
    flips: Counter = Counter()

    for span, (tok_type, gold_pos) in gold.items():
        tok = python_by_span.get(span)
        if tok is None:
            continue
        compared += 1

        py_pos = tok.pos
        if gold_pos == py_pos:
            tag_matches += 1
        else:
            confusions[(gold_pos, py_pos)] += 1

        # cTAKES gates on token type first, then POS.
        gold_anchor = tok_type not in NON_ANCHOR_TOKEN_TYPES and is_anchor_tag(gold_pos)
        py_anchor = tok.is_word and is_anchor_tag(py_pos)

        if gold_anchor == py_anchor:
            anchor_matches += 1
        else:
            direction = (
                "python_anchors_extra" if py_anchor else "python_misses_anchor"
            )
            flips[direction] += 1
            flips[(direction, gold_pos, py_pos)] += 1

    return PosScore(
        source_file=record.get("source_file", "?"),
        compared=compared,
        unaligned=len(gold) - compared,
        tag_matches=tag_matches,
        anchor_matches=anchor_matches,
        confusions=confusions,
        anchor_flips=flips,
    )


def score_jsonl(jsonl_path: str | Path) -> list[PosScore]:
    return [score_document(record) for record in iter_records(jsonl_path)]


def aggregate(scores: list[PosScore]) -> AggregatePosScore:
    confusions: Counter = Counter()
    flips: Counter = Counter()
    for s in scores:
        confusions.update(s.confusions)
        flips.update(s.anchor_flips)
    return AggregatePosScore(
        documents=len(scores),
        compared=sum(s.compared for s in scores),
        unaligned=sum(s.unaligned for s in scores),
        tag_matches=sum(s.tag_matches for s in scores),
        anchor_matches=sum(s.anchor_matches for s in scores),
        confusions=confusions,
        anchor_flips=flips,
    )

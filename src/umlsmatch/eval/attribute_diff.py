"""Which attribute calls are wrong, and which cue caused each one.

:mod:`umlsmatch.eval.attributes` measures *how much* a rule set disagrees with
cTAKES. This answers *why*, the way :mod:`umlsmatch.eval.negation_diff` does for
polarity: every disagreement is attributed to the rule and the cue phrase behind
it, then aggregated across the corpus.

The point is to decide where the next hour goes. "Our subject precision is 0.69"
supports no decision. "The section rule is at 0.84, self-reference at 0.66 and
the cue window at 0.51" supports an obvious one, and that decomposition is
exactly what turned a first-cut F1 of 0.745 into 0.809 -- it showed that a
forward window from a kinship term was reaching into the *next row* of an EHR
family-history table, which no aggregate number could have shown.

Three rules can claim a mention and they are reported apart:

  * ``section``  -- the window sits in a section whose body is about relatives
    (or about the past, for ``history_of``). No cue to name.
  * ``self``     -- the mention *is* the cue phrase.
  * ``cue``      -- a cue's scope covers the mention. Attributed by phrase and
    direction, since those are the two things a lexicon edit can change.

Alignment is :func:`~umlsmatch.eval.negation_spans.align_mentions`, so these
counts reconcile with what :mod:`umlsmatch.eval.attributes` reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.assertion.attributes import AttributeSpec, get_attribute
from umlsmatch.eval.negation_spans import Occurrence, align_mentions
from umlsmatch.eval.records import iter_records

__all__ = [
    "CUE",
    "MISSED",
    "SECTION",
    "SELF",
    "CueErrors",
    "RuleErrors",
    "collect_errors",
    "top_cues",
]

#: Rule labels. ``MISSED`` is not a rule -- it is the absence of one, recorded
#: so the false negatives have somewhere to go and are counted rather than
#: quietly dropped.
SECTION = "section"
SELF = "self"
CUE = "cue"
MISSED = "(no rule fired)"

_MAX_EXAMPLES = 3


@dataclass
class CueErrors:
    """How often one cue fired under one rule, and how often it was wrong."""

    label: str
    rule: str
    correct: int = 0
    wrong: int = 0
    #: (mention text, sentence) for a few of the wrong ones.
    examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def fired(self) -> int:
        return self.correct + self.wrong

    @property
    def precision(self) -> float:
        """Share of this cue's calls that matched cTAKES. 0.0 if it never fired."""
        return self.correct / self.fired if self.fired else 0.0

    def add(self, *, correct: bool, mention: str, sentence: str) -> None:
        if correct:
            self.correct += 1
            return
        self.wrong += 1
        if len(self.examples) < _MAX_EXAMPLES:
            self.examples.append((mention, sentence))


@dataclass
class RuleErrors:
    """Corpus-wide attribution for one attribute."""

    attribute: str
    by_cue: dict[tuple[str, str], CueErrors] = field(default_factory=dict)
    #: Mention text -> count, for positives cTAKES made and we missed.
    false_negatives: Counter = field(default_factory=Counter)
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0

    def cue(self, rule: str, label: str) -> CueErrors:
        return self.by_cue.setdefault((rule, label), CueErrors(label, rule))

    def by_rule(self) -> dict[str, tuple[int, int]]:
        """Rule -> (correct, wrong). The first cut anyone should look at."""
        out: dict[str, list[int]] = {}
        for (rule, _), errors in self.by_cue.items():
            row = out.setdefault(rule, [0, 0])
            row[0] += errors.correct
            row[1] += errors.wrong
        return {rule: (c, w) for rule, (c, w) in out.items()}


def _sentence_text(tokens, text: str) -> str:
    if not tokens:
        return ""
    return " ".join(text[tokens[0].start : tokens[-1].end].split())


def collect_errors(
    jsonl_path: str | Path,
    pipeline,
    attribute: str | AttributeSpec,
    explain,
) -> RuleErrors:
    """Attribute every disagreement across a silver-standard corpus.

    `explain` is a callable ``(text) -> (occurrences, reasons)`` where
    ``occurrences`` is the usual ``(start, end, cui, value)`` list and
    ``reasons`` maps ``(start, end, cui)`` to ``(rule, label, sentence)``.
    Passing it in rather than importing a specific attribute's rules keeps this
    module from growing a branch per attribute: ``subject`` and ``history_of``
    have different cue lexicons and the same error shape.
    """
    spec = attribute if isinstance(attribute, AttributeSpec) else get_attribute(attribute)
    errors = RuleErrors(attribute=spec.name)

    for record in iter_records(jsonl_path):
        text = record["text"]
        gold: list[Occurrence] = []
        for mention in record["mentions"]:
            value = spec.gold(mention)
            begin, end = int(mention["begin"]), int(mention["end"])
            gold.extend(
                (begin, end, c["cui"], value) for c in mention["concepts"] if c.get("cui")
            )

        predicted, reasons = explain(text)
        pairs, _ = align_mentions(gold, predicted)

        for g, p in pairs:
            gold_value, predicted_value = g[3], p[3]
            if not gold_value and not predicted_value:
                continue
            mention = text[p[0] : p[1]]

            if predicted_value:
                rule, label, sentence = reasons.get(
                    (p[0], p[1], p[2]), (MISSED, MISSED, "")
                )
                if gold_value:
                    errors.n_true_positives += 1
                else:
                    errors.n_false_positives += 1
                errors.cue(rule, label).add(
                    correct=bool(gold_value), mention=mention, sentence=sentence
                )
            else:
                errors.n_false_negatives += 1
                errors.false_negatives[mention.casefold()] += 1

    return errors


def top_cues(errors: RuleErrors, n: int = 20) -> list[CueErrors]:
    """Cues ranked by how many wrong calls they caused."""
    return sorted(
        errors.by_cue.values(), key=lambda c: (-c.wrong, -c.fired, c.rule, c.label)
    )[:n]

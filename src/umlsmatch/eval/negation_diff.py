"""Which negation flags are wrong, and which trigger phrase caused each one.

``eval.negation_spans`` measures *how much* the polarity rules disagree with
Java cTAKES (precision ~0.50 -- about half of all negation flags are wrong).
This answers *why*: every disagreement is attributed to the trigger phrase whose
scope covered the mention, then aggregated across the corpus.

The point is to decide where the remaining work goes. "Half our flags are wrong"
supports no decision; it becomes actionable only once it decomposes into
something like "three phrases cause 60% of it" (fix the lexicon) versus "the
errors are spread thin across forty phrases" (the rules are at their ceiling and
the trained classifier cTAKES uses is the honest path). Those two shapes call
for opposite investments, and the difference is
measurable rather than arguable.

Two error directions are reported:

  * **false positives** -- we called it negated, Java did not. This is the
    dominant class at P~0.50 and the one that decomposes by trigger.
  * **false negatives** -- Java called it negated, we did not. These have no
    trigger to blame (that is the failure), so they are aggregated by the text
    of the mention we missed and by whether *any* trigger fired in that
    sentence -- distinguishing "no trigger matched at all" (a lexicon gap) from
    "a trigger fired but its scope did not reach" (a scope problem).

Alignment between the two sides is ``eval.negation_spans.align_mentions``, so
the counts here reconcile with the precision that module reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.assertion.negation import (
    MAX_SCOPE_TOKENS,
    find_triggers,
    negation_evidence,
    trigger_text,
)
from umlsmatch.dictionary.matcher import RareWordMatcher, Token
from umlsmatch.eval.negation_spans import Occurrence, align_mentions, gold_mention_polarity
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "NegationErrors",
    "TriggerErrors",
    "collect_errors",
    "top_false_negatives",
    "top_triggers",
]

_MAX_EXAMPLES = 3

#: Recorded for a false negative when no trigger fired anywhere in its sentence.
NO_TRIGGER = "(no trigger in sentence)"


@dataclass
class TriggerErrors:
    """How often one trigger phrase fired, and how often it was wrong."""

    trigger: str
    #: Times this trigger scoped a mention that Java also called negated.
    correct: int = 0
    #: Times it scoped a mention Java called affirmed.
    wrong: int = 0
    #: (mention text, sentence) for a few of the wrong ones.
    examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def fired(self) -> int:
        return self.correct + self.wrong

    @property
    def precision(self) -> float:
        """Share of this trigger's flags that were right. 0.0 if it never fired."""
        return self.correct / self.fired if self.fired else 0.0

    def add(self, *, correct: bool, mention: str, sentence: str) -> None:
        if correct:
            self.correct += 1
            return
        self.wrong += 1
        if len(self.examples) < _MAX_EXAMPLES:
            self.examples.append((mention, sentence))


@dataclass
class NegationErrors:
    """Corpus-wide negation error attribution."""

    by_trigger: dict[str, TriggerErrors] = field(default_factory=dict)
    #: Mention text -> count, for negations Java made and we missed.
    false_negatives: Counter = field(default_factory=Counter)
    #: Why each false negative was missed: NO_TRIGGER, or the triggers that
    #: fired in the sentence but whose scope did not reach the mention.
    false_negative_causes: Counter = field(default_factory=Counter)
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0

    def trigger(self, phrase: str) -> TriggerErrors:
        return self.by_trigger.setdefault(phrase, TriggerErrors(phrase))


def _sentence_text(tokens: list[Token], text: str) -> str:
    """The document slice the sentence covers, whitespace collapsed for display."""
    if not tokens:
        return ""
    return " ".join(text[tokens[0].start : tokens[-1].end].split())


def _analyze_document(
    record: dict,
    matcher: RareWordMatcher,
    errors: NegationErrors,
    max_scope: int | None,
) -> None:
    text = record["text"]

    # Keep each match's sentence context so an error can be shown in situ; the
    # aligner works on flat occurrence tuples and cannot carry it.
    predicted: list[Occurrence] = []
    context: dict[tuple[int, int, str], tuple[tuple[Token, ...], str]] = {}

    for tokens in annotate_sentences(text):
        matches = matcher.match(tokens)
        if not matches:
            continue
        evidence = negation_evidence(tokens, matches, max_scope=max_scope)
        triggers, _ = find_triggers(tokens)
        sentence = _sentence_text(tokens, text)
        trigger_names = tuple(trigger_text(tokens, t) for t in triggers)

        for m in matches:
            causes = evidence.get(m, ())
            predicted.append((m.start, m.end, m.cui, bool(causes)))
            # `causes` explains a false positive; `trigger_names` (every trigger
            # in the sentence, scoping or not) is what separates a false
            # negative's two causes -- no trigger at all vs. one out of reach.
            context[(m.start, m.end, m.cui)] = (causes, sentence, trigger_names, tokens)

    pairs, _ = align_mentions(gold_mention_polarity(record), predicted)

    for gold, pred in pairs:
        gold_negated, pred_negated = gold[3], pred[3]
        if not gold_negated and not pred_negated:
            continue

        key = (pred[0], pred[1], pred[2])
        causes, sentence, trigger_names, tokens = context[key]
        mention = text[pred[0] : pred[1]]

        if pred_negated and gold_negated:
            errors.n_true_positives += 1
            for c in causes:
                errors.trigger(trigger_text(tokens, c)).add(
                    correct=True, mention=mention, sentence=sentence
                )
        elif pred_negated and not gold_negated:
            errors.n_false_positives += 1
            for c in causes:
                errors.trigger(trigger_text(tokens, c)).add(
                    correct=False, mention=mention, sentence=sentence
                )
        else:  # gold negated, we did not
            errors.n_false_negatives += 1
            errors.false_negatives[mention.casefold()] += 1
            if trigger_names:
                # A trigger existed but its scope missed -- a scope problem.
                errors.false_negative_causes[
                    f"out of scope: {', '.join(sorted(set(trigger_names)))}"
                ] += 1
            else:
                errors.false_negative_causes[NO_TRIGGER] += 1


def collect_errors(
    jsonl_path: str | Path,
    db_path: str | Path,
    *,
    max_scope: int | None = MAX_SCOPE_TOKENS,
) -> NegationErrors:
    """Attribute every negation disagreement across a silver-standard corpus."""
    errors = NegationErrors()
    with RareWordMatcher(db_path) as matcher:
        for record in iter_records(jsonl_path):
            _analyze_document(record, matcher, errors, max_scope)
    return errors


def top_triggers(errors: NegationErrors, n: int = 20) -> list[TriggerErrors]:
    """Triggers ranked by how many wrong flags they caused."""
    return sorted(
        errors.by_trigger.values(), key=lambda t: (-t.wrong, -t.fired, t.trigger)
    )[:n]


def top_false_negatives(errors: NegationErrors, n: int = 20) -> list[tuple[str, int]]:
    """Mention texts we most often failed to mark negated."""
    return errors.false_negatives.most_common(n)

"""Re-run the pipeline's assertion rules, keeping the reason for each call.

``ClinicalPipeline.analyze`` returns annotations and throws away *why* each
attribute got the value it did -- rightly, since carrying an evidence tuple
per annotation would cost the hot path something every caller pays and almost
none use. The error-attribution tools need the reasons, so they re-run the
rules here.

**The risk this module creates, and how it is contained.** A second copy of the
pipeline loop can drift from the first, and a drifted explainer attributes
errors the pipeline never made. Two things stop that: window construction and
section tracking come from the same
:func:`~umlsmatch.assertion.sections.track_sections` the pipeline uses, and the
decision itself comes from the same ``*_evidence`` function whose keys the
rule's own ``*_matches`` is defined to reproduce. What is duplicated here is
the loop, not the judgement.

Each explainer returns ``(occurrences, reasons)``:

  * ``occurrences`` -- ``(start, end, cui, value)``, the shape
    :func:`~umlsmatch.eval.negation_spans.align_mentions` consumes.
  * ``reasons`` -- ``(start, end, cui)`` -> ``(rule, label, sentence)``, for
    the positives only. A negative has no rule to name; that is the point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from umlsmatch.assertion.history import history_evidence, history_matches
from umlsmatch.assertion.sections import track_sections
from umlsmatch.assertion.subject import family_member_matches, subject_evidence
from umlsmatch.assertion.uncertainty import uncertain_matches, uncertainty_evidence
from umlsmatch.dictionary.matcher import Match, RareWordMatcher, Token
from umlsmatch.eval.attribute_diff import CUE, SECTION, SELF
from umlsmatch.eval.negation_spans import Occurrence
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = [
    "EXPLAINERS",
    "Explainer",
    "history_explainer",
    "subject_explainer",
    "uncertainty_explainer",
]

#: ``(text) -> (occurrences, reasons)``. See the module docstring.
Explainer = Callable[
    [str], tuple[list[Occurrence], dict[tuple[int, int, str], tuple[str, str, str]]]
]


def _sentence_text(tokens: Sequence[Token], text: str) -> str:
    if not tokens:
        return ""
    return " ".join(text[tokens[0].start : tokens[-1].end].split())


def _cue_label(tokens: Sequence[Token], cue) -> str:
    """The cue phrase plus its direction -- the two things a lexicon edit changes."""
    phrase = " ".join(t.norm for t in tokens[cue.start : cue.end])
    return f"{phrase} ({cue.direction})"


def _explainer(matcher: RareWordMatcher, decide, explain_fn, **options) -> Explainer:
    """Build an :data:`Explainer` from a rule's ``*_matches``/``*_evidence`` pair.

    One loop for every lexical attribute. ``subject`` and ``history_of`` differ
    only in which two functions they hand in, and a copy of this loop per
    attribute would be a copy of the drift risk the module docstring describes.

    `options` are rule settings forwarded to **both** functions, which is the
    only way they may be passed. The module contract is that ``*_evidence``
    reproduces its ``*_matches`` key set; handing a setting to one and not the
    other breaks that quietly, so there is deliberately no way to spell it here.
    """

    def explain(text: str):
        occurrences: list[Occurrence] = []
        reasons: dict[tuple[int, int, str], tuple[str, str, str]] = {}

        for tokens, section in track_sections(annotate_sentences(text)):
            matches: Sequence[Match] = matcher.match(tokens)
            if not matches:
                continue
            positive = decide(tokens, matches, section=section, **options)
            evidence = explain_fn(tokens, matches, section=section, **options)
            sentence = _sentence_text(tokens, text)

            for m in matches:
                is_positive = m in positive
                occurrences.append((m.start, m.end, m.cui, is_positive))
                if not is_positive:
                    continue
                cues = evidence.get(m, ())
                if not cues:
                    # No cue to name: the section claimed it. That this branch
                    # and the section rule agree is guaranteed by the evidence
                    # function, which sets an empty tuple for exactly these and
                    # nothing else.
                    rule, label = SECTION, (section or "(unknown section)")
                else:
                    first = cues[0]
                    overlapping = first.start < m.token_end and m.token_start < first.end
                    rule = SELF if overlapping else CUE
                    label = _cue_label(tokens, first)
                reasons[(m.start, m.end, m.cui)] = (rule, label, sentence)

        return occurrences, reasons

    return explain


def subject_explainer(matcher: RareWordMatcher) -> Explainer:
    """An :data:`Explainer` for ``subject``, over a live dictionary."""
    return _explainer(matcher, family_member_matches, subject_evidence)


def history_explainer(
    matcher: RareWordMatcher, *, history_sections: frozenset[str] | None = None
) -> Explainer:
    """An :data:`Explainer` for ``history_of``, over a live dictionary.

    `history_sections` is the same override
    :func:`~umlsmatch.assertion.history.history_matches` takes: ``None`` for the
    shipped default (the rule off), or
    :data:`~umlsmatch.assertion.history.CANDIDATE_HISTORY_SECTIONS` for the
    ``clinical_recall`` behaviour. Without it this explainer could only ever
    attribute the configuration the pipeline is *not* running under when a
    caller selects that profile, which makes the error attribution a report on
    the wrong rule set.
    """
    return _explainer(
        matcher, history_matches, history_evidence, history_sections=history_sections
    )


def uncertainty_explainer(matcher: RareWordMatcher) -> Explainer:
    """An :data:`Explainer` for ``uncertain``, over a live dictionary.

    The rules take no section, so the two functions are wrapped to swallow it.
    The alternative -- giving them a ``section`` parameter they ignore -- would
    advertise a rule that does not exist.
    """
    return _explainer(
        matcher,
        lambda t, m, *, section=None: uncertain_matches(t, m),
        lambda t, m, *, section=None: uncertainty_evidence(t, m),
    )


#: Attribute name -> the explainer factory for it. Keyed by name so
#: ``tools/diff_attributes.py`` needs no branch per attribute.
#:
#: ``Callable[..., Explainer]`` rather than ``Callable[[RareWordMatcher], ...]``
#: because :func:`history_explainer` takes a keyword-only rule override and the
#: others do not. Every factory is still callable with a matcher alone, which is
#: what a caller that wants the shipped configuration should do.
EXPLAINERS: dict[str, Callable[..., Explainer]] = {
    "subject": subject_explainer,
    "history_of": history_explainer,
    "uncertain": uncertainty_explainer,
}

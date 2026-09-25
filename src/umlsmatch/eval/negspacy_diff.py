"""Second-opinion negation diff: this pipeline vs. negspaCy's NegEx.

**What this is for.** The human-adjudicated gold standard is built but
unannotated, and cTAKES polarity is not a valid reference for
negation. That leaves the negation rules with no trustworthy external
check at all. An independent NegEx implementation is not a gold standard
either -- but where two implementations of the same algorithm disagree, one of
them is usually wrong in a way that is cheap to read off, and it costs no
annotation to find out. This is a *lead generator* for lexicon and scope bugs,
not a score.

Read a disagreement count here as "look at these", never as "we are wrong this
often". Neither side is truth, and the two are not measuring quite the same
thing (see the caveats below).

**Three caveats that shape what the numbers mean.**

1. *Non-overlapping mentions only.* negspaCy decides polarity for ``doc.ents``,
   and spaCy entities may not overlap -- but this pipeline deliberately emits
   overlapping matches ("chest", "chest pain", "pain"), as cTAKES does. The
   comparison therefore runs on the longest non-overlapping subset. That is a
   real restriction: roughly half of all mentions are dropped, and nested
   mentions are exactly where scope rules are most likely to differ.

2. *Different sentence boundaries.* negspaCy scopes by spaCy's sentences; this
   pipeline re-splits them on label colons (``pipeline.tokenizer``). A
   disagreement can therefore be a segmentation difference wearing a polarity
   costume.

3. *Different lexicons.* negspaCy ships its own termset. Disagreements caused
   by a phrase one side simply does not know are the *useful* findings here --
   they are lexicon gaps, which is the cheapest class of bug this project has
   left. Disagreements caused by scope are the interesting ones.

Usage::

    from umlsmatch.eval.negspacy_diff import compare_documents

    report = compare_documents(texts, db_path="data/umls_sno_rx.sqlite")
    print(report.summary())
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from umlsmatch.analyze import Annotation, ClinicalPipeline

if TYPE_CHECKING:  # spaCy is an optional extra; the annotation must not import it
    from spacy.language import Language

__all__ = ["DiffReport", "Disagreement", "compare_documents", "load_negex"]

#: Sentinel label for mentions negspaCy could not be asked about.
_UNALIGNED = "unaligned"


def load_negex(model: str = "en_core_web_sm") -> Language:
    """Load a spaCy pipeline with negspaCy's ``negex`` component attached.

    Raises a message naming the extra rather than an opaque ImportError --
    negspaCy is an optional comparison dependency, not a runtime one, and
    nothing in the shipping pipeline imports it.
    """
    try:
        import negspacy.negation  # noqa: F401  (registers the "negex" factory)
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The negspaCy comparison needs the optional 'compare' extra. "
            'Install it with:  pip install -e ".[compare]"'
        ) from exc

    nlp = spacy.load(model)
    if "negex" not in nlp.pipe_names:
        # negex reads ent._.negex, so the NER pipe stays enabled even though
        # entities are overwritten below -- removing it changes the extension
        # registration path for no gain.
        nlp.add_pipe("negex", config={"chunk_prefix": []})
    return nlp


@dataclass(frozen=True)
class Disagreement:
    """One mention the two implementations label differently."""

    document: str
    cui: str
    start: int
    end: int
    mention: str
    ours: bool
    theirs: bool
    #: Sentence the mention sits in. PHI -- printed only on request.
    sentence: str = ""


@dataclass
class DiffReport:
    documents: int = 0
    compared: int = 0
    #: Mentions dropped because they overlap another (see caveat 1).
    skipped_overlapping: int = 0
    #: Mentions spaCy could not align to a character span.
    skipped_unaligned: int = 0
    agree_negated: int = 0
    agree_affirmed: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def we_negate_they_do_not(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.ours and not d.theirs]

    @property
    def they_negate_we_do_not(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.theirs and not d.ours]

    def summary(self) -> str:
        agree = self.agree_negated + self.agree_affirmed
        rate = agree / self.compared if self.compared else 0.0
        lines = [
            f"{self.documents} documents, {self.compared} comparable mentions",
            f"  ({self.skipped_overlapping:,} skipped as overlapping, "
            f"{self.skipped_unaligned:,} unalignable)",
            "",
            f"  agreement          {rate:.3f}  ({agree}/{self.compared})",
            f"    both negated     {self.agree_negated}",
            f"    both affirmed    {self.agree_affirmed}",
            f"  disagreements      {len(self.disagreements)}",
            f"    ours only        {len(self.we_negate_they_do_not)}",
            f"    negspaCy only    {len(self.they_negate_we_do_not)}",
        ]
        return "\n".join(lines)


def _non_overlapping(annotations: Sequence[Annotation]) -> list[Annotation]:
    """Longest-first, greedily dropping any annotation overlapping a kept one.

    Mirrors ``matcher.longest_non_overlapping`` but works on
    :class:`~umlsmatch.analyze.Annotation`, which carries the polarity this
    module is comparing.
    """
    kept: list[Annotation] = []
    taken: list[tuple[int, int]] = []
    for a in sorted(annotations, key=lambda a: (a.start - a.end, a.start)):
        if any(a.start < e and s < a.end for s, e in taken):
            continue
        kept.append(a)
        taken.append((a.start, a.end))
    return sorted(kept, key=lambda a: a.start)


def compare_documents(
    texts: Iterable[tuple[str, str]],
    *,
    db_path: str | Path | None = None,
    model: str = "en_core_web_sm",
    keep_sentences: bool = False,
    **pipeline_kwargs,
) -> DiffReport:
    """Run both implementations over ``(name, text)`` pairs and diff their polarity.

    `keep_sentences` stores the surrounding sentence on each disagreement. It
    is off by default because that text is PHI when the corpus is real notes;
    the caller decides whether to carry it.
    """
    nlp = load_negex(model)
    report = DiffReport()

    with ClinicalPipeline(db_path, **pipeline_kwargs) as ours:
        for name, text in texts:
            report.documents += 1
            annotations = ours.analyze(text)
            flat = _non_overlapping(annotations)
            report.skipped_overlapping += len(annotations) - len(flat)

            doc = nlp(text)
            spans = []
            keep: list[Annotation] = []
            for a in flat:
                span = doc.char_span(a.start, a.end, label=a.cui, alignment_mode="expand")
                if span is None:
                    report.skipped_unaligned += 1
                    continue
                spans.append(span)
                keep.append(a)

            try:
                doc.ents = spans
            except ValueError:
                # spaCy rejects the whole assignment if any pair still overlaps
                # after alignment_mode="expand" widened them onto each other.
                report.skipped_unaligned += len(spans)
                continue

            nlp.get_pipe("negex")(doc)
            for a, ent in zip(keep, doc.ents, strict=True):
                theirs = bool(ent._.negex)
                report.compared += 1
                if a.negated == theirs:
                    if a.negated:
                        report.agree_negated += 1
                    else:
                        report.agree_affirmed += 1
                    continue
                report.disagreements.append(
                    Disagreement(
                        document=name,
                        cui=a.cui,
                        start=a.start,
                        end=a.end,
                        mention=a.text,
                        ours=a.negated,
                        theirs=theirs,
                        sentence=ent.sent.text.strip() if keep_sentences else "",
                    )
                )
    return report

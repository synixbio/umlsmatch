"""Per-CUI false-positive/false-negative diff: Python matcher vs. Java cTAKES.

``umlsmatch.eval.parity`` answers "how much do we agree" (aggregate P/R/F1).
This module answers "on which concepts do we disagree" -- aggregating false
positives (Python found a CUI, Java didn't) and false negatives (Java found
a CUI, Python didn't) across a whole silver-standard corpus, ranked by
frequency, with example matched text.

For false positives, each example also carries the POS tags spaCy assigned
across the matched span.

Those tags were collected to settle whether over-generation was
POS-tagger-driven or structural. **The answer is neither.**
Perfect POS and perfect tokenization moved precision by
+0.009, and the bulk of the apparent false positives turned out to be a bug
in the silver-standard reader itself -- ``ontologyConceptArr`` holds several
whitespace-separated ids and was being parsed as one, so 2,709 multi-concept
gold mentions resolved to nothing and every correct match against them
scored as a false positive.

The POS evidence is still worth recording -- it is what a false-positive
triage starts from -- but read a large FP count here as a question about the
dictionary build, not the tagger.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from umlsmatch.dictionary.matcher import Match, RareWordMatcher
from umlsmatch.eval.records import iter_records
from umlsmatch.pipeline.tokenizer import annotate_sentences

__all__ = ["CuiEvidence", "diff_corpus", "top"]

_MAX_EXAMPLES = 3

#: Parameters per ``WHERE cui IN (...)`` batch. SQLite's compile-time
#: ``SQLITE_MAX_VARIABLE_NUMBER`` is 999 on builds predating 3.32; a corpus can
#: easily produce more distinct false-positive CUIs than that, so chunk rather
#: than betting on the host's limit.
_SQL_PARAM_CHUNK = 900


@dataclass
class CuiEvidence:
    """Aggregated evidence for one CUI's disagreements across a corpus."""

    cui: str
    preferred_text: str | None = None
    count: int = 0
    docs: set[str] = field(default_factory=set)
    example_texts: list[str] = field(default_factory=list)
    example_pos: list[tuple[str, ...]] = field(default_factory=list)

    def add(
        self,
        text: str,
        pos: tuple[str, ...],
        doc: str,
        preferred_text: str | None = None,
    ) -> None:
        self.count += 1
        self.docs.add(doc)
        if self.preferred_text is None and preferred_text:
            self.preferred_text = preferred_text
        if len(self.example_texts) < _MAX_EXAMPLES:
            self.example_texts.append(text)
            self.example_pos.append(pos)


def _python_matches_with_pos(
    text: str, matcher: RareWordMatcher
) -> list[tuple[Match, tuple[str, ...]]]:
    """Python matches for one document, each paired with its span's POS tags."""
    out: list[tuple[Match, tuple[str, ...]]] = []
    for tokens in annotate_sentences(text):
        for m in matcher.match(tokens):
            span_pos = tuple(t.pos or "" for t in tokens[m.token_start : m.token_end])
            out.append((m, span_pos))
    return out


def diff_corpus(
    jsonl_path: str | Path, db_path: str | Path
) -> tuple[dict[str, CuiEvidence], dict[str, CuiEvidence]]:
    """Aggregate false-positive and false-negative CUI evidence across a JSONL corpus.

    Returns ``(false_positives, false_negatives)``, each a ``{cui: CuiEvidence}``
    map. A CUI is a false positive for a document if the Python matcher found
    it and no mention in that document's silver record carries it; false
    negative is the reverse.
    """
    false_positives: dict[str, CuiEvidence] = {}
    false_negatives: dict[str, CuiEvidence] = {}

    with RareWordMatcher(db_path) as matcher:
        for record in iter_records(jsonl_path):
            doc = record["source_file"]
            text = record["text"]

            silver_examples: dict[str, tuple[str, str | None]] = {}
            for mention in record["mentions"]:
                for concept in mention["concepts"]:
                    cui = concept.get("cui")
                    if cui:
                        silver_examples[cui] = (mention["text"], concept.get("preferred_text"))
            silver_cui_set = set(silver_examples)

            python_matches = _python_matches_with_pos(text, matcher)
            python_cui_set = {m.cui for m, _ in python_matches}

            for m, pos in python_matches:
                if m.cui not in silver_cui_set:
                    ev = false_positives.setdefault(m.cui, CuiEvidence(m.cui))
                    ev.add(text[m.start : m.end], pos, doc)

            for cui in silver_cui_set - python_cui_set:
                gold_text, preferred_text = silver_examples[cui]
                ev = false_negatives.setdefault(cui, CuiEvidence(cui))
                ev.add(gold_text, (), doc, preferred_text=preferred_text)

    _fill_preferred_text(false_positives, db_path)
    return false_positives, false_negatives


def _fill_preferred_text(evidence: dict[str, CuiEvidence], db_path: str | Path) -> None:
    """Look up preferred_text for CUIs that have no silver mention to source it from.

    False positives have no Java mention to read a preferred_text off of (that's
    the point -- Java never annotated them), so this queries the Python
    dictionary directly for readability in reports.
    """
    if not evidence:
        return
    cuis = list(evidence)
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        for i in range(0, len(cuis), _SQL_PARAM_CHUNK):
            chunk = cuis[i : i + _SQL_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT cui, preferred_text FROM concept WHERE cui IN ({placeholders})",
                chunk,
            )
            for cui, preferred_text in rows:
                evidence[cui].preferred_text = preferred_text
    finally:
        conn.close()


def top(evidence: dict[str, CuiEvidence], n: int = 20) -> list[CuiEvidence]:
    """The `n` most frequent CUIs by disagreement count, ties broken by CUI."""
    return sorted(evidence.values(), key=lambda e: (-e.count, e.cui))[:n]

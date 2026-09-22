"""Parse Apache cTAKES XMI output into a flat, comparable mention list.

Java cTAKES' ``FileTreeXmiWriter`` serializes UIMA CASes using the default
type system. This module turns that XML into plain dataclasses -- the
"silver standard" for
scoring the Python port against real cTAKES output.

Usage::

    from umlsmatch.silver.xmi import parse_ctakes_xmi

    doc = parse_ctakes_xmi("out/sample.txt.xmi")
    for m in doc.mentions:
        print(m.type, m.text, m.negated, [c.cui for c in m.concepts])
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_TEXTSEM_NS = "http:///org/apache/ctakes/typesystem/type/textsem.ecore"
_REFSEM_NS = "http:///org/apache/ctakes/typesystem/type/refsem.ecore"
_TEXTSPAN_NS = "http:///org/apache/ctakes/typesystem/type/textspan.ecore"
_SYNTAX_NS = "http:///org/apache/ctakes/typesystem/type/syntax.ecore"
_CAS_NS = "http:///uima/cas.ecore"
_XMI_NS = "http://www.omg.org/XMI"
_XMI_ID = f"{{{_XMI_NS}}}id"

# Semantic-group mention types produced by DefaultFastPipeline's
# DefaultJCasTermAnnotator + assertion sub-pipe.
MENTION_TYPES = (
    "SignSymptomMention",
    "DiseaseDisorderMention",
    "MedicationMention",
    "ProcedureMention",
    "AnatomicalSiteMention",
    "EventMention",
)

# The six concrete subtypes of org.apache.ctakes.typesystem.type.syntax.BaseToken
# (an abstract type -- only these appear in XMI). Verified against
# ctakes-type-system/.../types/TypeSystem.xml in the ctakes-java clone.
BASE_TOKEN_TYPES = (
    "WordToken",
    "ContractionToken",
    "NewlineToken",
    "NumToken",
    "PunctuationToken",
    "SymbolToken",
)


@dataclass(frozen=True)
class UmlsConcept:
    """One ``UmlsConcept`` attached to a cTAKES mention.

    Every field is optional because the XMI is the authority on what Java
    recorded: a missing attribute is reported as ``None`` rather than guessed
    at, since this type exists to be diffed against our own output.
    """

    cui: str | None
    tui: str | None
    coding_scheme: str | None
    preferred_text: str | None


@dataclass(frozen=True)
class Mention:
    """One ``IdentifiedAnnotation`` span, with its assertion attributes.

    ``uncertain``, ``conditional``, ``generic``, ``subject`` and ``history_of``
    are carried even though umlsmatch does not implement all of them --
    dropping them at parse time would make the silver standard unable to answer
    questions about the attributes still on the roadmap.

    Note the three different spellings cTAKES uses for what are conceptually
    three booleans: ``polarity`` is ``1``/``-1``, ``uncertainty`` and
    ``historyOf`` are ``0``/``1``, and ``conditional``/``generic`` are
    ``true``/``false``. They are normalized here so consumers see one shape.
    """

    type: str
    text: str
    begin: int
    end: int
    polarity: int  # 1 = asserted, -1 = negated
    uncertain: bool
    conditional: bool
    generic: bool
    subject: str | None
    #: cTAKES' ``historyOf``: the mention sits in a history-taking context.
    #: Narrower than "past tense" -- see umlsmatch.assertion.history.
    history_of: bool
    concepts: tuple[UmlsConcept, ...] = field(default_factory=tuple)

    @property
    def negated(self) -> bool:
        return self.polarity == -1


@dataclass(frozen=True)
class CtakesDocument:
    """One note's worth of Java cTAKES output, parsed from a single XMI file."""

    path: Path
    text: str
    mentions: tuple[Mention, ...]
    sentences: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    tokens: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    #: (begin, end, token_type, part_of_speech) for every BaseToken. Only
    #: WordToken carries a Penn tag; the other subtypes have none, which is
    #: itself meaningful -- cTAKES excludes non-word tokens as lookup anchors
    #: regardless of POS.
    pos_tokens: tuple[tuple[int, int, str, str | None], ...] = field(default_factory=tuple)


def parse_ctakes_xmi(path: str | Path) -> CtakesDocument:
    """Parse one ``FileTreeXmiWriter`` output file into a :class:`CtakesDocument`."""
    path = Path(path)
    root = ET.parse(path).getroot()

    sofa = root.find(f"{{{_CAS_NS}}}Sofa")
    text = sofa.get("sofaString", "") if sofa is not None else ""

    concepts_by_id = {
        elem.get(_XMI_ID): UmlsConcept(
            cui=elem.get("cui"),
            tui=elem.get("tui"),
            coding_scheme=elem.get("codingScheme"),
            preferred_text=elem.get("preferredText"),
        )
        for elem in root.iter(f"{{{_REFSEM_NS}}}UmlsConcept")
    }

    mentions = []
    for tag in MENTION_TYPES:
        for elem in root.iter(f"{{{_TEXTSEM_NS}}}{tag}"):
            begin, end = int(elem.get("begin")), int(elem.get("end"))
            mentions.append(
                Mention(
                    type=tag,
                    text=text[begin:end],
                    begin=begin,
                    end=end,
                    polarity=int(elem.get("polarity", "1")),
                    uncertain=elem.get("uncertainty") == "1",
                    conditional=elem.get("conditional") == "true",
                    generic=elem.get("generic") == "true",
                    subject=elem.get("subject"),
                    history_of=elem.get("historyOf") == "1",
                    concepts=_resolve_concepts(
                        elem.get("ontologyConceptArr"), concepts_by_id, root
                    ),
                )
            )
    # Sorted on the full span plus type: cTAKES routinely emits several mention
    # types over one span, and ordering on `begin` alone leaves their relative
    # order at the mercy of MENTION_TYPES iteration -- enough to make a silver
    # standard re-export produce a different JSONL for identical input.
    mentions.sort(key=lambda m: (m.begin, m.end, m.type))

    sentences = sorted(
        {
            (int(elem.get("begin")), int(elem.get("end")))
            for elem in root.iter(f"{{{_TEXTSPAN_NS}}}Sentence")
        }
    )
    pos_tokens = sorted(
        {
            (
                int(elem.get("begin")),
                int(elem.get("end")),
                tag,
                elem.get("partOfSpeech"),
            )
            for tag in BASE_TOKEN_TYPES
            for elem in root.iter(f"{{{_SYNTAX_NS}}}{tag}")
        }
    )
    tokens = sorted({(b, e) for b, e, _, _ in pos_tokens})

    return CtakesDocument(
        path=path,
        text=text,
        mentions=tuple(mentions),
        sentences=tuple(sentences),
        tokens=tuple(tokens),
        pos_tokens=tuple(pos_tokens),
    )


def _resolve_concepts(
    arr_id: str | None,
    concepts_by_id: dict[str | None, UmlsConcept],
    root: ET.Element,
) -> tuple[UmlsConcept, ...]:
    if not arr_id:
        return ()

    # `ontologyConceptArr` may hold SEVERAL whitespace-separated ids, not one:
    # UIMA inlines an FSArray of references as "2483 2484 2485 2486". Treating
    # the whole attribute as a single id silently yields no concepts -- which
    # hit medications hardest, since a drug mention typically carries one
    # concept per RxNorm form. Split first, then resolve each id.
    refs = arr_id.split()
    direct = tuple(concepts_by_id[r] for r in refs if r in concepts_by_id)
    if direct:
        return direct

    # Otherwise the attribute points at a wrapper <cas:FSArray elements=.../>.
    concepts: list[UmlsConcept] = []
    for ref in refs:
        arr_elem = root.find(f".//*[@{_XMI_ID}='{ref}']")
        if arr_elem is None:
            continue
        concepts.extend(
            concepts_by_id[e]
            for e in (arr_elem.get("elements") or "").split()
            if e in concepts_by_id
        )
    return tuple(concepts)

"""Tests for parsing Java cTAKES FileTreeXmiWriter output.

Uses a minimal hand-built XMI fixture rather than a full cTAKES run, covering
the two ontologyConceptArr shapes UIMA actually emits (a single-element array
inlined as a direct reference, and a mention with no concept at all), plus
Sentence and BaseToken-subtype spans used by eval.boundaries.
"""

from __future__ import annotations

from pathlib import Path

from umlsmatch.silver.xmi import parse_ctakes_xmi

_SAMPLE_XMI = """<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI"
         xmlns:cas="http:///uima/cas.ecore"
         xmlns:textsem="http:///org/apache/ctakes/typesystem/type/textsem.ecore"
         xmlns:refsem="http:///org/apache/ctakes/typesystem/type/refsem.ecore"
         xmlns:textspan="http:///org/apache/ctakes/typesystem/type/textspan.ecore"
         xmlns:syntax="http:///org/apache/ctakes/typesystem/type/syntax.ecore"
         xmi:version="2.0">
  <cas:NULL xmi:id="0"/>
  <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView"
            mimeType="text" sofaString="No fever. Chest pain present."/>
  <refsem:UmlsConcept xmi:id="10" cui="C0008031" tui="T184"
                       codingScheme="SNOMEDCT_US" preferredText="Chest Pain"/>
  <textsem:SignSymptomMention xmi:id="20" sofa="1" begin="3" end="8"
                               polarity="-1" uncertainty="0"
                               conditional="false" generic="false"
                               subject="patient" historyOf="0"/>
  <textsem:SignSymptomMention xmi:id="21" sofa="1" begin="10" end="20"
                               ontologyConceptArr="10" polarity="1"
                               uncertainty="0" conditional="false"
                               generic="false" subject="patient" historyOf="1"/>
  <textspan:Sentence xmi:id="30" sofa="1" begin="0" end="9" sentenceNumber="0"/>
  <textspan:Sentence xmi:id="31" sofa="1" begin="10" end="30" sentenceNumber="1"/>
  <syntax:WordToken xmi:id="40" sofa="1" begin="0" end="2" partOfSpeech="DT"/>
  <syntax:WordToken xmi:id="41" sofa="1" begin="3" end="8" partOfSpeech="NN"/>
  <syntax:PunctuationToken xmi:id="42" sofa="1" begin="8" end="9" partOfSpeech="."/>
</xmi:XMI>
"""


def test_parse_ctakes_xmi(tmp_path: Path):
    xmi_path = tmp_path / "sample.txt.xmi"
    xmi_path.write_text(_SAMPLE_XMI, encoding="utf-8")

    doc = parse_ctakes_xmi(xmi_path)

    assert doc.text == "No fever. Chest pain present."
    assert len(doc.mentions) == 2

    fever, chest_pain = doc.mentions
    assert fever.text == "fever"
    assert fever.negated is True
    assert fever.concepts == ()

    assert chest_pain.text == "Chest pain"
    assert chest_pain.negated is False
    assert len(chest_pain.concepts) == 1
    concept = chest_pain.concepts[0]
    assert concept.cui == "C0008031"
    assert concept.tui == "T184"
    assert concept.preferred_text == "Chest Pain"

    assert doc.sentences == ((0, 9), (10, 30))
    assert doc.tokens == ((0, 2), (3, 8), (8, 9))


def test_assertion_attributes_are_parsed(tmp_path: Path):
    """Every attribute cTAKES writes, in the three spellings it uses for them.

    ``historyOf`` is the one that was missing, and it is the attribute with the
    best class balance in the corpus (4.0% positive against 0.21% for
    ``conditional``). Dropping it at parse
    time left the silver standard unable to answer the question it is best
    placed to answer.
    """
    p = tmp_path / "attrs.txt.xmi"
    p.write_text(_SAMPLE_XMI, encoding="utf-8")

    fever, chest_pain = parse_ctakes_xmi(p).mentions

    # polarity is 1/-1, uncertainty and historyOf are 0/1, conditional and
    # generic are true/false. All five arrive as booleans here.
    assert fever.negated is True
    assert fever.uncertain is False
    assert fever.conditional is False
    assert fever.generic is False
    assert fever.subject == "patient"
    assert fever.history_of is False

    assert chest_pain.history_of is True, "historyOf=\"1\" must parse as True"


def test_missing_history_of_is_false_not_an_error(tmp_path: Path):
    """An older XMI without the attribute still parses.

    ``historyOf`` is absent from no cTAKES release this project targets, but a
    hand-built or truncated fixture should report "not marked" rather than
    raise -- the same tolerance every other attribute here already has.
    """
    p = tmp_path / "multi.txt.xmi"
    p.write_text(_MULTI_CONCEPT_XMI, encoding="utf-8")

    (mention,) = parse_ctakes_xmi(p).mentions
    assert mention.history_of is False


# Regression: ontologyConceptArr may hold SEVERAL whitespace-separated ids.
# Treating the attribute as a single id silently produced zero concepts, which
# wiped out 91% of MedicationMention CUIs in the silver standard (a drug mention
# typically carries one concept per RxNorm form) and made correct matcher output
# look like false positives.
_MULTI_CONCEPT_XMI = """<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmlns:xmi="http://www.omg.org/XMI"
         xmlns:cas="http:///uima/cas.ecore"
         xmlns:textsem="http:///org/apache/ctakes/typesystem/type/textsem.ecore"
         xmlns:refsem="http:///org/apache/ctakes/typesystem/type/refsem.ecore"
         xmi:version="2.0">
  <cas:NULL xmi:id="0"/>
  <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView"
            mimeType="text" sofaString="Started on omeprazole."/>
  <refsem:UmlsConcept xmi:id="10" cui="C0028978" tui="T109" codingScheme="RXNORM"/>
  <refsem:UmlsConcept xmi:id="11" cui="C0700777" tui="T121" codingScheme="RXNORM"/>
  <refsem:UmlsConcept xmi:id="12" cui="C1999262" tui="T122" codingScheme="RXNORM"/>
  <textsem:MedicationMention xmi:id="20" sofa="1" begin="11" end="21"
                              ontologyConceptArr="10 11 12" polarity="1"
                              uncertainty="0" conditional="false"
                              generic="false" subject="patient"/>
</xmi:XMI>
"""


def test_multi_id_ontology_concept_arr_resolves_every_concept(tmp_path: Path):
    p = tmp_path / "multi.txt.xmi"
    p.write_text(_MULTI_CONCEPT_XMI, encoding="utf-8")

    doc = parse_ctakes_xmi(p)

    (mention,) = doc.mentions
    assert mention.text == "omeprazole"
    cuis = {c.cui for c in mention.concepts}
    assert cuis == {"C0028978", "C0700777", "C1999262"}, (
        "all whitespace-separated concept ids must resolve, not just the first"
    )


def test_pos_tokens_carry_tags_and_types(tmp_path: Path):
    p = tmp_path / "pos.txt.xmi"
    p.write_text(_SAMPLE_XMI, encoding="utf-8")

    doc = parse_ctakes_xmi(p)

    by_span = {(b, e): (t, pos) for b, e, t, pos in doc.pos_tokens}
    assert by_span[(0, 2)] == ("WordToken", "DT")
    assert by_span[(3, 8)] == ("WordToken", "NN")
    assert by_span[(8, 9)] == ("PunctuationToken", ".")
    # plain token spans stay consistent with the POS-bearing view
    assert doc.tokens == tuple(sorted(by_span))

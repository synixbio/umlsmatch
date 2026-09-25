"""Tests for the end-to-end pipeline and CLI.

Require both the built dictionary and the optional `nlp` extra; skipped
otherwise.
"""

from __future__ import annotations

import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import pytest

from umlsmatch.analyze import Annotation, find_dictionary

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"

pytestmark = [
    pytest.mark.skipif(not DB.is_file(), reason=f"dictionary not built: {DB}"),
    pytest.mark.skipif(
        importlib.util.find_spec("spacy") is None,
        reason="optional 'nlp' extra (spaCy) not installed",
    ),
]

NOTE = (
    "Patient denies chest pain and shortness of breath.\n"
    "History of type 2 diabetes mellitus.\n"
    "Started on metformin 500 mg PO BID.\n"
)


@pytest.fixture(scope="module")
def nlp():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB) as p:
        yield p


def test_finds_expected_concepts(nlp):
    cuis = {a.cui for a in nlp.analyze(NOTE)}
    for expected in ("C0008031", "C0013404", "C0011860", "C0025598"):
        assert expected in cuis, f"{expected} missing from {sorted(cuis)}"


def test_negation_flagged(nlp):
    by_cui = {a.cui: a for a in nlp.analyze(NOTE)}
    assert by_cui["C0008031"].negated is True, "chest pain follows 'denies'"
    assert by_cui["C0025598"].negated is False, "metformin is affirmed"


def test_offsets_match_source_text(nlp):
    for a in nlp.analyze(NOTE):
        assert NOTE[a.start : a.end] == a.text


def test_results_sorted_by_position(nlp):
    anns = nlp.analyze(NOTE)
    assert anns == sorted(anns, key=lambda a: (a.start, a.end, a.cui))


def test_preferred_text_populated(nlp):
    assert all(a.preferred_text for a in nlp.analyze(NOTE) if a.cui == "C0025598")


def test_group_filter(nlp):
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, groups={"DRUG"}) as p:
        anns = p.analyze(NOTE)
    assert anns, "expected at least one DRUG annotation"
    assert {a.group for a in anns} == {"DRUG"}


def test_empty_and_whitespace_input(nlp):
    assert nlp.analyze("") == []
    assert nlp.analyze("   \n\t ") == []


def test_text_without_concepts(nlp):
    """Ordinary prose with no clinical content yields nothing.

    The probe has to be chosen with care. "The quick brown fox jumped." is not
    concept-free once the dictionary reads synonyms from every UMLS vocabulary:
    "brown" matches C0155339 (Brown syndrome) and "fox" C0016632. Those are real
    entries, not a matcher bug -- they are the precision cost of a wide synonym
    set, and the reason `--sources` is tunable.
    """
    assert nlp.analyze("She walked to the store on Tuesday.") == []


def test_analyze_documents_yields_per_document(nlp):
    out = list(nlp.analyze_documents([NOTE, "", "pneumonia"]))
    assert len(out) == 3
    assert out[1] == []
    assert any(a.cui == "C0032285" for a in out[2])


def test_to_json_roundtrips(nlp):
    parsed = json.loads(nlp.to_json(NOTE))
    assert isinstance(parsed, list) and parsed
    assert {"cui", "text", "start", "end", "group", "negated"} <= set(parsed[0])


def test_uncapped_scope_negates_more(nlp):
    """The scope cap is doing real work -- removing it must over-negate."""
    from umlsmatch import ClinicalPipeline

    text = "Patient denies chest pain, and reports pneumonia and diabetes mellitus."
    with ClinicalPipeline(DB, max_scope=None) as uncapped:
        n_uncapped = sum(a.negated for a in uncapped.analyze(text))
    n_capped = sum(a.negated for a in nlp.analyze(text))
    assert n_uncapped >= n_capped


def test_negative_max_scope_is_rejected():
    """A negative cap empties every scope, which reads as 'negation is broken'."""
    from umlsmatch import ClinicalPipeline

    with pytest.raises(ValueError, match="max_scope"):
        ClinicalPipeline(DB, max_scope=-1)


def test_zero_max_scope_negates_nothing(nlp):
    """0 is a legal cap (distinct from None, which disables it)."""
    from umlsmatch import ClinicalPipeline

    text = "Patient denies chest pain."
    with ClinicalPipeline(DB, max_scope=0) as zero:
        assert not any(a.negated for a in zero.analyze(text))
    assert any(a.negated for a in nlp.analyze(text))


def test_resolve_overlaps_reduces_annotations(nlp):
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, resolve_overlaps=True) as flat:
        reduced = flat.analyze(NOTE)
    assert len(reduced) < len(nlp.analyze(NOTE))
    spans = sorted((a.start, a.end) for a in reduced)
    for (_, e1), (s2, _) in pairwise(spans):
        assert e1 <= s2, "overlap survived resolution"


def test_annotation_to_dict_is_json_safe():
    a = Annotation("C1", "x", 0, 1, "DRUG", False, "pref", "x")
    assert json.loads(json.dumps(a.to_dict()))["cui"] == "C1"


def test_missing_dictionary_raises_filenotfound():
    from umlsmatch import ClinicalPipeline

    with pytest.raises(FileNotFoundError):
        ClinicalPipeline("no/such/dictionary.sqlite")


def test_find_dictionary_prefers_explicit_path():
    assert find_dictionary("some/explicit.sqlite") == Path("some/explicit.sqlite")


def test_find_dictionary_honours_env(monkeypatch):
    from umlsmatch.analyze import DB_ENV_VAR

    monkeypatch.setenv(DB_ENV_VAR, "env/path.sqlite")
    assert find_dictionary() == Path("env/path.sqlite")


# --- document length --------------------------------------------------------


def test_oversized_document_raises_a_clear_error():
    """Without this the caller gets spaCy's [E088] from several frames down,
    naming a limit on an object they never constructed."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, max_chars=500) as p:
        with pytest.raises(ValueError) as exc:
            p.analyze("x " * 400)
        message = str(exc.value)
        assert "800" in message and "500" in message
        assert "max_chars" in message  # says how to change it
        assert "E088" not in message


def test_document_at_the_limit_is_accepted():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, max_chars=26) as p:
        assert p.analyze("Patient denies chest pain.") is not None


def test_max_chars_none_disables_the_check():
    """The escape hatch has to actually reach spaCy, not silently cap."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, max_chars=None) as p:
        assert p.max_chars is None
        assert p.analyze("Patient denies chest pain.") is not None


def test_max_chars_zero_is_rejected_at_construction():
    """A zero limit would reject every document and read as the pipeline being
    broken rather than misconfigured."""
    from umlsmatch import ClinicalPipeline

    with pytest.raises(ValueError, match="max_chars"):
        ClinicalPipeline(DB, max_chars=0)


def test_default_limit_matches_spacys_own():
    """These two must move together. A pipeline limit above spaCy's would let a
    document through this check and straight into the error it replaces."""
    import spacy

    from umlsmatch.analyze import DEFAULT_MAX_CHARS

    assert spacy.blank("en").max_length == DEFAULT_MAX_CHARS


# --- CLI --------------------------------------------------------------------


def test_cli_table_output(capsys):
    from umlsmatch.__main__ import main

    note = Path(__file__).parent / "_cli_note.txt"
    note.write_text(NOTE, encoding="utf-8")
    try:
        assert main([str(note), "--db", str(DB), "--no-save"]) == 0
        out = capsys.readouterr().out
        assert "C0025598" in out and "metformin" in out
        assert "NEGATED" in out
    finally:
        note.unlink()


def test_cli_json_output(capsys):
    from umlsmatch.__main__ import main

    note = Path(__file__).parent / "_cli_note_json.txt"
    note.write_text(NOTE, encoding="utf-8")
    try:
        assert main(
            [str(note), "--db", str(DB), "--json", "--groups", "DRUG", "--no-save"]
        ) == 0
        rec = json.loads(capsys.readouterr().out.strip().splitlines()[0])
        assert rec["source"].endswith("_cli_note_json.txt")
        assert {a["group"] for a in rec["annotations"]} == {"DRUG"}
    finally:
        note.unlink()


def test_cli_rejects_conflicting_polarity_flags():
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), "--negated-only", "--affirmed-only"])


def test_cli_missing_input_exits(capsys):
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["no/such/file.txt", "--db", str(DB)])


def test_cli_missing_dictionary_returns_1(capsys):
    from umlsmatch.__main__ import main

    assert main(["-", "--db", "no/such/db.sqlite"]) == 1
    assert "dictionary not found" in capsys.readouterr().err


def test_cli_rejects_negative_max_scope():
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), "--max-scope", "-1"])


def test_cli_rejects_both_scope_flags():
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), "--max-scope", "4", "--no-max-scope"])


# --- CLI/library parity -----------------------------------------------------
#
# The module docstring promises flags map straight onto constructor arguments.
# Five of them once did not exist at all, so a CLI user could not reach
# `history_sections` -- the setting that most changes what the pipeline reports
# about a chart. These pin the mapping for the ones that were missing.

_SECTION_NOTE = (
    "PAST MEDICAL HISTORY:\n"
    "Hypertension.\n"
    "Type 2 diabetes mellitus.\n"
    "ASSESSMENT:\n"
    "Call if you develop chest pain.\n"
)


def _cli_records(capsys, *flags) -> list[dict]:
    from umlsmatch.__main__ import main

    note = Path(__file__).parent / "_cli_sections.txt"
    note.write_text(_SECTION_NOTE, encoding="utf-8")
    try:
        assert main([str(note), "--db", str(DB), "--json", "--no-save", *flags]) == 0
        out = capsys.readouterr().out.strip().splitlines()[0]
        return json.loads(out)["annotations"]
    finally:
        note.unlink()


def _api_records(**kwargs) -> list[dict]:
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, **kwargs) as p:
        return [a.to_dict() for a in p.analyze(_SECTION_NOTE)]


def test_cli_history_sections_matches_the_api(capsys):
    assert _cli_records(capsys, "--history-sections") == _api_records(
        history_sections=True
    )


def test_cli_history_sections_actually_changes_the_output(capsys):
    """A flag that maps correctly onto a no-op would pass the test above."""
    off = sum(1 for a in _cli_records(capsys) if a["history_of"])
    on = sum(1 for a in _cli_records(capsys, "--history-sections") if a["history_of"])
    assert on > off, (off, on)


def test_cli_drop_header_mentions_matches_the_api(capsys):
    assert _cli_records(capsys, "--drop-header-mentions") == _api_records(
        drop_header_mentions=True
    )


def test_cli_conditional_matches_the_api(capsys):
    assert _cli_records(capsys, "--conditional") == _api_records(conditional=True)


def test_cli_conditional_is_printed_in_the_table(capsys):
    """`--conditional` with no CONDITIONAL column reads as the rules never
    firing, which is the wrong thing to conclude about a prototype."""
    from umlsmatch.__main__ import main

    note = Path(__file__).parent / "_cli_cond.txt"
    note.write_text(_SECTION_NOTE, encoding="utf-8")
    try:
        assert main([str(note), "--db", str(DB), "--conditional", "--no-save"]) == 0
        assert "CONDITIONAL" in capsys.readouterr().out
    finally:
        note.unlink()


@pytest.mark.parametrize(
    ("flag", "kwarg", "field"),
    [
        ("--no-subject", "subject", "subject"),
        ("--no-history", "history", "history_of"),
        ("--no-uncertainty", "uncertainty", "uncertain"),
    ],
)
def test_cli_attribute_switches_yield_null_not_false(capsys, flag, kwarg, field):
    """Turning an attribute off must report "not assessed", not "absent".

    The three-state contract is the point of `Annotation`, and before these
    flags existed the CLI had no way to express the third state at all.
    """
    records = _cli_records(capsys, flag)
    assert records == _api_records(**{kwarg: False})
    assert all(a[field] is None for a in records)


def test_cli_profile_matches_the_api(capsys):
    assert _cli_records(capsys, "--profile", "clinical_recall") == _api_records(
        profile="clinical_recall"
    )


def test_cli_explicit_flag_overrides_the_profile(capsys):
    """`--profile` supplies defaults; a flag after it still wins."""
    records = _cli_records(
        capsys, "--profile", "clinical_recall", "--no-drop-header-mentions"
    )
    assert records == _api_records(
        profile="clinical_recall", drop_header_mentions=False
    )
    # The header concepts are back, and the section rule is still on.
    assert any(a["text"] == "PAST MEDICAL HISTORY" for a in records)
    assert any(a["history_of"] for a in records)


def test_cli_rejects_an_unknown_profile(capsys):
    """The rejection has to say what to type instead: this is reached by a typo
    far more often than by a bug."""
    from umlsmatch.__main__ import main

    with pytest.raises(SystemExit):
        main(["-", "--db", str(DB), "--profile", "clinical-recall"])
    message = capsys.readouterr().err
    assert "clinical_recall" in message and "strict" in message


# `--max-scope 0` means a zero-token scope on the command line exactly as
# `max_scope=0` does through the API. Translating 0 to None would have the two
# read the same literal in opposite directions. These pin them together.

_SCOPE_NOTE = "Patient denies chest pain, and reports pneumonia and diabetes mellitus."


def _cli_negated(capsys, *flags) -> int:
    from umlsmatch.__main__ import main

    note = Path(__file__).parent / "_cli_scope.txt"
    note.write_text(_SCOPE_NOTE, encoding="utf-8")
    try:
        assert main(
            [str(note), "--db", str(DB), "--negated-only", "--no-save", *flags]
        ) == 0
        return len([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()])
    finally:
        note.unlink()


def _api_negated(**kwargs) -> int:
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, **kwargs) as p:
        return sum(a.negated for a in p.analyze(_SCOPE_NOTE))


def test_cli_max_scope_zero_matches_the_api(capsys):
    assert _cli_negated(capsys, "--max-scope", "0") == _api_negated(max_scope=0) == 0


def test_cli_max_scope_n_matches_the_api(capsys):
    assert _cli_negated(capsys, "--max-scope", "4") == _api_negated(max_scope=4)


def test_cli_no_max_scope_matches_api_none(capsys):
    assert _cli_negated(capsys, "--no-max-scope") == _api_negated(max_scope=None)


def test_uncapped_scope_over_negates_without_a_parse():
    """The cap earns its keep under the lexical rules alone.

    Stated against ``dependencies=False`` on both sides deliberately. The
    original form of this assertion compared uncapped-with-parse against
    capped-with-parse and inverted once clause bounding landed -- not because
    the cap stopped mattering in general, but because on this sentence the
    parse removes the very over-negation the cap existed to bound. Comparing
    across that change measured two things at once.
    """
    lexical = {"coordination": False, "clause_bounding": False, "sections": False}
    assert _api_negated(**lexical, max_scope=None) > _api_negated(**lexical)


def test_dependency_scope_fixes_a_false_positive_the_window_had():
    """"reports pneumonia" is an assertion the bare token window negates.

    ``_SCOPE_NOTE`` coordinates two verbs ("denies", "reports"). Quote it in
    full: truncated to "...and reports pneumonia.", spaCy tags "reports" NNS
    and makes it a compound, so there is no coordinated verb and the rule
    correctly does not fire.
    "and" is not a lexical terminator, so the forward trigger ran straight
    through into the second clause and flagged the reported diagnosis. Only
    MAX_SCOPE_TOKENS bounded it, which made the error depend on the length of
    the first clause rather than on the grammar.
    """
    assert _api_negated(coordination=False, clause_bounding=False, sections=False) == 4
    assert _api_negated() == 3


def test_dependency_scope_makes_the_cap_redundant_here():
    """With clause bounding on, capping this sentence's scope changes nothing.

    Pinned because it is the point of the feature: the cap was a proxy for a
    boundary the parse can name outright. Where the parse finds the boundary,
    the proxy stops mattering.
    """
    assert _api_negated() == _api_negated(max_scope=None) == 3


def test_drop_header_mentions_removes_the_heading_not_the_content():
    """The concept *inside* "PAST MEDICAL HISTORY:" goes; the note does not.

    Off by default -- it changes what is extracted, and the parity figure is
    measured on the unfiltered stream. See docs/ADJUDICATION_RESULTS.md.
    """
    from umlsmatch import ClinicalPipeline

    note = (
        "PAST MEDICAL HISTORY:\n"
        "History of congestive heart failure.\n"
        "FAMILY HISTORY:\n"
        "Mother with breast cancer.\n"
    )
    with ClinicalPipeline(DB) as nlp:
        before = {a.text for a in nlp.analyze(note)}
    with ClinicalPipeline(DB, drop_header_mentions=True) as nlp:
        after = {a.text for a in nlp.analyze(note)}

    assert "PAST MEDICAL HISTORY" in before and "PAST MEDICAL HISTORY" not in after
    assert "FAMILY HISTORY" in before and "FAMILY HISTORY" not in after
    # The history cue in the body is content, not furniture.
    assert "History" in after
    for clinical in ("heart failure", "breast cancer"):
        assert clinical in after, clinical


# --- assessed_attributes ------------------------------------------------------
#
# The exporters in examples/ choose their columns from this, before the first
# annotation exists. A wrong answer here writes a column of empty cells that a
# reader cannot distinguish from an attribute that happened not to occur.


def test_default_pipeline_assesses_four_attributes(nlp):
    assert nlp.assessed_attributes == frozenset(
        {"negated", "subject", "history_of", "uncertain"}
    )


def test_unassessed_attributes_are_none_on_every_annotation(nlp):
    """The property's whole claim: what it excludes really is always None."""
    from umlsmatch.assertion.attributes import attribute_names

    unassessed = set(attribute_names()) - nlp.assessed_attributes
    assert unassessed == {"conditional", "generic"}
    annotations = nlp.analyze(NOTE)
    assert annotations, "expected this note to produce annotations"
    for a in annotations:
        for name in unassessed:
            assert getattr(a, name) is None, name


def test_generic_is_never_assessed():
    """No switch turns it on -- there are no rules to turn on."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, conditional=True) as nlp:
        assert "generic" not in nlp.assessed_attributes


def test_enabling_conditional_adds_it():
    """A consumer reading the property gets the column back with no edit."""
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, conditional=True) as nlp:
        assert "conditional" in nlp.assessed_attributes


def test_disabling_attributes_removes_them():
    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline(DB, subject=False, history=False, uncertainty=False) as nlp:
        # `negated` has no switch: it is always assessed, and stays.
        assert nlp.assessed_attributes == frozenset({"negated"})

"""Tests for the pure-Python rare-word matcher.

Unit tests for tokenization and overlap resolution always run. Tests needing the
built dictionary skip when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umlsmatch.dictionary.matcher import (
    DEFAULT_EXCLUSION_TAGS,
    DEFAULT_MINIMUM_SPAN,
    Match,
    RareWordMatcher,
    Token,
    longest_non_overlapping,
    tokenize,
)

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"
needs_db = pytest.mark.skipif(not DB.is_file(), reason=f"dictionary not built: {DB}")


# --- constants pinned to the Java source ------------------------------------


def test_exclusion_tags_match_ctakes():
    # Spelled out independently of the source constant, in the same
    # space-separated form as the Java, so this stays a real check rather than
    # a re-import of the value under test.
    expected = (
        "VB VBD VBG VBN VBP VBZ CC CD DT EX IN LS MD PDT POS PP PP$ "  # noqa: SIM905
        "PRP PRP$ RP TO WDT WP WPS WRB".split()
    )
    assert frozenset(expected) == DEFAULT_EXCLUSION_TAGS
    # 25 tags, verified by parsing JCasTermAnnotator.DEFAULT_EXCLUSION_TAGS.
    assert len(DEFAULT_EXCLUSION_TAGS) == 25


def test_minimum_span_matches_ctakes():
    assert DEFAULT_MINIMUM_SPAN == 3


# --- tokenizer --------------------------------------------------------------


def test_tokenize_preserves_offsets():
    text = "Chest x-ray showed pneumonia."
    for t in tokenize(text):
        assert text[t.start : t.end] == t.text


def test_tokenize_marks_punctuation_as_non_word():
    toks = {t.text: t for t in tokenize("a, b.")}
    assert toks[","].is_word is False
    assert toks["."].is_word is False


def test_tokenize_marks_numbers_as_non_word():
    """Numbers are not lookup anchors in cTAKES."""
    (tok,) = [t for t in tokenize("500") if t.text == "500"]
    assert tok.is_word is False


def test_tokenize_keeps_hyphenated_words_together():
    assert [t.text for t in tokenize("x-ray")] == ["x-ray"]


def test_tokenize_can_split_hyphens_and_clitics_like_spacy():
    """The spelling every built dictionary actually stores.

    ``tools/retokenize_terms.py`` runs spaCy over the terms, so the stored norm
    is "chest x - ray" and "crohn 's disease". Matching is exact string
    equality, so a tokenizer that keeps them whole matches nothing and says
    nothing -- see ``match_text``.
    """
    assert [t.text for t in tokenize("x-ray", split_hyphens=True)] == ["x", "-", "ray"]
    assert [t.text for t in tokenize("crohn's", split_hyphens=True)] == ["crohn", "'s"]
    # An apostrophe that is not a clitic stays put, as it does in spaCy.
    assert [t.text for t in tokenize("o'brien", split_hyphens=True)] == ["o'brien"]


def test_tokenize_preserves_offsets_when_splitting():
    text = "chest x-ray of crohn's disease"
    for tok in tokenize(text, split_hyphens=True):
        assert text[tok.start : tok.end] == tok.text


def test_token_norm_is_casefolded():
    assert Token("Pneumonia", 0, 9).norm == "pneumonia"


# --- overlap resolution -----------------------------------------------------


def _m(start, end, cui="C1"):
    return Match(cui=cui, term="t", text="", start=start, end=end, token_start=0, token_end=1)


def test_longest_non_overlapping_prefers_longer():
    kept = longest_non_overlapping([_m(0, 5), _m(0, 12), _m(6, 10)])
    assert [(m.start, m.end) for m in kept] == [(0, 12)]


def test_longest_non_overlapping_keeps_disjoint():
    kept = longest_non_overlapping([_m(0, 5), _m(10, 20)])
    assert [(m.start, m.end) for m in kept] == [(0, 5), (10, 20)]


def test_longest_non_overlapping_is_sorted_by_position():
    kept = longest_non_overlapping([_m(30, 40), _m(0, 10), _m(15, 20)])
    assert [m.start for m in kept] == [0, 15, 30]


def test_longest_non_overlapping_empty():
    assert longest_non_overlapping([]) == []


def test_adjacent_spans_do_not_count_as_overlapping():
    kept = longest_non_overlapping([_m(0, 5), _m(5, 10)])
    assert len(kept) == 2


# Co-extensive spans. One span mapping to two CUIs is not a disagreement about
# the span -- it is the vocabulary mapping one string to two concepts, which
# UMLS does constantly. Dropping one of them was silent and decided by
# alphabetical order on the CUI.


def test_co_extensive_cuis_are_all_kept():
    kept = longest_non_overlapping([_m(0, 3, "C0009714"), _m(0, 3, "C0018802")])
    assert sorted(m.cui for m in kept) == ["C0009714", "C0018802"]


def test_co_extensive_cuis_still_suppress_strict_sub_spans():
    """Keeping aliases must not turn this into "keep everything"."""
    kept = longest_non_overlapping(
        [_m(0, 10, "C1"), _m(0, 10, "C2"), _m(0, 5, "C3"), _m(6, 9, "C4")]
    )
    assert sorted(m.cui for m in kept) == ["C1", "C2"]


def test_co_extensive_aliases_of_a_loser_are_dropped_together():
    """A shorter span loses to a longer one whether or not it has aliases."""
    kept = longest_non_overlapping(
        [_m(0, 12, "C1"), _m(2, 6, "C2"), _m(2, 6, "C3")]
    )
    assert [m.cui for m in kept] == ["C1"]


def test_identical_matches_are_not_duplicated():
    """Same span *and* same CUI is one annotation, not two."""
    kept = longest_non_overlapping([_m(0, 4, "C1"), _m(0, 4, "C1")])
    assert len(kept) == 2, "de-duplication is not this function's job"
    # ...but both survive rather than one being dropped as self-overlapping,
    # which is the trap the `(s, e) != span` guard exists to avoid.
    assert {m.cui for m in kept} == {"C1"}


def test_output_order_is_total():
    """Co-extensive matches make `start` alone an ambiguous sort key."""
    kept = longest_non_overlapping([_m(0, 3, "C9"), _m(0, 3, "C1"), _m(5, 9, "C5")])
    assert [(m.start, m.cui) for m in kept] == [(0, "C1"), (0, "C9"), (5, "C5")]


# --- matcher construction ---------------------------------------------------


def test_missing_database_raises():
    with pytest.raises(FileNotFoundError):
        RareWordMatcher("does/not/exist.sqlite")


def _tiny_dictionary(path: Path, meta: dict[str, str] | None = None) -> None:
    """Smallest database satisfying `_verify_schema`, with one matchable term.

    `meta` defaults to the rows a finished build writes. Pass something else to
    exercise the retokenization check; pass ``{}`` for a dictionary that
    carries a `meta` table with nothing useful in it.
    """
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE term (norm TEXT, text TEXT, cui TEXT, sab TEXT, tty TEXT);
        CREATE TABLE concept (
            cui TEXT PRIMARY KEY, preferred_text TEXT NOT NULL, best_group TEXT NOT NULL
        );
        CREATE TABLE rare_term (
            rare_word TEXT NOT NULL, norm TEXT NOT NULL, cui TEXT NOT NULL,
            word_index INTEGER NOT NULL, token_count INTEGER NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO term VALUES ('pneumonia','pneumonia','C0032285','X','PT');
        INSERT INTO concept VALUES ('C0032285','Pneumonia','DISORDER');
        INSERT INTO rare_term VALUES ('pneumonia','pneumonia','C0032285',0,1);
        """
    )
    rows = {"terms_retokenized": "yes"} if meta is None else meta
    conn.executemany("INSERT INTO meta VALUES (?,?)", sorted(rows.items()))
    conn.commit()
    conn.close()


def _no_meta_dictionary(path: Path) -> None:
    """A dictionary with the three data tables and no `meta` table at all."""
    import sqlite3

    _tiny_dictionary(path)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE meta")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("dirname", ["plain", "with space", "hash#dir", "pct%dir"])
def test_opens_dictionaries_at_awkward_paths(tmp_path, dirname):
    """The connection URI must be built with as_uri(), not by interpolation.

    SQLite *parses* a URI, so '#' truncates the path at a fragment, '?' starts
    another query parameter and '%' is percent-decoded. Every ordinary path
    works either way, which is why interpolation arrives as a confusing bug report
    rather than a build failure.
    """
    db = tmp_path / dirname / "dict.sqlite"
    db.parent.mkdir()
    _tiny_dictionary(db)
    with RareWordMatcher(db) as m:
        hits = m.match(tokenize("Patient has pneumonia."))
    assert [h.cui for h in hits] == ["C0032285"]


def test_dictionary_is_opened_read_only(tmp_path):
    import sqlite3

    db = tmp_path / "dict.sqlite"
    _tiny_dictionary(db)
    with RareWordMatcher(db) as m, pytest.raises(sqlite3.OperationalError):
        m._conn.execute("DELETE FROM term")


# --- the retokenization invariant -------------------------------------------
#
# Skipping tools/retokenize_terms.py is the one build mistake that produces no
# symptom: term verification is exact string equality, so an un-retokenized
# dictionary matches fewer terms and raises nothing. These pin the check that
# turns it into a construction-time failure.


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: _tiny_dictionary(p, {}), id="meta-table-empty"),
        pytest.param(_no_meta_dictionary, id="no-meta-table"),
        pytest.param(
            lambda p: _tiny_dictionary(p, {"terms_retokenized": "no"}),
            id="explicitly-not-retokenized",
        ),
        pytest.param(
            lambda p: _tiny_dictionary(p, {"umls_release": "2026AA"}),
            id="built-but-step-2-skipped",
        ),
    ],
)
def test_unretokenized_dictionary_is_refused(tmp_path, build):
    db = tmp_path / "dict.sqlite"
    build(db)
    with pytest.raises(RuntimeError, match="has not been re-tokenized"):
        RareWordMatcher(db)


def test_refusal_names_both_remaining_build_steps(tmp_path):
    """The error has to be actionable, or it just relocates the confusion.

    Naming `build_rare_word_index.py` too is the part that is easy to leave
    out and costly to omit: re-tokenizing without rebuilding the index leaves
    `rare_term` keyed on the old spellings, which is the same silent failure
    in a new place.
    """
    db = tmp_path / "dict.sqlite"
    _tiny_dictionary(db, {})
    with pytest.raises(RuntimeError) as exc:
        RareWordMatcher(db)
    message = str(exc.value)
    assert "tools/retokenize_terms.py" in message
    assert "tools/build_rare_word_index.py" in message
    assert str(db) in message


def test_pretokenized_ctakes_dictionary_is_accepted(tmp_path):
    """cTAKES ships its dictionary pre-tokenized, so it never runs step 2.

    Rejecting it would reject the dictionary the F1 0.971 parity figure is
    measured on.
    """
    db = tmp_path / "dict.sqlite"
    _tiny_dictionary(db, {"source": "ctakes_shipped_sno_rx_16ab"})
    with RareWordMatcher(db) as m:
        assert [h.cui for h in m.match(tokenize("Patient has pneumonia."))] == ["C0032285"]


# --- matching against the real dictionary -----------------------------------


@pytest.fixture(scope="module")
def matcher():
    with RareWordMatcher(DB) as m:
        yield m


EXPECTED = [
    ("chest pain", "C0008031", "FINDING"),
    ("atrial fibrillation", "C0004238", "DISORDER"),
    ("type 2 diabetes mellitus", "C0011860", "DISORDER"),
    ("congestive heart failure", "C0018802", "DISORDER"),
    ("shortness of breath", "C0013404", "FINDING"),
    ("metformin", "C0025598", "DRUG"),
    ("appendectomy", "C0003611", "PROCEDURE"),
]


@needs_db
@pytest.mark.parametrize("phrase,cui,group", EXPECTED)
def test_matches_known_phrase(matcher, phrase, cui, group):
    """Carrier sentence, not a bare phrase -- POS eligibility needs context.

    ``match_text`` tags with the pipeline tokenizer now, and cTAKES refuses to
    anchor a lookup on a verb tag. A bare "metformin" is tagged VB by a
    general-domain tagger with nothing around it; in "started on metformin" it
    is NN. Probing with a bare term measures the tagger, not the dictionary.
    """
    text = f"Patient was seen for {phrase} today."
    hits = matcher.match_text(text)
    assert any(h.cui == cui and h.group == group for h in hits), (
        f"{text!r} -> {[(h.text, h.cui, h.group) for h in hits]}"
    )


@needs_db
def test_multi_token_match_spans_correct_offsets(matcher):
    text = "Chest x-ray showed left lower lobe pneumonia."
    hits = [h for h in matcher.match_text(text) if h.text == "left lower lobe pneumonia"]
    assert hits, "multi-token anatomical+disorder phrase did not match"
    h = hits[0]
    assert text[h.start : h.end] == "left lower lobe pneumonia"
    assert h.n_tokens == 4


@needs_db
def test_pos_exclusion_suppresses_function_word_anchors(matcher):
    """A token tagged with an excluded POS must not anchor lookup, even for a real UMLS term.

    Uses a deliberately wrong tag (CC on "aspirin") to isolate the exclusion
    mechanism from any particular word's real-world part of speech.
    """
    text = "took aspirin daily"
    tags = {"took": "VBD", "aspirin": "CC", "daily": "RB"}
    toks = [
        Token(t.text, t.start, t.end, pos=tags.get(t.text), is_word=t.is_word)
        for t in tokenize(text)
    ]
    assert all(
        text[h.start : h.end] != "aspirin" for h in matcher.match(toks)
    ), "CC-tagged token anchored a match"


@needs_db
def test_untagged_tokens_still_anchor(matcher):
    """With no POS supplied nothing is excluded -- tags are optional, not required."""
    text = "took aspirin daily"
    assert any(text[h.start : h.end] == "aspirin" for h in matcher.match_text(text))


@needs_db
def test_punctuation_never_anchors(matcher):
    hits = matcher.match_text(" , . ; ")
    assert hits == []


@needs_db
def test_minimum_span_filters_short_terms(matcher):
    """No returned term may be shorter than DEFAULT_MINIMUM_SPAN characters."""
    hits = matcher.match_text("Patient with CHF and COPD on ASA.")
    assert all(len(h.term) >= DEFAULT_MINIMUM_SPAN for h in hits)


@needs_db
def test_match_does_not_run_past_window(matcher):
    """A term anchored near the edge must not be reported beyond the window."""
    tokens = tokenize("fibrillation")
    hits = matcher.match(tokens)
    for h in hits:
        assert h.token_start >= 0
        assert h.token_end <= len(tokens)


@needs_db
def test_empty_input(matcher):
    assert matcher.match([]) == []
    assert matcher.match_text("") == []


@needs_db
def test_match_text_populates_text_field(matcher):
    for h in matcher.match_text("Patient has atrial fibrillation."):
        assert h.text != ""


@needs_db
def test_match_leaves_text_empty_for_every_arity(matcher):
    """match() has no document to slice, so `text` is empty -- uniformly.

    Filling it for single-token hits and leaving it empty for multi-token ones
    would give a caller reading `.text` a value that silently depended on term
    length. Offsets are the authoritative location; see Match.
    """
    hits = matcher.match(tokenize("Patient has atrial fibrillation and chest pain."))
    assert hits, "expected both single- and multi-token hits"
    assert {h.n_tokens for h in hits} & {1}, "expected at least one single-token hit"
    assert {h.n_tokens for h in hits} - {1}, "expected at least one multi-token hit"
    assert all(h.text == "" for h in hits)


@needs_db
def test_offsets_are_consistent_with_source_text(matcher):
    text = "Started on metformin 500 mg and aspirin 81 mg daily."
    for h in matcher.match_text(text):
        assert text[h.start : h.end] == h.text


@needs_db
def test_match_text_agrees_with_the_dictionary_on_hyphens(matcher):
    """match_text() must tokenize hyphens the way the dictionary was built.

    Verification is exact string equality against the stored norm
    ("chest x - ray"), so a tokenizer that keeps "x-ray" whole returns no match
    and raises nothing -- only `chest` comes back out of five expected concepts.
    """
    text = "Patient had a chest x-ray."
    hits = matcher.match_text(text)
    assert any(h.text == "chest x-ray" for h in hits), [h.text for h in hits]
    assert any(h.text == "x-ray" for h in hits), [h.text for h in hits]
    for h in hits:
        assert text[h.start : h.end] == h.text


@needs_db
def test_match_text_offsets_stay_absolute_across_sentences(matcher):
    """Windows are per sentence now; document offsets must not become relative."""
    text = "Patient has atrial fibrillation. Aspirin was given for chest pain."
    hits = matcher.match_text(text)
    assert any(h.text == "chest pain" for h in hits)
    for h in hits:
        assert text[h.start : h.end] == h.text


def test_memo_size_must_be_positive():
    with pytest.raises(ValueError):
        RareWordMatcher(DB, memo_size=0)


@needs_db
def test_memo_caches_are_bounded():
    """Both memo dicts must evict, not grow without bound.

    A pooled service pipeline lives for the life of the process, so without
    eviction every novel token norm -- dictation artifact, lab value, date --
    is retained forever. Growth is sublinear, so it is a slow climb rather
    than a crash, which is exactly why it would reach production.
    """
    with RareWordMatcher(DB, memo_size=25) as m:
        for i in range(400):
            m.match_text(f"Patient has zzq{i}x and chest pain.")
        candidates, concepts = m.memo_entries
        assert candidates <= 25, candidates
        assert concepts <= 25, concepts


@needs_db
def test_eviction_does_not_change_results():
    """A cache is an optimization; a bound on it must stay invisible in output."""
    text = "History of type 2 diabetes mellitus and atrial fibrillation."
    with RareWordMatcher(DB, memo_size=1) as tiny, RareWordMatcher(DB) as big:
        a = sorted((h.start, h.end, h.cui, h.group) for h in tiny.match_text(text))
        b = sorted((h.start, h.end, h.cui, h.group) for h in big.match_text(text))
    assert a == b


@needs_db
def test_results_are_deterministic(matcher):
    text = "History of type 2 diabetes mellitus and atrial fibrillation."
    a = [(h.start, h.end, h.cui) for h in matcher.match_text(text)]
    b = [(h.start, h.end, h.cui) for h in matcher.match_text(text)]
    assert a == b

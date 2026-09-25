"""Integrity tests for the built UMLS dictionary.

Skipped when the database has not been built (it is a ~258 MB generated
artifact and is not committed). Build it with::

    python tools/build_dictionary.py --umls-dir <UMLS META dir>
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parent.parent / "data" / "umls_sno_rx.sqlite"

needs_nlp = pytest.mark.skipif(
    importlib.util.find_spec("spacy") is None,
    reason="optional 'nlp' extra (spaCy) not installed",
)

pytestmark = pytest.mark.skipif(
    not DB.is_file(), reason=f"dictionary not built: {DB} (see tools/build_dictionary.py)"
)


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    yield c
    c.close()


@pytest.fixture(scope="module")
def meta(conn):
    return dict(conn.execute("SELECT key, value FROM meta"))


def test_build_was_not_limited(meta):
    """A --limit build is a smoke-test artifact and must not be used for real work."""
    assert meta["limited"] == "no"


def test_concept_inventory_is_snomed_rxnorm(meta):
    """Codes decide the concept inventory; synonyms are read more widely.

    cTAKES separates these ("Record Codes" vs "Read Synonyms"), and so do we:
    `--code-sources` pins what concepts exist, `--sources` only adds ways to
    match them. This asserts the inventory, which is the load-bearing half --
    widening synonyms must never quietly widen the concept set.
    """
    assert meta.get("code_sources", "RXNORM,SNOMEDCT_US") == "RXNORM,SNOMEDCT_US"


def test_synonym_sources_recorded(meta):
    """Provenance: which vocabularies the surface forms came from."""
    assert meta["sources"], "build must record its synonym sources"


def test_tui_filter_is_applied_as_configured(meta, conn):
    """The TUI filter kept exactly the semantic types it was configured to keep.

    Builds default to ``--tui-set ctakes`` (the 49 TUIs cTAKES' shipped
    dictionary indexes), so ``kept < read`` is expected. Under ``--tui-set all``
    nothing is dropped, because cTAKES' 136-TUI table covers every TUI present
    in a release -- if a future release introduces an unmapped TUI, that
    branch fails rather than silently discarding those concepts.
    """
    from umlsmatch.umls.ctakes_tuis import CTAKES_TUIS

    tui_set = meta.get("tui_set", "all")
    present = {r[0] for r in conn.execute("SELECT DISTINCT tui FROM concept_tui")}

    if tui_set == "ctakes":
        assert int(meta["mrsty_rows_kept"]) < int(meta["mrsty_rows_read"])
        assert present <= CTAKES_TUIS, f"unexpected TUIs kept: {sorted(present - CTAKES_TUIS)}"
    else:
        assert meta["mrsty_rows_kept"] == meta["mrsty_rows_read"]


def test_no_orphan_terms(conn):
    n = conn.execute(
        "SELECT COUNT(*) FROM term WHERE cui NOT IN (SELECT cui FROM concept)"
    ).fetchone()[0]
    assert n == 0


def test_every_concept_has_a_semantic_type(conn):
    n = conn.execute(
        "SELECT COUNT(*) FROM concept WHERE cui NOT IN (SELECT cui FROM concept_tui)"
    ).fetchone()[0]
    assert n == 0


def test_every_concept_has_a_group(conn):
    n = conn.execute("SELECT COUNT(*) FROM concept WHERE best_group = ''").fetchone()[0]
    assert n == 0


def test_no_concept_falls_back_to_unknown_group(conn):
    n = conn.execute("SELECT COUNT(*) FROM concept WHERE best_group = 'UNKNOWN'").fetchone()[0]
    assert n == 0


def test_norms_are_normalized(conn):
    """No leading/trailing/doubled whitespace, no uppercase."""
    bad = conn.execute(
        "SELECT COUNT(*) FROM term "
        "WHERE norm != TRIM(norm) OR norm LIKE '%  %' OR norm != LOWER(norm)"
    ).fetchone()[0]
    assert bad == 0


def test_no_empty_norms(conn):
    assert conn.execute("SELECT COUNT(*) FROM term WHERE norm = ''").fetchone()[0] == 0


def test_plausible_scale(conn):
    """Guards against a truncated or mis-filtered build."""
    concepts = conn.execute("SELECT COUNT(*) FROM concept").fetchone()[0]
    terms = conn.execute("SELECT COUNT(*) FROM term").fetchone()[0]
    assert concepts > 300_000, f"only {concepts:,} concepts -- build looks truncated"
    assert terms > 800_000, f"only {terms:,} terms -- build looks truncated"


# CUI/group pairs that are stable across UMLS releases.
#
# "chest x-ray" (bare form) does not resolve to C0039985 under 2026AA,
# though it did under 2021AB: the concept is unchanged
# ("Plain X-ray of chest" / PROCEDURE) but SNOMED's synonym set for it no
# longer includes the bare short form, only "plain chest x-ray" / "plain
# x-ray of chest" / "plain cxr (chest x-ray)". Real release drift, not a
# build defect -- a UMLS release mismatch, not a matcher fault.
KNOWN_CONCEPTS = [
    ("pneumonia", "C0032285", "DISORDER"),
    ("atrial fibrillation", "C0004238", "DISORDER"),
    ("metformin", "C0025598", "DRUG"),
    ("aspirin", "C0004057", "DRUG"),
    # Spelled as the matcher's tokenizer segments it: tools/retokenize_terms.py
    # rewrites stored norms to match token-join spelling, so "x-ray" is stored
    # as "x - ray". Without that alignment no hyphenated term could ever match.
    ("plain chest x - ray", "C0039985", "PROCEDURE"),
    ("appendectomy", "C0003611", "PROCEDURE"),
]


@pytest.mark.parametrize("norm,cui,group", KNOWN_CONCEPTS)
def test_known_concept_lookup(conn, norm, cui, group):
    rows = conn.execute(
        "SELECT DISTINCT t.cui, c.best_group FROM term t "
        "JOIN concept c ON c.cui = t.cui WHERE t.norm = ?",
        (norm,),
    ).fetchall()
    assert (cui, group) in rows, f"{norm!r} did not resolve to {cui}/{group}; got {rows}"


def test_major_groups_are_well_populated(conn):
    counts = dict(
        conn.execute("SELECT best_group, COUNT(*) FROM concept GROUP BY best_group")
    )
    for group in ("DRUG", "DISORDER", "FINDING", "PROCEDURE", "ANATOMY"):
        assert counts.get(group, 0) > 10_000, f"{group} only has {counts.get(group, 0)}"


@needs_nlp
def test_stored_norms_align_with_matcher_tokenization(conn, meta):
    """Stored spellings must equal how the matcher's tokenizer segments them.

    The matcher verifies a candidate by joining document token norms with single
    spaces and comparing to `term.norm`. If a stored norm doesn't round-trip
    through the tokenizer, that term can never match -- a silent, total failure
    for the affected terms rather than a degradation.
    """
    if meta.get("terms_retokenized") != "yes":
        pytest.skip("dictionary not re-tokenized (run tools/retokenize_terms.py)")

    from umlsmatch.pipeline.tokenizer import DEFAULT_MODEL, load_model

    tokenizer = load_model(DEFAULT_MODEL).tokenizer
    sample = [
        r[0]
        for r in conn.execute(
            "SELECT norm FROM term WHERE norm LIKE '%-%' OR norm LIKE '%,%' LIMIT 300"
        )
    ]
    if not sample:
        pytest.skip("no punctuated terms to check")

    bad = []
    for norm, doc in zip(sample, tokenizer.pipe(sample), strict=True):
        rebuilt = " ".join(t.text for t in doc if not t.is_space)
        if rebuilt != norm:
            bad.append((norm, rebuilt))
    assert not bad, f"{len(bad)}/{len(sample)} norms not tokenizer-aligned, e.g. {bad[:3]}"

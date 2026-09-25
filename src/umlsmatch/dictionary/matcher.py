"""Pure-Python rare-word dictionary matcher.

A port of cTAKES' ``DefaultJCasTermAnnotator.findTerms`` (ctakes-dictionary-
lookup-fast). Despite the name often attached to this stage, cTAKES does **not**
use an Aho-Corasick automaton. Each dictionary term is indexed by its rarest
constituent token together with that token's offset in the term; at lookup time
every eligible token in the window fetches its candidate terms and each
candidate is verified by aligning it against the surrounding tokens.

For a pure-Python implementation this is the better algorithm anyway: the index
is a SQLite B-tree, so lookup memory stays flat, whereas a pure-Python
Aho-Corasick over ~745k phrases would need a multi-million-node dict trie
resident in RAM.

Faithful to the Java, per ``DefaultJCasTermAnnotator`` lines 54-86:

  * candidate terms whose text is shorter than ``minimum_span`` chars are skipped
  * single-token terms match at the anchor token directly
  * multi-token terms are anchored at ``index - word_index`` and must fit
    entirely inside the window
  * the matched span runs from the first token's start to the last token's end

Standard library only -- no third-party dependencies.
"""

from __future__ import annotations

import sqlite3
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

__all__ = [
    "DEFAULT_EXCLUSION_TAGS",
    "DEFAULT_MINIMUM_SPAN",
    "PRETOKENIZED_SOURCES",
    "Match",
    "RareWordMatcher",
    "Token",
    "longest_non_overlapping",
    "tokenize",
]

#: One ``rare_term`` row, as ``RareWordMatcher._QUERY`` selects it:
#: ``(norm, cui, word_index, token_count)``.
_CandidateRow = tuple[str, str, int, int]

#: Value type of a memo cache, so :meth:`RareWordMatcher._remember` can serve
#: both caches without widening either one to ``Any``.
_V = TypeVar("_V")

#: cTAKES ``JCasTermAnnotator.DEFAULT_EXCLUSION_TAGS`` -- Penn Treebank tags that
#: may not anchor a lookup. Verbs, conjunctions, cardinals, determiners,
#: pronouns, prepositions and wh-words.
DEFAULT_EXCLUSION_TAGS = frozenset(
    # Kept as a space-separated string so it stays diffable against the Java
    # constant it transcribes; a list of 25 quoted strings does not.
    "VB VBD VBG VBN VBP VBZ CC CD DT EX IN LS MD PDT POS PP PP$ "  # noqa: SIM905
    "PRP PRP$ RP TO WDT WP WPS WRB".split()
)

#: cTAKES ``JCasTermAnnotator.DEFAULT_MINIMUM_SPAN``. Applied to the candidate
#: term's character length, not to the anchor token.
DEFAULT_MINIMUM_SPAN = 3

#: ``meta.source`` values whose term spellings arrive tokenizer-aligned from
#: upstream and therefore never pass through ``tools/retokenize_terms.py``.
#: cTAKES ships its own dictionary pre-tokenized (``'2 , 4 -
#: dichlorophenoxyacetic acid'``), so ``import_ctakes_dictionary.py`` produces a
#: database that satisfies the alignment invariant without the rewrite step --
#: and that script refuses to run on one, since re-tokenizing it would corrupt
#: the spellings. Without this list the parity dictionary, which is the whole
#: basis of the F1 0.971 figure, would be rejected as un-retokenized.
PRETOKENIZED_SOURCES = frozenset({"ctakes_shipped_sno_rx_16ab"})


@dataclass(frozen=True, slots=True)
class Token:
    """A token in a lookup window, carrying document character offsets."""

    text: str
    start: int
    end: int
    pos: str | None = None
    #: True for punctuation, numbers, symbols and contractions -- cTAKES
    #: excludes these as lookup anchors regardless of POS.
    is_word: bool = True
    #: Dependency relation to :attr:`head`, e.g. ``"conj"``. Empty when the
    #: producer did no parsing -- :func:`tokenize` and hand-built test tokens
    #: both leave it so, and every consumer must degrade gracefully rather
    #: than assume a parse is present.
    dep: str = ""
    #: Index of this token's syntactic head *within its own window*, or -1
    #: for the root, for an unparsed token, or when the head fell outside the
    #: window (sentence splitting can separate a token from its head).
    head: int = -1

    @property
    def norm(self) -> str:
        return self.text.casefold()


@dataclass(frozen=True, slots=True)
class Match:
    """A dictionary hit aligned to a document span.

    ``start``/``end`` are the authoritative location: they index the document
    the tokens came from. ``text`` is a convenience copy of that slice and is
    left empty by :meth:`RareWordMatcher.match`, which is handed tokens rather
    than the document and so cannot reproduce the original spacing.
    :meth:`RareWordMatcher.match_text` and
    :class:`~umlsmatch.analyze.ClinicalPipeline` both fill it in.
    """

    cui: str
    term: str
    #: Document slice ``text[start:end]``, or '' when the producer had no
    #: document to slice; never reconstruct it by joining token texts.
    text: str
    start: int
    end: int
    token_start: int
    token_end: int  # exclusive
    group: str = ""

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start

    def __len__(self) -> int:
        return self.end - self.start


def tokenize(text: str, *, split_hyphens: bool = False) -> list[Token]:
    """Minimal offset-preserving tokenizer, for tests and demos.

    Deliberately NOT a cTAKES-equivalent tokenizer -- the real pipeline uses
    ``TokenizerAnnotatorPTB`` plus ``ContextDependentTokenizerAnnotator``, which
    handle dates, doses, fractions and ranges. This exists so the matcher can be
    exercised standalone; swap in the real tokenizer when it lands.

    `split_hyphens` breaks ``x-ray`` into ``['x', '-', 'ray']`` and ``crohn's``
    into ``['crohn', "'s"]``, which is what spaCy does and therefore what
    ``tools/retokenize_terms.py`` baked into every built dictionary. It is off
    by default so existing callers are unaffected, and
    :meth:`RareWordMatcher.match_text` turns it on -- see that method for why
    the mismatch is worth a flag at all.
    """
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isalnum():
            j = i
            while j < n and (text[j].isalnum() or text[j] in "-'"):
                if split_hyphens and text[j] == "-":
                    break
                # "crohn's" -> "crohn" + "'s", but "o'brien" stays whole:
                # spaCy splits a clitic, not an internal apostrophe.
                if split_hyphens and text[j] == "'" and text[j + 1 : j + 2] == "s":
                    break
                j += 1
            word = text[i:j]
            tokens.append(Token(word, i, j, is_word=any(c.isalpha() for c in word)))
            i = j
        elif split_hyphens and ch == "'" and text[i + 1 : i + 2] == "s":
            tokens.append(Token(text[i : i + 2], i, i + 2, is_word=True))
            i += 2
        else:
            tokens.append(Token(ch, i, i + 1, is_word=False))
            i += 1
    return tokens


class RareWordMatcher:
    """Looks up UMLS concepts in a token window using the rare-word index.

    The database must have been built by ``tools/build_dictionary.py``, then
    ``tools/retokenize_terms.py``, then ``tools/build_rare_word_index.py`` --
    all three, in that order. Construction verifies it, because skipping the
    middle step is the one build error that produces no symptom at all; see
    :meth:`_verify_retokenized`.

    **Not thread-safe.** The SQLite connection is opened with
    ``check_same_thread=False`` so it can be handed to another thread, but the
    candidate and concept memo dicts are unsynchronized and concurrent
    ``match()`` calls will interleave on them. Give each thread its own
    instance, or scale with processes, one matcher per worker.

    Args:
        db_path: dictionary built by ``tools/build_dictionary.py``, then
            ``tools/retokenize_terms.py``, then
            ``tools/build_rare_word_index.py``.
        exclusion_tags: Penn Treebank tags that may not anchor a lookup.
        minimum_span: candidate terms shorter than this many characters are
            skipped, applied to the term text as in the Java.
        attach_groups: look up each hit's semantic group. Off saves a query per
            distinct CUI when the caller does not need it.
        cache_size: SQLite page cache, in KiB. The default is ~195 MB and is
            the dominant fixed cost of an instance; lower it when running
            several matchers over the same file, since the OS page cache is
            shared between them anyway.
        memo_size: maximum entries in each of the two memo caches before the
            least recently used is evicted. Bounds resident growth in a
            long-lived process; raise it for batch work that revisits the same
            vocabulary, lower it under a tight container limit.
    """

    _QUERY = (
        "SELECT norm, cui, word_index, token_count FROM rare_term WHERE rare_word = ?"
    )
    # One query serves both `group` and `preferred_text`: they live in the same
    # `concept` row, so fetching them separately doubles the lookups and needs
    # two caches for one row.
    _CONCEPT_QUERY = "SELECT best_group, preferred_text FROM concept WHERE cui = ?"

    def __init__(
        self,
        db_path: str | Path,
        *,
        exclusion_tags: Iterable[str] = DEFAULT_EXCLUSION_TAGS,
        minimum_span: int = DEFAULT_MINIMUM_SPAN,
        attach_groups: bool = True,
        cache_size: int = 200_000,
        memo_size: int = 100_000,
    ) -> None:
        # Argument validation before any I/O. Ordering matters twice here: the
        # connection below is opened before this check used to run, so a bad
        # `memo_size` raised with the handle already open and nothing to close
        # it; and the dictionary check right after raises FileNotFoundError,
        # which masked this ValueError entirely on any machine without a built
        # dictionary -- CI included.
        if memo_size < 1:
            raise ValueError(f"memo_size must be >= 1, got {memo_size}")

        db_path = Path(db_path)
        if not db_path.is_file():
            raise FileNotFoundError(
                f"dictionary not found: {db_path}\n"
                "Build one with:  python -m umlsmatch.build --umls-dir <UMLS META dir>"
            )
        # Read-only: the matcher must never mutate the dictionary artifact.
        #
        # `as_uri()` rather than an f-string: in URI mode SQLite parses what it
        # is handed, so a path is not a string to interpolate. '#' starts a
        # fragment and truncates the path, '?' starts another query parameter,
        # and '%' is percent-decoded -- so "C:/data/100%_umls/db.sqlite" or any
        # directory with a '#' in it opens the wrong file or none at all, while
        # every ordinary path works. That combination surfaces as a confusing
        # one-off bug report, not a build failure.
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        self.db_path = db_path
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.execute(f"PRAGMA cache_size = {-abs(cache_size)}")
        self._verify_schema()

        self.exclusion_tags = frozenset(exclusion_tags)
        self.minimum_span = minimum_span
        self.attach_groups = attach_groups
        self.memo_size = memo_size  # validated at the top, before the connection
        # Insertion-ordered, evicted least-recently-used first. Plain dicts
        # would do for a script and are wrong for the service: a pooled
        # pipeline lives for the life of the process, and every novel token
        # norm -- dictation artifact, lab value, date, misspelling -- would
        # keep its candidate rows forever. That growth is sublinear (distinct
        # token *types*, not occurrences), so it is a slow climb rather than a
        # crash, but it does not belong in a container whose limit already
        # covers a page cache and a spaCy model per pipeline.
        self._candidate_cache: OrderedDict[str, tuple[_CandidateRow, ...]] = OrderedDict()
        #: cui -> (best_group, preferred_text), memoized together.
        self._concept_cache: OrderedDict[str, tuple[str, str]] = OrderedDict()
        #: match_text() warns once per instance, not once per call.
        self._warned_tokenizer = False

    def _verify_schema(self) -> None:
        tables = {
            r[0]
            for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = {"rare_term", "concept", "term"} - tables
        if missing:
            raise RuntimeError(
                f"dictionary is missing table(s): {', '.join(sorted(missing))}. "
                "Run tools/build_rare_word_index.py."
            )
        self._verify_retokenized("meta" in tables)

    def _verify_retokenized(self, has_meta: bool) -> None:
        """Refuse a dictionary whose terms are not spelled the way we tokenize.

        **This is the one invariant that fails silently.** Every other way of
        getting the dictionary wrong raises: a missing table is caught above, a
        missing file in ``__init__``, a bad path by SQLite. But verification in
        :meth:`match` is exact string equality between a stored ``norm`` and the
        document's token norms joined with spaces, so a dictionary spelled
        ``"chest x-ray"`` against a tokenizer producing ``"chest x - ray"``
        matches nothing, raises nothing, and returns a shorter annotation list
        that looks entirely plausible. Per ``tools/retokenize_terms.py`` that is
        ~96k terms, roughly 10% of a 2026AA build: every hyphenated term, every
        clitic, and much of the drug vocabulary.

        The build writes ``terms_retokenized='yes'`` and the invariant was
        already asserted in ``tests/test_dictionary.py`` and documented as a
        manual check in ``docs/USER_GUIDE.md`` -- everywhere, that is, except
        the code that depends on it. This closes that.

        No bypass flag, deliberately: a dictionary pre-tokenized by some other
        route can record the fact itself, and writing the row *is* the
        assertion. A keyword argument that skips the check would be one more
        way to get the silent failure back.
        """
        meta = dict(self._conn.execute("SELECT key, value FROM meta")) if has_meta else {}
        if meta.get("terms_retokenized") == "yes":
            return
        if meta.get("source") in PRETOKENIZED_SOURCES:
            return
        raise RuntimeError(
            f"dictionary has not been re-tokenized: {self.db_path}\n"
            "Its terms are spelled the way MRCONSO spells them, not the way the "
            "tokenizer segments text, so every hyphenated and clitic term "
            "(~10% of the vocabulary) would silently fail to match.\n"
            "Fix it by finishing the build, in this order:\n"
            f"  python tools/retokenize_terms.py --db {self.db_path}\n"
            f"  python tools/build_rare_word_index.py --db {self.db_path}\n"
            "The second step is not optional: rare-word statistics must be "
            "recomputed over the final spellings.\n"
            "If these terms were pre-tokenized by some other route, record that "
            "in the dictionary itself:\n"
            "  INSERT OR REPLACE INTO meta VALUES ('terms_retokenized','yes');"
        )

    # -- internals ---------------------------------------------------------

    def _remember(self, cache: OrderedDict[str, _V], key: str, value: _V) -> None:
        """Store `value`, evicting the least recently used entry past ``memo_size``.

        Deliberately not ``functools.lru_cache``: on a method it keys on
        ``self``, so one cache is shared by every instance of the class and
        every matcher stays alive as long as any entry does. In a
        ``PipelinePool`` that turns a per-pipeline cache into a cross-pipeline
        one -- exactly the sharing the pool exists to prevent, and it would be
        a data race rather than a visible error.
        """
        cache[key] = value
        if len(cache) > self.memo_size:
            cache.popitem(last=False)

    def _candidates(self, word: str) -> tuple[_CandidateRow, ...]:
        """Candidate terms keyed by rare word, memoized (bounded LRU)."""
        hit = self._candidate_cache.get(word)
        if hit is None:
            hit = tuple(self._conn.execute(self._QUERY, (word,)))
            self._remember(self._candidate_cache, word, hit)
        else:
            self._candidate_cache.move_to_end(word)
        return hit

    def _concept(self, cui: str) -> tuple[str, str]:
        """(best_group, preferred_text) for `cui`, memoized. ('', '') if unknown."""
        hit = self._concept_cache.get(cui)
        if hit is None:
            row = self._conn.execute(self._CONCEPT_QUERY, (cui,)).fetchone()
            hit = (row[0] or "", row[1] or "") if row else ("", "")
            self._remember(self._concept_cache, cui, hit)
        else:
            self._concept_cache.move_to_end(cui)
        return hit

    @property
    def memo_entries(self) -> tuple[int, int]:
        """``(candidate_entries, concept_entries)`` currently held.

        Exposed so a deployment can confirm the bound is doing something rather
        than assume it; ``umlsmatch.service.api`` reports it as a gauge.
        """
        return len(self._candidate_cache), len(self._concept_cache)

    def preferred_text(self, cui: str) -> str:
        """The concept's canonical label, or '' if the CUI is unknown."""
        return self._concept(cui)[1]

    def _group(self, cui: str) -> str:
        if not self.attach_groups:
            return ""
        return self._concept(cui)[0]

    def _is_anchor(self, token: Token) -> bool:
        """Port of the eligibility test in ``getAnnotationsInWindow``."""
        if not token.is_word:
            return False
        # An untagged token stays eligible: cTAKES treats a null POS as
        # non-excluded rather than as a reason to skip.
        return token.pos is None or token.pos not in self.exclusion_tags

    # -- public API --------------------------------------------------------

    def match(self, tokens: Sequence[Token]) -> list[Match]:
        """Find all dictionary hits within one lookup window (a sentence).

        Returns every hit, including overlaps, exactly as cTAKES does. Apply
        :func:`longest_non_overlapping` if you need a reduced set.

        `tokens` must be the window's token list with whitespace already
        removed -- multi-token terms are verified by joining token norms with
        single spaces, so a retained newline token makes any term spanning it
        unmatchable. :func:`~umlsmatch.pipeline.tokenizer.annotate_sentences`
        produces lists in that form.

        Returned matches carry empty ``text``; see :class:`Match`.
        """
        norms = [t.norm for t in tokens]
        n = len(tokens)
        out: list[Match] = []

        for i, token in enumerate(tokens):
            if not self._is_anchor(token):
                continue

            for term_norm, cui, word_index, token_count in self._candidates(norms[i]):
                if len(term_norm) < self.minimum_span:
                    continue

                if token_count == 1:
                    out.append(
                        Match(
                            cui=cui,
                            term=term_norm,
                            text="",  # see Match.text -- offsets are authoritative
                            start=token.start,
                            end=token.end,
                            token_start=i,
                            token_end=i + 1,
                            group=self._group(cui),
                        )
                    )
                    continue

                start = i - word_index
                if start < 0 or start + token_count > n:
                    # Term would extend beyond the window.
                    continue
                end = start + token_count  # exclusive

                if " ".join(norms[start:end]) != term_norm:
                    continue

                out.append(
                    Match(
                        cui=cui,
                        term=term_norm,
                        text="",  # see Match.text -- offsets are authoritative
                        start=tokens[start].start,
                        end=tokens[end - 1].end,
                        token_start=start,
                        token_end=end,
                        group=self._group(cui),
                    )
                )
        return out

    def _tokenize_for_match(self, text: str) -> list[list[Token]]:
        """Lookup windows for `text`, spelled the way the dictionary is spelled.

        **Why this is not just ``tokenize(text)``.** Verification in
        :meth:`match` is exact string equality against a stored term norm, so a
        tokenizer that disagrees with the one that *built* the dictionary
        produces no match and no error. Every dictionary here was retokenized
        with spaCy by ``tools/retokenize_terms.py``, which splits hyphens and
        clitics: the stored term is ``"chest x - ray"``, and the built-in
        tokenizer's ``"chest x-ray"`` silently matched nothing but ``chest`` --
        four of five expected concepts gone, quietly.

        So the pipeline tokenizer is used when the optional ``nlp`` extra is
        installed, which makes agreement exact by construction rather than by
        a rule kept in sync by hand. Without it, :func:`tokenize` runs with
        ``split_hyphens=True``, which covers ordinary clinical spellings but
        not spaCy's behavior on chemical names, so the caller is warned once
        rather than left to discover the gap in output.
        """
        try:
            from umlsmatch.pipeline.tokenizer import annotate_sentences
        except ImportError:
            if not self._warned_tokenizer:
                self._warned_tokenizer = True
                warnings.warn(
                    "match_text() is approximating the tokenizer the dictionary "
                    "was built with; install the 'nlp' extra for exact matching, "
                    "or pass pipeline tokens to match() directly.",
                    UserWarning,
                    stacklevel=3,
                )
            return [tokenize(text, split_hyphens=True)]
        return [w for w in annotate_sentences(text) if w]

    def match_text(self, text: str) -> list[Match]:
        """Convenience: tokenize `text` and match, one window per sentence.

        The sentence is cTAKES' lookup window, so this splits on sentences when
        it can -- ``token_start``/``token_end`` on the returned matches index
        each window, not the document, exactly as in :meth:`match`. Document
        character offsets are unaffected and remain authoritative.

        **Probe with a sentence, not a bare term.** With the ``nlp`` extra
        installed the tokens carry real POS tags, and cTAKES will not anchor a
        lookup on a verb tag (``DEFAULT_EXCLUSION_TAGS``). A general-domain
        tagger handed the single word "metformin" calls it ``VB`` and the
        lookup is skipped; in "started on metformin" it is ``NN`` and matches.
        That is the pipeline's real behavior rather than an artifact of this
        method -- which is the point of routing through the same tokenizer --
        but it does mean a one-word probe measures the tagger.
        """
        matches = [m for window in self._tokenize_for_match(text) for m in self.match(window)]
        return [
            Match(
                cui=m.cui,
                term=m.term,
                text=text[m.start : m.end],
                start=m.start,
                end=m.end,
                token_start=m.token_start,
                token_end=m.token_end,
                group=m.group,
            )
            for m in matches
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RareWordMatcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def longest_non_overlapping(matches: Iterable[Match]) -> list[Match]:
    """Greedily keep the longest matches, dropping any that *partially* overlap.

    cTAKES itself emits overlapping annotations and resolves them downstream;
    this is a convenience for consumers that want a flat span set. Ties on
    length are broken by earlier start, then by CUI for determinism.

    **Co-extensive matches are all kept.** The question this function answers is
    "which spans", and two CUIs on one span are not a disagreement about the
    span -- they are one span the vocabulary maps to two concepts. ``chf`` is
    both ``C0009714`` and ``C0018802`` in a 2026AA build, both ``DISORDER``, and
    the earlier form of this function dropped whichever sorted later on CUI:
    a real concept discarded, silently, by alphabetical accident. A strict
    sub-span is genuinely redundant with its container and is still dropped, so
    ``chest pain`` continues to suppress ``chest`` and ``pain``.

    ``resolve_overlaps`` remains off by default on
    :class:`~umlsmatch.analyze.ClinicalPipeline` -- keeping every overlap
    matches cTAKES and measures better. This only makes the reduced set an
    honest reduction.
    """
    ordered = sorted(matches, key=lambda m: (-(m.end - m.start), m.start, m.cui))
    kept: list[Match] = []
    # Spans already accepted. A match is rejected only if it overlaps one of
    # these *without being identical to it*, so every alias of an accepted span
    # survives while anything strictly inside one does not.
    taken: set[tuple[int, int]] = set()
    for m in ordered:
        span = (m.start, m.end)
        if any(m.start < e and s < m.end for s, e in taken if (s, e) != span):
            continue
        kept.append(m)
        taken.add(span)
    return sorted(kept, key=lambda m: (m.start, m.end, m.cui))

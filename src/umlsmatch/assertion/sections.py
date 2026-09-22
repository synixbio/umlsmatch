"""Clinical section headers, and which sections are lists of negative findings.

cTAKES has a section annotator; this is the much smaller piece of one that
negation actually needs. It exists because of a failure the token window and
the dependency parse both leave behind.

**The problem.** A Review-of-Systems entry is a negation trigger followed by
an arbitrarily long list:

    Review of Systems: Negative for chills, fever, night sweats, weight loss,
    fatigue, headache, visual changes, chest pain, palpitations, cough ...

``MAX_SCOPE_TOKENS`` stops after eight tokens. Coordination propagation
(``negation.COORDINATION_DEPS``) gets much further, but it is only as good as
the parse, and a general-domain parser handed a forty-token comma list of bare
nouns does not reliably chain it -- these lists have no verb, no clause
structure, and often no sentence-final period. The section header is the one
signal that survives all of that, because it is *lexical*.

**The rule.** Inside a negative-findings section, a trigger's scope is not
capped -- it runs to the next terminator or the sentence edge, exactly the
``max_scope=None`` behavior. Note what this does **not** do: it does not negate
the section's contents outright. A trigger must still fire. "Review of Systems:
Positive for cough. Negative for fever." negates only the second sentence, and
a section with no trigger in it negates nothing.

That restraint is the point. Blanket-negating a section would be the obvious
implementation and would be wrong: Review-of-Systems sections routinely carry
positives, and Physical Exam sections mix findings freely. Uncapping a scope
that a trigger already opened is a much weaker claim, and it is the one the
evidence supports.

**Why only these sections *open* one.** :data:`NEGATIVE_FINDINGS_SECTIONS` is
deliberately short. It covers headers whose content is *conventionally* a
negative checklist. Physical Exam is excluded despite being full of negatives,
because it is equally full of affirmed findings and its sentences are short
enough that the window already handles them.

**Why a much longer list *closes* one.** Section state spans windows, so
something has to end it, and for a long time the only thing that did was a
window ending in ``:``. That leaks: ``IMPRESSION``, ``PLAN`` and
``HOSPITAL COURSE`` carry no colon, spaCy does not split a bare header onto its
own window, and a header whose colon is followed by lowercase text
("physical exam: unremarkable") is not split either. One Review of Systems then
uncapped every trigger in the rest of the note -- measured on a synthetic note,
a pleural effusion reported in an IMPRESSION flipped to negated purely because
of a header several sentences earlier. So :data:`OTHER_SECTIONS` exists only to
answer "has a new section started?", and any recognized header sets the state
to whatever *that* section implies.

**Honesty about provenance.** :data:`OTHER_SECTIONS` is hand-curated from
ordinary discharge-summary and progress-note conventions, not transcribed from
a cTAKES resource -- the same caveat :mod:`umlsmatch.assertion.negation` makes
about its trigger lexicon. It is used only to *close* a negative-findings
section, so a missing entry costs a leaked scope and a wrong entry costs a
re-capped one, never a wholesale polarity flip.

**What extending this table is worth: measured, and the answer is nothing.**
The residual leak is narrow -- it needs a header this table does not recognize
*and* inline text after the colon, since :func:`is_header_only` already closes
on any window ending at its colon. Scanning a real-note corpus not included here for label-colon
windows this table misses finds 686 occurrences across 317 distinct labels, of
which only **49 are leak-shaped**; the colon fallback covers the rest. Four
candidate changes were scored against the silver standard, all attributes:

===========================================================  ===========
variant                                                      negated F1
===========================================================  ===========
baseline                                                     0.614
plus the administrative labels below                         0.614
plus subsystem labels ("Cardiovascular:") as closers         0.614
subsystem labels instead *continuing* the ROS                0.613
a generic capitalized-label-colon rule closing any section   0.615
===========================================================  ===========

``subject``, ``history_of`` and ``uncertain`` were identical in every variant.
Two conclusions, both worth keeping:

  * **The generic regex rule is not adopted.** +0.001 on one attribute is one
    or two mentions, not evidence, and it would close a section on any
    capitalized label-colon in running narrative -- a false-close risk taken
    for a gain indistinguishable from noise.
  * **Subsystem labels are deliberately absent.** They are ambiguous by
    position: the same "Cardiovascular:" heads a negative checklist under
    Review of Systems and a mixed findings list under Physical Exam. Making
    them continue the ROS *lowered* precision, which is that ambiguity showing
    up as over-negation.

The administrative entries were kept anyway, at zero measured cost, because
they are unambiguous and the corpus behind these calls was a narrow set of
real-note templates -- absence of an effect there is not evidence of
absence elsewhere. That is the honest reason, and it is a weaker one than a
number. The 20-note synthetic corpus shipped here is narrower still (one
note per specialty, machine-written), so it does not strengthen the case;
nothing in this module was re-measured against it.

Usage::

    from umlsmatch.assertion.sections import negative_findings_section

    section = negative_findings_section(tokens)   # None when not a header
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence

from umlsmatch.dictionary.matcher import Token

__all__ = [
    "NEGATIVE_FINDINGS_SECTIONS",
    "OTHER_SECTIONS",
    "is_header_only",
    "negative_findings_section",
    "section_header",
    "track_sections",
]

#: Section headers whose body is conventionally a list of negatives. Keys are
#: normalized header phrases; the value is the canonical section name reported
#: back, so several spellings collapse to one label.
_SECTION_SPELLINGS: dict[tuple[str, ...], str] = {
    ("review", "of", "systems"): "review_of_systems",
    ("review", "of", "symptoms"): "review_of_systems",
    ("ros",): "review_of_systems",
    ("systems", "review"): "review_of_systems",
    ("allergies",): "allergies",
    ("drug", "allergies"): "allergies",
    ("medication", "allergies"): "allergies",
    ("allergies", "/", "reactions"): "allergies",
}

#: Headers that merely mark *some other* section starting. Recognizing one is
#: how a negative-findings section ends; see the module docstring. These never
#: uncap a scope -- only :data:`NEGATIVE_FINDINGS_SECTIONS` does that.
_OTHER_SPELLINGS: dict[tuple[str, ...], str] = {
    ("chief", "complaint"): "chief_complaint",
    ("history", "of", "present", "illness"): "history_of_present_illness",
    ("hpi",): "history_of_present_illness",
    ("past", "medical", "history"): "past_medical_history",
    ("medical", "history"): "past_medical_history",
    ("pmh",): "past_medical_history",
    ("past", "surgical", "history"): "past_surgical_history",
    ("surgical", "history"): "past_surgical_history",
    ("family", "history"): "family_history",
    ("social", "history"): "social_history",
    ("medications",): "medications",
    ("current", "medications"): "medications",
    ("home", "medications"): "medications",
    ("home", "meds"): "medications",
    ("discharge", "medications"): "medications",
    ("admission", "medications"): "medications",
    ("medications", "on", "admission"): "medications",
    ("physical", "exam"): "physical_exam",
    ("physical", "examination"): "physical_exam",
    ("vital", "signs"): "vital_signs",
    ("vitals",): "vital_signs",
    ("laboratory", "data"): "laboratory",
    ("laboratory",): "laboratory",
    ("labs",): "laboratory",
    ("imaging",): "imaging",
    ("radiology",): "imaging",
    ("assessment",): "assessment",
    ("assessment", "and", "plan"): "assessment",
    ("impression",): "assessment",
    ("plan",): "plan",
    ("hospital", "course"): "hospital_course",
    ("diagnosis",): "diagnosis",
    ("diagnoses",): "diagnosis",
    ("discharge", "diagnosis"): "diagnosis",
    ("discharge", "diagnoses"): "diagnosis",
    ("admission", "diagnosis"): "diagnosis",
    ("procedure",): "procedures",
    ("procedures",): "procedures",
    ("findings",): "findings",
    ("disposition",): "disposition",
    ("discharge", "instructions"): "discharge_instructions",
    ("follow", "up"): "follow_up",
    ("subjective",): "subjective",
    ("objective",): "objective",
    ("indication",): "indication",
    ("indications",): "indication",
    ("technique",): "technique",
    ("comparison",): "comparison",
    # Administrative labels, from a scan of a real-note corpus not included here for label-colon
    # windows this table did not recognize. They are here because they are
    # unambiguously top-level: none of them can occur as a subsystem line
    # inside a Review of Systems, which is the property that makes an entry
    # safe to add. See "What extending this table is worth" in the module
    # docstring -- on this corpus it measured as exactly nothing, and these are
    # kept for notes laid out differently rather than for a number.
    ("service",): "administrative",
    ("location",): "administrative",
    ("attending", "provider"): "administrative",
    ("disp",): "disposition",
    ("marital", "status"): "social_history",
    ("assessment", "/", "plan"): "assessment",
}

_ALL_SPELLINGS: dict[tuple[str, ...], str] = {**_SECTION_SPELLINGS, **_OTHER_SPELLINGS}

#: Canonical names of the sections whose body is a negative checklist. Only
#: these uncap a negation scope.
NEGATIVE_FINDINGS_SECTIONS = frozenset(_SECTION_SPELLINGS.values())

#: Canonical names of every other section this module recognizes. Recognizing
#: one closes a negative-findings section and does nothing else.
OTHER_SECTIONS = frozenset(_OTHER_SPELLINGS.values()) - NEGATIVE_FINDINGS_SECTIONS

#: Longest header phrase, so lookup can stop early.
_MAX_HEADER_TOKENS = max(len(phrase) for phrase in _ALL_SPELLINGS)

#: A list marker or Markdown heading marker standing in front of a header:
#: "1.", "1)", "(1)", "A.", "-", "*", "#", "##". Real notes number their
#: sections, and an unstripped "1." makes the header unrecognizable -- the
#: phrase lookup is anchored at the start of the window, so *any* leading
#: token defeats it regardless of how wide the window is.
#:
#: Bare punctuation and a bare enumerator are both included because the marker
#: does not always arrive as one token, and the two tokenizers in this project
#: disagree about where it splits: spaCy gives ``['1', '.']`` but keeps "A."
#: whole, while ``matcher.tokenize`` gives ``['A', '.']`` and ``['(', '1', ')']``.
#: Each piece therefore has to be skippable on its own.
_LIST_MARKER = re.compile(
    r"^(?:[-*+#•>.()\[\]]+|\(?[0-9]{1,2}[.):]?|\(?[a-z][.):]?)$", re.I
)

#: How many leading marker tokens to skip. "( 1 )" is three; nothing legitimate
#: needs more, and an unbounded skip would walk into the header itself.
_MAX_MARKERS = 3


def _strip_markers(tokens: Sequence[Token]) -> Sequence[Token]:
    """Drop leading list/Markdown markers so the header lands at index 0."""
    i = 0
    while i < min(_MAX_MARKERS, len(tokens)) and _LIST_MARKER.match(tokens[i].text):
        i += 1
    return tokens[i:] if i else tokens


def _detached(
    tokens: Sequence[Token],
) -> tuple[list[str], list[str], list[Token]]:
    """Leading ``(norms, texts, owners)`` with a trailing ``:`` split out.

    Whether the colon arrives attached ("Systems:") or standalone depends on
    how spaCy tokenized the surrounding text, and both spellings occur in these
    notes. ``tokenizer._split_on_label_colons`` copes by testing
    ``endswith(":")``; normalizing here lets the phrase lookup ignore the
    difference entirely.

    The parallel ``texts`` list exists because the no-colon header rule below
    has to inspect the *original* capitalization of the token following the
    phrase, which the casefolded norm has thrown away.

    ``owners`` is the token each element came from, so a caller holding an index
    into ``norms`` can recover the character offset it corresponds to. Splitting
    a colon makes ``norms`` longer than the token list, and every attempt to
    recover the header's extent by treating the two as parallel gets it wrong on
    exactly the windows where the colon arrived attached.
    """
    norms: list[str] = []
    texts: list[str] = []
    owners: list[Token] = []
    for tok in tokens[: _MAX_HEADER_TOKENS + 1]:
        norm = tok.norm
        if len(norm) > 1 and norm.endswith(":"):
            norms.append(norm[:-1])
            texts.append(tok.text[:-1])
            owners.append(tok)
            norms.append(":")
            texts.append(":")
            owners.append(tok)
        else:
            norms.append(norm)
            texts.append(tok.text)
            owners.append(tok)
    return norms, texts, owners


def _starts_new_sentence(text: str) -> bool:
    """True when `text` looks like the start of the header's *content*.

    Same test ``tokenizer._split_on_label_colons`` uses to decide a label ended:
    an uppercase or numeric first character. It is what separates the header
    "IMPRESSION Patient is alert" -- which spaCy hands over as one window
    because a bare header has no sentence-final punctuation -- from the
    ordinary sentence "Plan to discharge tomorrow", where "to" is lowercase and
    "Plan" is just the subject.
    """
    return bool(text) and (text[0].isupper() or text[0].isdigit())


def section_header(tokens: Sequence[Token]) -> str | None:
    """Canonical section name when `tokens` *opens* with any recognized header.

    Covers both :data:`NEGATIVE_FINDINGS_SECTIONS` and :data:`OTHER_SECTIONS`;
    callers that only care about the former should use
    :func:`negative_findings_section`.

    A header is a known phrase at the start of the window -- after any list or
    Markdown marker -- followed either by ``:`` or by a token that starts new
    content. Requiring one of those two is what keeps this from firing on prose
    that merely mentions the words ("allergies to penicillin were discussed",
    "plan to discharge tomorrow").

    Returns ``None`` for any window that is not such a header, including one
    that sits *inside* a section -- tracking that is the caller's job, since it
    spans windows. See ``ClinicalPipeline.analyze``.
    """
    found = section_header_span(tokens)
    return found[0] if found else None


def section_header_span(tokens: Sequence[Token]) -> tuple[str, int] | None:
    """:func:`section_header`, plus the character offset the header ends at.

    The offset is where the header stops and its *content* begins, including a
    trailing colon when there is one. A mention lying entirely before it is part
    of the heading rather than part of the note -- which is what
    ``ClinicalPipeline(drop_header_mentions=True)`` uses to suppress it.

    Separate from :func:`section_header` because almost every caller wants only
    the name, and returning a tuple everywhere would make the common call read
    worse for one uncommon need.
    """
    norms, texts, owners = _detached(_strip_markers(tokens))
    for length in range(min(len(norms) - 1, _MAX_HEADER_TOKENS), 0, -1):
        section = _ALL_SPELLINGS.get(tuple(norms[:length]))
        if section is None:
            continue
        if norms[length] == ":":
            # The colon belongs to the header.
            return section, owners[length].end
        if _starts_new_sentence(texts[length]):
            return section, owners[length - 1].end
    return None


def negative_findings_section(tokens: Sequence[Token]) -> str | None:
    """Canonical name when `tokens` opens a section whose body is negatives.

    The :func:`section_header` result, filtered to
    :data:`NEGATIVE_FINDINGS_SECTIONS`. ``None`` for every other header, which
    is what stops Physical Exam or Impression from uncapping a scope.
    """
    section = section_header(tokens)
    return section if section in NEGATIVE_FINDINGS_SECTIONS else None


def track_sections(
    windows: Iterable[Sequence[Token]],
) -> Iterator[tuple[Sequence[Token], str | None]]:
    """Pair each window with the section it sits in, carrying state across windows.

    Section state spans windows, so something has to hold it, and it is held
    here rather than in ``ClinicalPipeline.analyze``: the evaluation tools
    re-run the same loop to explain a decision, and a second copy of this state
    machine is a second
    chance for the two to disagree about where a section ended. One of them
    deciding an attribute while the other explains it would make the
    explanation worthless, which is the same argument
    :func:`umlsmatch.assertion.negation._trigger_scopes` makes for sharing its
    scope computation.

    Yields ``(tokens, section)`` for every window, including header-only ones:
    a header window still carries concepts -- 19 of them across the earlier
    corpus -- and skipping it would silently drop them. The section reported
    for a header window is the section that header *opens*.
    """
    section: str | None = None
    for tokens in windows:
        if not tokens:
            continue
        header = section_header(tokens)
        if header is not None:
            # Any recognized header sets the state to whatever *that* section
            # implies, so a section always ends at the next one. See the module
            # docstring on why a much longer list closes a section than opens
            # one.
            section = header
        elif is_header_only(tokens):
            # An *unrecognized* header still closes the section.
            section = None
        yield tokens, section


def is_header_only(tokens: Sequence[Token]) -> bool:
    """True when `tokens` ends at its colon, carrying no content after it.

    ``tokenizer.annotate_sentences`` splits "Review of Systems: Negative for
    fever" into two windows, so a header usually arrives alone -- but only when
    the following token is capitalized or numeric, which is the splitter's rule
    and not a guarantee. When it arrives inline instead, the header window also
    holds the content that the section rule has to apply to, and the caller
    must not skip it.

    Still used as a fallback close for *unrecognized* headers: a window that is
    nothing but "Wound Care:" ends the previous section even though no entry in
    :data:`OTHER_SECTIONS` names it.
    """
    return bool(tokens) and tokens[-1].norm.endswith(":")

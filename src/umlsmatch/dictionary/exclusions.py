"""Build-time term-text exclusions, ported from cTAKES' dictionary-creation tool.

cTAKES' *runtime* dictionary lookup (``DefaultJCasTermAnnotator`` /
``DefaultTermConsumer``) applies no semantic-group or TTY restriction at
all: every one of the 17 ``SemanticGroup`` values produces a typed mention
unconditionally if the dictionary contains a matching CUI (verified by
reading ``AbstractJCasTermAnnotator.java`` / ``DefaultTermConsumer.java`` /
``SemanticGroup.java`` in the ``ctakes-java`` clone). "Restrict to the
groups/TTYs cTAKES actually covers" is therefore not a lever that exists in
the runtime code -- there is nothing there to restrict to.

The only exclusion mechanisms found anywhere in the cTAKES source live in
its *build-time* dictionary-creation GUI tool (``ctakes-gui``), which
``tools/build_dictionary.py`` does not otherwise use. Two exact-text
blacklists from that tool are mechanically ported here:

  * ``BAD_POS_TERM_SET`` -- verbatim from
    ``ctakes-gui/src/main/java/org/apache/ctakes/gui/dictionary/util/RareWordUtil.java``
    lines 32-75. A hardcoded function-word list (be/have/do, cardinals,
    conjunctions, determiners, pronouns, modals, prepositions, wh-words, a
    few honorifics). Upstream uses it for two things: picking which token of
    a multi-word term anchors its rare-word index entry
    (``isRarableToken()``), and rejecting a term outright when its *entire*
    text is exactly one of these words (``UmlsTermUtil.isTextValid()``).
    Only the second behavior applies to a whole-term exclusion list; ported
    here as a flat set since this project's matcher already handles anchor
    eligibility itself via POS tags (see ``dictionary.matcher.DEFAULT_EXCLUSION_TAGS``).
  * ``UNWANTED_TEXTS`` -- verbatim from
    ``ctakes-gui/src/main/resources/org/apache/ctakes/gui/dictionary/data/tiny/UnwantedTexts.txt``.
    A small hand-curated exact-text blacklist of ambiguous short terms
    (brand names, abbreviations that collide with common words -- e.g.
    "date" is a UMLS synonym for an allergenic-extract concept, not a
    calendar date). Provenance caveat: only the "tiny" GUI data profile
    ships this file in this repo (the "default"/"small"/"tim" profiles do
    not) -- it is not confirmed to be what built the actual shipped
    ``sno_rx_16ab`` (2016AB) dictionary, just the only evidence of this
    exact curation found in the cTAKES source.

Empirically (see ``tools/diff_parity.py`` against real cTAKES output),
these two lists explain 2 of the ~15 most frequent false-positive CUIs
found in a real-note corpus ("past" via ``BAD_POS_TERM_SET``, "date" via
``UNWANTED_TEXTS``). They do NOT explain the bulk of that false-positive
volume (tablet, mouth, patient, family, surgery, normal, oral, daily, day,
yr).

**The rest of that volume is accounted for elsewhere.** The curation that
explains it is recoverable: it ships with every cTAKES install as an
HSQLDB ``.script`` file, and importing it
(``tools/import_ctakes_dictionary.py``) shows it to be a **49-TUI
whitelist** out of the ~127 UMLS semantic types, now captured in
``umlsmatch.umls.ctakes_tuis`` and applied by ``build_dictionary.py``'s
default ``--tui-set ctakes``. Every one of the false positives listed above
belongs to a semantic group cTAKES never indexed (ENTITY, SUBJECT,
MODIFIER, ...). The filter was always real; it lived in the shipped data
rather than in the source code.

So the two lists here remain correct but minor: they are a build-time
text filter, not the mechanism that explains cTAKES' precision.
"""

from __future__ import annotations

__all__ = [
    "BAD_POS_TERM_SET",
    "EXCLUDED_TERM_TEXTS",
    "RARE_WORD_BAD_POS_TERMS",
    "UNWANTED_TEXTS",
]

# Verbatim from RareWordUtil.java lines 32-75 (org.apache.ctakes.gui.dictionary.util).
#
# `noqa: B033` -- "to", "which" and "that" each appear twice, because the Java
# lists them under more than one part-of-speech heading ("to" under both PP and
# TO; "which" under WDT and WP; "that" under DT and WP). The duplicates are
# harmless in a set and are kept so this block stays a line-for-line diff
# against the Java source, which is the whole point of the module. De-duplicating
# would make the next upstream comparison harder for no behavioural gain.
BAD_POS_TERM_SET: frozenset[str] = frozenset(
    {
        # VB verb
        "be", "has", "have", "had", "do", "does", "did", "is", "isn", "am", "are", "was", "were",
        # CD cardinal number
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        # CC coordinating conjunction
        "and", "or", "but", "for", "nor", "so", "yet", "while", "because",
        # DT determiner
        "this", "that", "these", "those", "the", "an", "a",
        # EX existential there
        "there",
        # MD modal
        "can", "should", "will", "may", "shall", "might", "must", "could", "would",
        # PDT predeterminer
        "some", "many", "any", "each", "all", "few", "most", "both", "half", "none", "twice",
        # PP prepositional phrase (preposition)
        "at", "before", "after", "behind", "beneath", "beside", "between", "into", "through",
        "across", "of", "concerning", "like", "unlike", "except", "with", "within", "without",
        "toward", "to", "past", "against", "during", "until", "throughout", "below", "besides",
        "beyond", "from", "inside", "near", "outside", "since", "upon",
        # PP$ possessive personal pronoun (Brown tag, not Penn Treebank)
        "my", "our", "your", "her", "their", "whose",
        # PRP personal pronoun, plurals added
        "i", "you", "he", "she", "it", "them", "they", "we", "us",
        # PRP$ possessive pronoun
        "mine", "yours", "his", "hers", "its", "ours", "theirs",
        # RP particle (contains some prepositions)
        "about", "off", "up", "along", "away", "back", "by", "down", "forward", "in", "on", "out",
        "over", "around", "under",
        # TO to (also a preposition)
        "to",  # noqa: B033 - deliberate duplicate; see the header note
        # WDT wh-determiner
        "what", "whatever", "which", "whichever",
        # WP, WPS wh-pronoun, nominative wh-pronoun
        # "which"/"that" repeat WDT/DT above -- see the header note.
        "who", "whom", "which", "that", "whoever", "whomever",  # noqa: B033
        # WRB
        "how", "where", "when", "however", "wherever", "whenever",
        # "Mine" per the Java comment -- correlative conjunctions etc.
        "no", "not", "oh", "mr", "mrs", "miss", "dr", "as", "only", "also", "either", "neither",
        "whether",
        # additional numbers
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion",
    }
)

# Verbatim from ctakes-gui/.../data/tiny/UnwantedTexts.txt (comments/blank lines stripped).
UNWANTED_TEXTS: frozenset[str] = frozenset(
    {
        "today", "tomorrow", "active", "rid", "tandem", "amen", "revive", "at 10", "perform",
        "level", "met", "men", "add", "got", "nos", "fed", "ever", "ea", "same", "gt", "lt", "rt",
        "dry", "acid", "top", "let", "hist", "date", "aim", "pam",
    }
)

#: Exact term-text exclusions applied at dictionary build time: a candidate
#: term whose *entire* normalized text matches an entry here is dropped
#: outright, mirroring ``UmlsTermUtil.isTextValid()``'s exact-text rejection.
EXCLUDED_TERM_TEXTS: frozenset[str] = BAD_POS_TERM_SET | UNWANTED_TEXTS


# Verbatim from RareWordTermMapCreator.java lines 51-84
# (org.apache.ctakes.dictionary.lookup2.dictionary), preserving the Java's own
# part-of-speech section comments so this stays a line-for-line diff.
#
# **This is a different list from BAD_POS_TERM_SET above**, and the resemblance
# is the trap. That one comes from the *GUI build tool* (RareWordUtil, 181
# terms); this one from the *runtime lookup* package (114 terms) and is a strict
# subset -- it has no forms of be/have/do, no teens or tens, no honorifics, no
# "no"/"not". They are used for different jobs and must not be substituted for
# each other: this set decides which token may anchor a term's rare-word index
# entry (``isRarableToken``), so swapping in the larger one would change which
# token indexes a term and silently alter what the dictionary can match.
#
# `noqa: B033` -- "that", "to" and "which" each appear twice, listed under more
# than one part-of-speech heading, exactly as above.
RARE_WORD_BAD_POS_TERMS: frozenset[str] = frozenset(
    {
        # CD  cardinal number
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        # CC  coordinating conjunction
        "and", "or", "but", "for", "nor", "so", "yet",
        # DT  determiner
        "this", "that", "these", "those", "the",
        # EX  existential there
        "there",
        # MD  modal
        "can", "should", "will", "may", "might", "must", "could", "would",
        # PDT  predeterminer
        "some", "any", "all", "both", "half", "none", "twice",
        # PP  prepositional phrase (preposition)
        "at", "before", "after", "behind", "beneath", "beside", "between", "into", "through",
        "across", "of",
        "concerning", "like", "except", "with", "without", "toward", "to", "past", "against",
        "during", "until",
        "throughout", "below", "besides", "beyond", "from", "inside", "near", "outside", "since",
        "upon",
        # PP$  possessive personal pronoun - Brown POS tag, not Penn TreeBank
        "my", "our",
        # PRP  personal pronoun
        "i", "you", "he", "she", "it",
        # PRP$  possesive pronoun
        "mine", "yours", "his", "hers", "its", "ours", "theirs",
        # RP  particle  - this contains some prepositions
        "about", "off", "up", "along", "away", "back", "by", "down", "forward", "in", "on", "out",
        "over", "around", "under",
        # TO  to  - also a preposition
        "to",  # noqa: B033
        # WDT  wh- determiner
        "what", "whatever", "which", "whichever",
        # WP, WPS  wh- pronoun, nominative wh- pronoun
        "who", "whom", "which", "that", "whoever", "whomever",  # noqa: B033
        # WRB
        "how", "where", "when", "however", "wherever", "whenever",
    }
)

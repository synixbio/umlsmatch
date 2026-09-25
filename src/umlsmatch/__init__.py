"""umlsmatch — a Python-native clinical NLP pipeline for concept extraction.

Extracts UMLS concepts from clinical text and assesses four assertion
attributes -- ``negated``, ``subject``, ``history_of`` and ``uncertain`` --
with no JVM and no UIMA.

    from umlsmatch import ClinicalPipeline

    with ClinicalPipeline() as nlp:
        for a in nlp.analyze("Patient denies chest pain."):
            print(a.cui, a.text, a.group, a.negated)

An assertion attribute this pipeline does not assess is ``None``, not
``False``; see :class:`umlsmatch.analyze.Annotation`.

See docs/USER_GUIDE.md to get started.
"""

from __future__ import annotations

#: Single source of truth for the version: pyproject declares it dynamic and
#: hatchling reads it from here, so a release bumps one literal.
__version__ = "0.2.0"

__all__ = ["Annotation", "ClinicalPipeline", "__version__", "find_dictionary"]


def __getattr__(name: str) -> object:
    """Expose the pipeline lazily.

    ``ClinicalPipeline`` needs the optional ``nlp`` extra (spaCy). Importing it
    eagerly would make ``import umlsmatch`` fail for anyone using only the
    dictionary tools, so it is resolved on first access instead.
    """
    if name in ("Annotation", "ClinicalPipeline", "find_dictionary"):
        from umlsmatch import analyze as _analyze

        return getattr(_analyze, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

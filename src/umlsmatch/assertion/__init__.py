"""Assertion attributes (Phase 4).

Four are assessed, each in its own module, and all four are lexicon-and-scope
rules over the shared machinery in :mod:`~umlsmatch.assertion.scope`:

* :mod:`~umlsmatch.assertion.negation`    -- polarity. Plus parse-driven rules
  the other three deliberately do not share; see that module.
* :mod:`~umlsmatch.assertion.subject`     -- patient vs. family member.
* :mod:`~umlsmatch.assertion.history`     -- history-taking context.
* :mod:`~umlsmatch.assertion.uncertainty` -- hedging. A diagnostic only.

A fifth, :mod:`~umlsmatch.assertion.conditional`, is a prototype: same shape,
same machinery, but off unless ``ClinicalPipeline(conditional=True)`` asks for
it, because no measurement of it exists yet.

:mod:`~umlsmatch.assertion.attributes` is the registry that names all six
cTAKES attributes, including the one this package deliberately does not
implement and the one that is not switched on, and
:mod:`~umlsmatch.assertion.sections` supplies the section state the first three
consult.

Everything here runs on the standard library alone. That is a project rule
enforced by ``tests/test_zero_dependency_core.py``, and it is why none of these
attributes is a classifier -- a trained model is deliberately not the
escalation path, and the cost in accuracy is accepted knowingly.
"""

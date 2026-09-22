"""The core package must stay importable with nothing installed.

This is a project rule, not an accident of the current import graph: a batch or
CLI user installs no web stack and no model, `pip install umlsmatch` pulls
nothing, and the dictionary tools stay usable in environments where spaCy will
not build. The rule is easy to break by adding one convenient import, and a
`dependencies` list has already acquired an unused entry once, so it is pinned
here rather than left to review.

Third-party imports are allowed only in modules that are themselves behind an
extra, and only where the failure is an ImportError naming that extra --
:mod:`umlsmatch` and :mod:`umlsmatch.service` both resolve those lazily through
``__getattr__``.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "umlsmatch"
PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Modules permitted to import a third party, and the extra that provides it.
#: Every entry must be unreachable from ``import umlsmatch`` without that extra.
#:
#: Kept to the minimum that actually needs it. The ``eval`` scorers reach spaCy
#: *transitively* through ``pipeline.tokenizer`` and import nothing third-party
#: themselves, so they are deliberately not listed: exempting a module that does
#: not need exempting turns this test into a rubber stamp for that module.
GATED_MODULES = {
    "pipeline/tokenizer.py": "nlp",
    "service/api.py": "service",
    "eval/negspacy_diff.py": "compare",
}


def _third_party_imports(path: pathlib.Path) -> set[str]:
    """Top-level names this module imports that are neither stdlib nor ours."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        found.update(
            n for n in names if n not in sys.stdlib_module_names and n != "umlsmatch"
        )
    return found


def test_core_modules_import_no_third_party():
    """Only the gated modules may touch a third-party package at all."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in GATED_MODULES:
            continue
        imports = _third_party_imports(path)
        if imports:
            offenders[rel] = imports
    assert not offenders, (
        f"third-party imports outside the gated modules: {offenders}. "
        "Either defer the import behind an ImportError naming an extra, or add "
        "the module to GATED_MODULES with the extra that provides it."
    )


def test_importing_umlsmatch_pulls_in_no_third_party():
    """``import umlsmatch`` must not transitively import an extra's package.

    The lazy ``__getattr__`` in ``umlsmatch/__init__.py`` is what makes this
    true; importing ``ClinicalPipeline`` eagerly would break it.
    """
    before = set(sys.modules)
    for name in [m for m in sys.modules if m.startswith("umlsmatch")]:
        del sys.modules[name]
    import umlsmatch  # noqa: F401

    pulled = {
        m.split(".")[0]
        for m in set(sys.modules) - before
        if m.split(".")[0] not in sys.stdlib_module_names
        and not m.startswith("umlsmatch")
    }
    # Anything already imported by another test is unavoidable; only a *new*
    # third-party module appearing here indicates a real eager import.
    assert not {"spacy", "fastapi", "pydantic", "negspacy"} & pulled, (
        f"import umlsmatch eagerly pulled in {pulled}"
    )


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_pyproject_declares_no_core_dependencies():
    """`dependencies` stays empty. Extras are where third parties belong."""
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert deps == [], (
        f"core dependencies must stay empty, found {deps}. "
        "Add it to an extra in [project.optional-dependencies] instead."
    )
    extras = data["project"]["optional-dependencies"]
    for extra in set(GATED_MODULES.values()):
        assert extra in extras, f"gated module references missing extra {extra!r}"

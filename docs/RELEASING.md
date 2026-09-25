# Releasing

How a version of umlsmatch reaches PyPI, what stands in front of that, and what
to do when a release goes wrong.

**Releases are made by CI from a tag, never from a laptop.** There is no PyPI
API token anywhere — not in the repository secrets, not in a `.pypirc`. GitHub
mints a short-lived OpenID Connect credential for one workflow in one
repository and PyPI exchanges it for a single upload. Nothing exists to leak or
rotate, and the artifact that ships is the one CI built from a tagged commit
rather than whatever happened to be in someone's `dist/`.

---

## 1. The routine release

Three steps. Everything else is automatic.

```bash
# 1. Bump the version. This is the only place it lives --
#    [tool.hatch.version] in pyproject.toml reads this file.
#    src/umlsmatch/__init__.py:  __version__ = "0.1.1"

# 2. Commit it, with the CHANGELOG entry for the release.
git commit -am "umlsmatch v0.1.1"

# 3. Tag and push. The tag is what triggers the release.
git tag v0.1.1
git push origin master
git push origin v0.1.1
```

Pushing the tag starts [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which runs two jobs in order:

| job | what it does |
|---|---|
| `test` | calls `ci.yml` — PHI containment, then lint and tests on 3.10 and 3.13 |
| `publish` | builds, checks the tag matches the built version, runs `twine check`, uploads |

`publish` runs only if `test` passes in full. It calls `ci.yml` rather than
restating the checks, so a release cannot clear a weaker gate than an ordinary
push — and so the two cannot drift apart, which is the failure this repository
has already had once with the PHI pattern.

### Watch it, then verify independently

A green workflow is evidence, not proof. Confirm against PyPI itself:

```bash
curl -s https://pypi.org/pypi/umlsmatch/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

And install it the way a user will, in a throwaway environment:

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install umlsmatch
/tmp/check/bin/python -c "import umlsmatch; print(umlsmatch.__version__)"
```

The base install must pull **no third-party packages**. If `pip` drags in spaCy
or FastAPI, an extra has leaked into `dependencies` and the release is wrong
even though it published.

---

## 2. What the guards catch

Both exist because a PyPI version number is spent the moment it is used, and
neither mistake is visible until after that.

**The tag must match the built version.** `git tag v0.2.0` on a tree still
reading `0.1.0` would otherwise publish `0.1.0` — or fail as a duplicate, long
after the tag implied otherwise. The step parses the built wheel filename and
compares:

```
tag=0.2.0 built=0.1.0
::error::tag v0.2.0 does not match built version 0.1.0
```

If you see this, you forgot step 1. Delete the tag, bump `__version__`, commit,
re-tag.

**`twine check` runs before the upload.** A `long_description` PyPI rejects, or
renders as plain text, is only discoverable once the version is gone.

---

## 3. Two things that are not in this repository

Both are browser-only, both are already configured, and a release fails without
them. They are written down here because nothing in the tree records them.

### GitHub — the `pypi` environment

Settings → Environments → `pypi`, restricted to **tags matching `v*`**.

A push to `master`, a feature branch, or a `workflow_dispatch` from a branch
cannot enter that environment and therefore cannot reach the publish step. This
is a tag restriction rather than a required reviewer on purpose: the realistic
accident is publishing from the wrong ref, not a malicious release, and an
approval prompt on every tag is friction that gets clicked through.

### PyPI — the trusted publisher

Your projects → umlsmatch → Publishing:

| field | value |
|---|---|
| Owner | `synixbio` |
| Repository name | `umlsmatch` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

All four are matched **literally** on every upload. Renaming the workflow file,
renaming the repository, or changing the environment name is a rejected
publish, not a silent fallback.

`Environment name` must not be blank. Blank displays as `(Any)` and means PyPI
does not check which environment a run used — it would accept an upload from
any `release.yml` run in the repository, leaving the GitHub-side tag
restriction as the only control. With it set, both ends enforce the same
constraint independently and neither alone can authorize an upload.

**Trusted publishers cannot be edited.** There is no edit control, only Remove
and Add. To change a field, add the replacement first, then remove the old
entry — PyPI allows several publishers on a project, so you are never left
with none.

---

## 4. When a release goes wrong

### The `publish` job fails

Nothing was uploaded — the version number is not spent, and the tag is fine.
Fix the cause and use **Actions → the failed run → Re-run failed jobs**, which
re-runs `publish` alone; the test jobs already passed and do not repeat. No new
tag and no new commit is needed.

The most common cause is the trusted publisher not matching. Open the failed
`publish` job and expand the `Publish` step — the error names which claim
failed to match.

### The release published, and it is wrong

**A published version can never be reused.** Deleting `0.1.1` from PyPI does not
free `0.1.1`; the filenames stay reserved and the fix is `0.1.2` either way.
Deleting the whole project frees the *name* for anyone to register, which is a
supply-chain problem rather than a tidy-up. And deletion does not unpublish:
mirrors, `pip` caches, corporate proxies and anyone who already installed it
still have the files.

So the tool is **yank**, not delete. On the release page, Options → Yank. A
yanked release disappears from normal resolution, so new installs skip it,
while an exact pin still resolves — existing lockfiles keep working. That is
what you want for a bad release. Reserve deletion for something that genuinely
must not exist, such as leaked credentials, and expect to publish a new version
regardless.

---

## 5. Building locally

Only for inspecting an artifact before tagging. Releases do not come from here.

```bash
pip install build twine
python -m build
python -m twine check dist/*

# What actually ships:
unzip -l dist/umlsmatch-*.whl          # LICENSE, NOTICE, py.typed, no __pycache__
tar -tzf dist/umlsmatch-*.tar.gz       # must contain no data/ and no .env
```

`python -m build` writes `dist/*.dist-info/entry_points.txt`, which the PHI
pattern's `\.txt$` clause matches. `tests/test_phi_guard.py` skips `dist/` and
`build/` for that reason — both are gitignored and outside what the gate
governs. Do not remove those entries from the skip set, or building a release
and then running the suite will fail it.

Do not `twine upload` from here. It would work, but it bypasses every gate
above and publishes an artifact no CI run ever saw.

---

## 6. The packaging constraints worth remembering

Two things about this package are easy to undo by accident.

**No direct URL references, ever.** PyPI refuses any distribution whose
metadata contains a PEP 440 direct reference (`name @ https://...`). This is
why the `nlp` extra cannot name `en_core_web_sm` — spaCy models are not on PyPI
— and why `python -m spacy download en_core_web_sm` is a separate documented
step. `allow-direct-references` in `pyproject.toml` only silences hatchling; the
index still rejects the upload. Adding one breaks releases outright.

**The base install stays dependency-free.** `dependencies = []` is deliberate:
spaCy, FastAPI and negspaCy are each an extra, so a batch or CLI user installs
no web stack, and the negspaCy cross-check can never become a runtime
dependency of the rules it audits. The post-release install check in §1 is what
catches a regression here.

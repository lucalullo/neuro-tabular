# Publishing NeuroTabular 0.3.0

NeuroTabular releases are published manually. Never reinitialize the existing
Git repository, move a published tag, or replace published release artifacts.

Before publishing, require a clean source tree and green CI on the exact commit
to be tagged. The configured CI covers Ruff, tests on supported Python versions,
Linux/Windows execution, minimum-PyTorch compatibility, distribution builds, and
installed-artifact smoke tests.

Build from clean source with:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python -m twine check dist/*
```

Install the wheel and sdist in separate clean environments and run binary,
multiclass, and regression smoke tests outside the checkout. Verify package
metadata, version consistency, license files, manifests, and artifact hashes.

The final release identity must agree across `pyproject.toml`, the package
version module, `CITATION.cff`, changelog, release notes, wheel, sdist, and Git
tag. For 0.3.0 the tag is `v0.3.0`.

The publishing workflow is manual-only (`workflow_dispatch`). It requires an
actual tag matching the package version plus an explicit matching confirmation.
TestPyPI/PyPI publication remains a separate maintainer action.

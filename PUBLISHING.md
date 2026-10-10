# NeuroTabular 1.0.0rc1 preparation and future publication

The current candidate is **local and unpublished**. No push, tag, release, PR,
workflow dispatch or upload is authorized by its preparation. Existing public
0.3.1 artifacts and tags remain immutable.

Before any later publication decision, require a clean candidate tree and green
CI on the **exact corrected commit**. The seven-job matrix covers Ruff, Linux
Python 3.10/3.11/3.12, Windows Python 3.12, the PyTorch 2.0 family and distribution
builds. Current status: **REMOTE_CI_PENDING_FOR_CORRECTED_COMMIT**.
Baseline CI cannot be transferred to a changed tree.

Independent user/other-machine pilots remain outstanding and a disclosed soft
risk, not a result created by maintainer-local testing. Physical NVIDIA CUDA is
optional and pending. See [support scope](docs/policies/SUPPORT_SCOPE.md).

Build from clean source with:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python -m twine check dist/*
```

Install wheel, sdist, source and editable source in separate clean environments.
Check pip consistency, metadata/import identity, binary/multiclass/regression,
Pipeline, cross validation and trusted pickle/joblib in fresh processes.
Inspect all archives for private/local material and verify artifact hashes.

Freeze release metadata and source bytes **before** regenerating
SOURCE_MANIFEST.sha256. Preserve its sorted, two-space SHA256 format and file
selection; document legitimate new product files and exclude the manifest
itself and the existing Git/workflow-only exclusions. Verify every entry against
both source and sdist. Never reformat covered files after generating it without
repeating the freeze and affected validation.

Version identity must agree across pyproject.toml, the package version module,
CITATION.cff, changelog, release notes, wheel and sdist. 1.0.0rc1 is a prerelease;
no release date or remote tag is assigned by this local preparation.

A maintainer must separately review evidence, obtain exact-commit CI and decide
whether to publish a prerelease or eventually stable 1.0. The existing manual
publishing workflow and TestPyPI/PyPI actions require separate authorization.
Never reinitialize the repository, move a published tag or replace its artifacts.

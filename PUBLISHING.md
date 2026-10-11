# NeuroTabular 1.0.0 preparation and publication

The local stable candidate assigns version 1.0.0 but creates no remote tag or
release. Published 0.3.1 and 1.0.0rc1 identities are immutable. Before any
publication require a clean final tree and green seven-job CI on the exact
stable commit: Ruff, Linux Python 3.10/3.11/3.12, Windows Python 3.12,
PyTorch 2.0-family compatibility and distribution builds. RC CI is evidence for
RC only; stable-commit remote CI remains pending until independently checked.

Build and validate in isolated environments:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python -m twine check dist/*
```

Separately install wheel, sdist, source and editable source. Check pip, version,
three tasks, Pipeline/CV and trusted same/fresh-process persistence. Inspect
every archive member and artifact digest. Retain exact pinned-runtime evidence
for historical models; do not infer cross-runtime portability.

Freeze metadata/source bytes before generating SOURCE_MANIFEST.sha256. Preserve
its sorted path list and two-space SHA256 format; exclude itself and existing
Git/workflow-only exclusions. Verify every entry in source and sdist. Repeat
affected validation if any covered file changes after freeze.

Independent human/other-machine pilots are outstanding soft risk; macOS has no
validated support claim; physical CUDA remains optional/PENDING. Review support,
security and existing legal/naming notes before a separate publication decision.
No local validation creates professional legal clearance.

A maintainer may subsequently authorize a branch push for exact-commit CI, then
an immutable annotated v1.0.0 tag and stable release with verified assets. Manual
commands belong to the local handoff and are never executed by this preparation.
Package-index publication requires separate authorization. Never overwrite a
published tag/artifact or reuse earlier CI for changed source.

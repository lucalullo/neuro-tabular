# NeuroTabular publishing guide

NeuroTabular 1.0.0 is published as the current stable release. Published tags
and release artifacts are immutable: do not move, overwrite or reuse them for
changed source. The same rule applies to 0.3.1 and 1.0.0rc1.

For any future release, require a clean final tree and green seven-job CI on the
exact release commit: Ruff, Linux Python 3.10/3.11/3.12, Windows Python 3.12,
PyTorch 2.0-family compatibility and distribution builds. CI from an earlier
commit or release does not satisfy that gate.

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
all three supported tasks, Pipeline/CV and trusted same/fresh-process persistence.
Inspect every archive member and artifact digest. Retain exact pinned-runtime
evidence for historical models; do not infer cross-runtime portability.

Freeze metadata/source bytes before generating `SOURCE_MANIFEST.sha256`. Preserve
its sorted path list and two-space SHA256 format; exclude the manifest itself and
existing Git/workflow-only exclusions. Verify every entry in source and sdist.
Repeat affected validation if any covered file changes after freeze.

Independent human/other-machine pilots remain a disclosed soft risk; macOS has
no validated support claim; physical CUDA remains optional/best-effort and
physically unvalidated. Review support, security and existing legal/naming notes
before each publication decision. No local validation creates professional legal
clearance.

For a future release, the maintainer may authorize a release-branch push for
exact-commit CI, then an immutable annotated tag and GitHub release with verified
assets. Package-index publication requires separate authorization. Never force
replace a published tag or artifact.

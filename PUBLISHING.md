# NeuroTabular publishing guide

The published NeuroTabular 1.0.0 is immutable; this tree prepares compatibility patch 1.0.2. Published tags
and release artifacts are immutable: do not move, overwrite or reuse them for
changed source. The same rule applies to 0.3.1 and 1.0.0rc1.

For this candidate require a clean final tree and successful CI on the exact
release commit: Ruff; Linux Python 3.10–3.14; Windows Python 3.11–3.14; the
PyTorch 2.0-family and useful minimum dependency gates; proposed Apple Silicon
macOS coverage; and distribution builds. The expanded matrix is a local proposal
and has not run remotely. Earlier seven-job results do not validate this candidate.
Review Python 3.15 forward compatibility separately while its ordinary PyPI Torch
distribution is unavailable.

Build and validate in isolated environments:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python -m twine check --strict dist/*
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

## Future Trusted Publishing for 1.0.2

The existing manual `.github/workflows/publish.yml` is retained unchanged.
Dispatch it against the immutable `v1.0.2` tag, after reviewing the exact tagged
commit and all required successful CI jobs. Select `target=testpypi` and enter
`confirmation=v1.0.2`. Dispatching from `main` fails the existing tag guard.
The workflow checks that the Git tag, confirmation and package version agree,
then tests and builds from that tag before publishing its new distributions.
It does not consume prebuilt GitHub release assets or accept artifact SHA256
inputs; archive hashes can differ from earlier local builds.

TestPyPI and PyPI use the separate `testpypi` and `pypi` environments. The publish
jobs request `id-token: write` and use the official PyPA action without stored
API tokens. The maintainer must separately configure registry publishers and
environment protections that allow the approved release tag. Local validation
does not establish those settings or prove an actual OIDC exchange.
PyPI publication remains a separately authorized action after TestPyPI review.

Follow the [PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/)
and [TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/).
After separately authorized publication, verify registry file digests and install
in a fresh environment. Resolve dependencies from PyPI first, then install
NeuroTabular alone with `--no-deps --index-url https://test.pypi.org/simple` for
TestPyPI; do not mix registries with `--extra-index-url`.

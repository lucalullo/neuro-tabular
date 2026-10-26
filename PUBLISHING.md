# NeuroTabular publishing process — 2.0.0

Publication requires a separate maintainer authorization. Version 1.0.3 and
all older tags and artifacts remain immutable. The explicit 2.0.0 decision
accepts incomplete native performance qualification as a nonblocking risk;
it does not turn historical measurements into PASS or waive critical-defect,
functional, packaging, exact-commit CI or publication checks.

Before any authorized release action, run Ruff, tests with warnings as errors,
the complete source manifest, package hygiene/RECORD/Twine, clean installations,
same-runtime equivalence and persistence, and exact-content platform checks.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python tools/release_artifacts.py source
python -m build
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python -m twine check --strict dist/*
```

Generate SOURCE_MANIFEST.sha256 from final LF bytes, with sorted safe paths;
exclude itself and private evidence. This candidate includes `.gitignore` so
clean checkouts and sdist can verify private-root rules. Do not stage private
datasets, checkpoints, reports, model files or local environments.

After explicit authorization only, the maintainer can commit on
`release/2.0.0` and push. Verify all expected CI jobs on the complete exact
SHA, including the complete source manifest. The manual publication helper
requires the matching immutable version tag, confirmation and successful push
CI on `release/<exact-version>`. Do not create the official commit or tag before that authorization.

The inherited trusted-publication workflow builds only for TestPyPI. A later
PyPI promotion must consume that successful run's original immutable artifact,
verify its ID/attempt/commit/tag/manifest and approved wheel/sdist SHA256 against
the actual TestPyPI downloads, then upload the same files without rebuilding.
Do not reuse a 1.0.3 run ID or hashes for this candidate. Public actions require the explicit final maintainer authorization and
all preceding gates; no credentials or secrets are requested.

A stable 2.0.0 is a distinct release process: the maintainer-designated final identity,
new manifests/artifacts/tests/CI, then its own staged registry verification.
Never treat a private validation build as published or overwrite an old artifact.

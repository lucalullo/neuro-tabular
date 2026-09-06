# NeuroTabular 0.2.0 release checklist

## Source and version

- [ ] Working tree is clean.
- [ ] `pyproject.toml` reports `0.2.0`.
- [ ] `src/neurotabular/_version.py` reports `0.2.0`.
- [ ] `python -c "import neurotabular; print(neurotabular.__version__)"` prints `0.2.0`.
- [ ] `SOURCE_MANIFEST.sha256` verifies every listed source file.
- [ ] No `.venv`, cache, bytecode, build, or local metadata is committed.

## Verification

- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m pytest -W error`
- [ ] GitHub Actions is green on Python 3.10, 3.11, and 3.12.
- [ ] Windows CI is green.
- [ ] PyTorch 2.0 compatibility job is green.
- [ ] `python -m build`
- [ ] `python -m twine check dist/*`
- [ ] Built wheel installs and smoke-tests outside the source tree.

## Release

- [ ] Commit the exact tested source.
- [ ] Create annotated tag `v0.2.0` from that commit.
- [ ] Confirm GitHub Release title is `NeuroTabular 0.2.0`.
- [ ] Mark the GitHub release as a pre-release while the project remains alpha.
- [ ] Attach wheel/sdist only if they were produced from the tagged commit.
- [ ] Do not rewrite the tag or published artifacts after release.

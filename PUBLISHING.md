# Publishing NeuroTabular 0.2.0

NeuroTabular 0.2.0 is the first public release in the current public history.
The repository should be created from the clean source tree, without virtual
environments, caches, bytecode, previous Git history, or local build artifacts.

## GitHub repository

Create `lucalullo/neuro-tabular` as an empty repository, then initialize the
clean source tree locally:

```bash
git init
git branch -M main
git add .
git commit -m "Release NeuroTabular 0.2.0"
git remote add origin https://github.com/lucalullo/neuro-tabular.git
git push -u origin main
```

Wait for CI to pass before tagging.

## Tag and GitHub pre-release

Create an annotated tag from the exact tested commit:

```bash
git tag -a v0.2.0 -m "NeuroTabular 0.2.0"
git push origin v0.2.0
```

Create a GitHub Release from `v0.2.0`, use `NeuroTabular 0.2.0` as the title,
copy the relevant text from `RELEASE_NOTES.md`, and mark the release as a
**pre-release** while the project remains alpha/pre-1.0.

Existing notebooks can install this exact tag with:

```bash
python -m pip install "git+https://github.com/lucalullo/neuro-tabular.git@v0.2.0"
```

## Package verification

Before any package-index publication:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python -m twine check dist/*
```

The GitHub Actions publishing workflow additionally requires an actual tag whose
name matches the package version and an explicit matching confirmation.

## Version identity

The following must all agree:

- `pyproject.toml`: `0.2.0`;
- `src/neurotabular/_version.py`: `0.2.0`;
- Git tag: `v0.2.0`;
- release title: `NeuroTabular 0.2.0`;
- wheel/sdist metadata: `0.2.0`.

Published tags and artifacts are immutable.

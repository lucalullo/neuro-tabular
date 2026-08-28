# Publishing NeuroTabular

NeuroTabular 0.1.1 is a local release candidate. The official repository is
`https://github.com/lucalullo/neuro-tabular`, its default branch is `main`, and
the immutable 0.1.0 release is tagged `v0.1.0`. Preparing this candidate does
not authorize a push, tag, GitHub release, or package-index publication.

## Maintainer decisions required first

Before any public action, a maintainer must:

1. complete professional naming review and address the material risk in
   `LEGAL.md`;
2. verify repository access, branch protection, and release permissions;
3. review author, maintainer, security-contact, and project metadata;
4. verify a monitored private vulnerability-reporting channel;
5. inspect every source and distribution file; and
6. explicitly authorize the tag, GitHub release, and any package-index target.

No invented contact address or credential is stored in the project.

## Local release verification

Use a clean checkout and environment:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m build
python -m twine check dist/*
```

Install the exact wheel into a second clean environment and run an import plus
synthetic fit/predict smoke test. Verify that `pyproject.toml`,
`src/neurotabular/_version.py`, `CHANGELOG.md`, release notes, and the intended
tag all contain the same version.

Build outputs are temporary verification artifacts. They do not belong in the
source repository.

## Mandatory documentation gate

No release may proceed without reviewing and updating, when applicable:

- `README.md`;
- `docs/API.md`;
- `docs/USAGE.md`;
- `CHANGELOG.md`; and
- release notes.

Every tag freezes the code and documentation for that version. Public behavior
requires matching tests. Performance or architecture changes require
reproducible benchmark evidence.

## Repository and branch protection

Update the existing repository through a reviewed pull request against `main`.
Before committing, inspect staged content for archives, build outputs,
environments, caches, credentials, private paths, and unintended benchmark data.

Protect the default branch with pull requests, required CI checks, conversation
resolution, force-push prevention, and branch-deletion restrictions. Require
review when more than one trusted maintainer exists.

## Immutable release sequence

```text
development -> tests -> version and documentation update -> reviewed merge
-> annotated tag -> release notes -> verified artifacts -> optional package index
```

Never move an official tag, replace a published artifact, or reuse a published
version number.

## Trusted Publishing only

The included publication workflow is manual and requires protected GitHub
environments. It uses OpenID Connect with job-scoped `id-token: write` and no
stored package-index password or API token.

Verify TestPyPI and PyPI Trusted Publisher mappings for repository
`lucalullo/neuro-tabular`, workflow `.github/workflows/publish.yml`, and the
protected environments before use. Require environment reviewers. Test the
exact artifact on TestPyPI before a separately reviewed PyPI action.

The workflow's existence is not authorization to publish.

## Final clean-tree scan

Before a public commit or release, run lint, tests, build checks, metadata
checks, secret scanning, and a recursive review for local paths, credentials,
temporary files, caches, environments, distributions, and generated archives.
Confirm that source, documentation, tag, release notes, and artifacts all match
the intended version.

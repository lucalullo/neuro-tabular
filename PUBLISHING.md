# Publishing NeuroTabular

Publishing is a maintainer-controlled process. Ordinary pushes and CI runs must
never publish a package automatically. Repository creation, release creation,
tagging, and package-index publication require explicit maintainer action.

## Maintainer decisions required before release

Before any release or package-index publication, a maintainer must:

1. complete professional naming review and address the material risk in
   `LEGAL.md`;
2. verify the canonical repository owner and URL;
3. review author, maintainer, security-contact, and project metadata;
4. enable a monitored private vulnerability-reporting channel;
5. inspect every source and distribution file; and
6. explicitly authorize the publication target and release.

No unresolved owner token or invented contact address is stored in the project.
Canonical package metadata URLs should be added only after the repository URL is
final.

## Local release verification

Use a clean checkout and environment:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
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

Initialize or update the canonical repository only from a reviewed clean source
tree. Before public commits and releases, inspect staged content for archives,
build outputs, environments, caches, credentials, private paths, and benchmark
data.

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

Configure TestPyPI and PyPI Trusted Publishers only after the canonical owner,
repository, workflow, and environments exist. Require environment reviewers.
Test the exact artifact on TestPyPI before a separately reviewed PyPI action.

The workflow's existence is not authorization to publish.

## Final clean-tree scan

Before a public commit or release, run lint, tests, build checks, metadata
checks, secret scanning, and a recursive review for local paths, credentials,
temporary files, caches, environments, distributions, and generated archives.
Confirm that source, documentation, tag, release notes, and artifacts all match
the intended version.

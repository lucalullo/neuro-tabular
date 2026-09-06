# Contributing to NeuroTabular

Thank you for considering a contribution. NeuroTabular is experimental, so
focused issues, synthetic reproductions, and measured changes are especially
valuable. Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -W error
python -m build
python -m twine check dist/*
```

## Pull requests

1. Keep each change focused and explain its user or research motivation.
2. Add fast synthetic tests for changed behavior.
3. Update README, API, usage, and changelog content when user-visible behavior
   changes.
4. Run lint, warnings-as-errors tests, build, and metadata checks.
5. Include a reproducible benchmark for architecture, default, preprocessing,
   or performance changes.
6. State compatibility, device coverage, and limitations explicitly.

Do not commit datasets, credentials, local paths, caches, environments, build
outputs, benchmark caches, or generated archives.

Public estimator constructors must remain side-effect-free and cloneable.
Training data must never influence validation preprocessing state. New runtime
dependencies require a necessity, maintenance, license, and size review.

## Original work

Contributions must be original work or material the contributor is authorized
to submit. Do not copy source, tests, or documentation from other projects. By
contributing, you agree that the submission may be distributed under the MIT
License.

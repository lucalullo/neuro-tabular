# Versioning policy

NeuroTabular uses [Semantic Versioning](https://semver.org/) as a practical
reference. The project is experimental and pre-1.0, so minor releases may
intentionally evolve the API or standard architecture.

## Pre-1.0 versions

Versions use `0.MINOR.PATCH`.

- PATCH releases contain compatible bug fixes, documentation updates,
  packaging corrections, and maintenance.
- MINOR releases may add features or intentionally change experimental
  behavior. Examples include multiclass support, regression, a new standard
  architecture, or an internal neural ensemble.

No roadmap item has a promised version or date.

## Immutable release identity

Every release must have one matching version in package metadata, the package
version module, changelog, Git tag, release notes, wheel, and source
distribution. Official tags and published artifacts are immutable.

Every Git tag freezes the code and documentation for that exact version.

## Mandatory documentation rule

No future release may be created without reviewing and, where behavior or
version references require it, updating all of:

- `README.md`;
- `docs/API.md`;
- `docs/USAGE.md`;
- `CHANGELOG.md`; and
- release notes.

Public behavior must not be released without matching tests and documentation.
Architecture or performance changes require reproducible benchmarks against
the current release baseline.

# Versioning policy

NeuroTabular uses [Semantic Versioning](https://semver.org/) as a practical
reference. The current checkout is unpublished 1.0.0rc1. Stable 1.0 approval
is separate; earlier 0.x minor releases may intentionally evolve experimental
API or architecture.

The current local-development rules are in
[the canonical versioning policy](docs/policies/VERSIONING_POLICY.md), with
[deprecation notice](docs/policies/DEPRECATION_POLICY.md) and
[persistence boundaries](docs/policies/PERSISTENCE_POLICY.md). Pre-1.0 evolution
still requires explicit notice and migration review. No version or 1.0 freeze
is created by adoption of these maintenance rules.

## Release candidates

1.0.0rc1 is a prerelease version. Local preparation assigns no public release
date or remote tag. The authorized target hardening is documented as an
intentional pre-1.0 correction; constructor defaults and valid prediction math
remain unchanged. Exact-source CI and a separate maintainer decision are pending.

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

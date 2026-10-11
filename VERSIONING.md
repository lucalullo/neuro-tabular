# Versioning policy

NeuroTabular 1.0.0 establishes the reviewed stable public API scope in
[API_BOUNDARY](docs/API_BOUNDARY.md). Semantic Versioning is the practical
change-classification rule: compatible fixes are patches, compatible additions
are minors, incompatible stable contracts require a major and notice.

The canonical [versioning policy](docs/policies/VERSIONING_POLICY.md),
[deprecation rules](docs/policies/DEPRECATION_POLICY.md) and
[persistence policy](docs/policies/PERSISTENCE_POLICY.md) define the limits.
Experimental and internal items retain their explicit classifications. API
stability does not establish universal serialized-artifact portability.

## Release identity

Package metadata, version module, CITATION, changelog, release notes, artifacts
and intended tag v1.0.0 must identify the same version and source commit.
Published tags/artifacts are immutable. Unpublished same-version builds require
source/artifact hashes. The published RC1 remains a separate immutable identity.

Review README, API, USAGE, changelog and release notes before every release.
Public behavior requires matching tests/documentation. Algorithm/performance
changes require reproducible baseline evidence; this promotion makes neither.
No roadmap version/date is promised. Local preparation is not publication;
the maintainer must obtain exact stable-commit CI and decide separately.

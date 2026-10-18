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
identify the same version. This local candidate is `1.0.3rc2` (the PEP 440
spelling of a patch release candidate); a future approved tag would be
`v1.0.3rc2`. This does not create a tag or authorize publication. Published tags/artifacts are
immutable. Unpublished same-version development builds require source/artifact
hashes. The published RC1 remains a separate immutable historical identity.

Review README, API, USAGE, changelog and release notes before every release.
Public behavior requires matching tests/documentation. Algorithm/performance
changes require reproducible baseline evidence. This maintenance candidate
uses frozen paired performance and exact same-runtime equivalence evidence;
it claims no general predictive accuracy or training improvement.
No roadmap version/date is promised. Every future publication requires a
separate maintainer decision and exact-commit validation appropriate to that
release.

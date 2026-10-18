# Versioning policy

NeuroTabular follows Semantic Versioning for the stable public API described in
docs/API_BOUNDARY.md: compatible fixes are patches, compatible additions minors,
and incompatible contracts require a major version and notice. The canonical
versioning, deprecation and persistence policies retain their scope.

## Release identity

The release version is 1.0.3; the matching immutable tag is v1.0.3.
Package metadata, version module, CITATION, changelog, notes and artifacts must
agree. Published releases and the frozen 1.0.3rc1/1.0.3rc2 evidence remain
separate identities and must never be overwritten or retagged.

This patch retains the RC2 computational implementation. Only its version
diagnostic changes. Source hashes are regenerated after final metadata and
documentation changes. Stable-version CI must certify the exact new commit;
the earlier RC2 CI is evidence for RC2, not a new run of the stable bytes.

The maintainer explicitly waived final professional independent attestation
for 1.0.3. The waiver is a maintainer decision, not third-party certification.
Other technical, TestPyPI and authorization gates remain required. No universal
accuracy, training, peak-RAM or serialized-artifact portability claim is added.
No roadmap version or date is promised.

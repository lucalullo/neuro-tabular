# NeuroTabular 1.0.3 release checklist

- Match package, runtime version, citation, changelog and notes to 1.0.3.
- Preserve RC1/RC2 archives and the 1.0.2 baseline; change no numerical algorithm,
  public API, defaults or dependency bounds.
- Verify LF source and the full source/sdist SHA256 manifest after final edits.
- Pass Ruff, meaningful release-guard tests, package hygiene and strict Twine.
- Build wheel/sdist; verify member allowlists, metadata, RECORD and runtime bytes.
- Install both artifacts in clean environments; check pip, import origin, API,
  sklearn, all three tasks, exact numerical equivalence and fresh persistence.
- Preserve qualified compatibility/performance evidence and its limitations.
  RC2 CI #17 passed all 17 jobs; the stable commit needs its own complete CI.
- Record the maintainer waiver of final professional independent attestation for
  1.0.3 without describing it as independent certification.
- After separate authorization, commit/push the stable branch manually and
  verify every required job against its full commit SHA.
- After separate authorization, create v1.0.3 on that exact successful commit.
  Never move, overwrite or force-push a published tag.
- Verify the registry Trusted Publisher mappings for publish.yml and the exact
  testpypi/pypi environments. OIDC uses no stored API token.
- After separate authorization, dispatch target=testpypi on v1.0.3. Retain
  its immutable run ID, artifact ID, manifest, logs and both file SHA256 values.
- Download the files actually published on TestPyPI, check SHA256, install the
  verified wheel in a clean environment and run the three-task smoke/persistence.
- Only after that verification and a separate approval, dispatch target=pypi
  using the successful TestPyPI run ID and both approved file hashes.
  Promotion must consume the original immutable bundle, never rebuild it.
- Verify the final PyPI file hashes and installation. Preserve receipts.
- Retain documented CUDA, cross-runtime, notebook and performance limitations.
  Existing legal/naming and final maintainer go/no-go remain separate decisions.
- Local preparation performs no commit, push, merge, tag, dispatch or upload.

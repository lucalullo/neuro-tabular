# NeuroTabular 1.0.3rc2 release checklist

- Freeze the exact local source snapshot and preexisting diff. Version, citation,
  changelog, notes and package metadata must agree on `1.0.3rc2`.
- Authenticate the certified Linux/Windows implementation. Reuse unaffected
  certificates explicitly; renew version, installation and persistence checks.
- Pass Ruff check/format, source manifest and same-runtime exact three-task
  version-only comparisons. Retain the historical equivalence evidence.
- Check trusted old and new persistence, including fresh processes. Do not infer
  cross-OS or cross-dependency portability from these tests.
- Review documentation, privacy/security/support/legal consistency and all
  residual issues; no unresolved critical or HIGH supported-use defect.
- Build wheel/sdist, Twine, member hygiene and complete source/sdist manifest.
- Install the new wheel and sdist and check metadata, three tasks and affected
  Pipeline/CV/version contracts. Certify the new wheel natively on Windows.
- Retain the frozen performance comparisons; scope claims to measured inference
  and temporary-array results. No general accuracy, training or peak-RAM gain.
- Verify a clean Git checkout preserves every source-manifest hash under
  different local EOL configurations; retain the Git tree identity.
- Freeze final source and record artifact SHA256 values.
- Obtain final review by another evaluator or a second approved physical host.
  The earlier separate automated audit is not that review.
- Before publication, require a separately authorized immutable release commit,
  exact-release CI and maintainer approval; local dirty-tree preparation is not
  a publishable tag. Retain the proposed expanded CI as unexecuted until run.
- Disclose macOS and physical CUDA as unvalidated and Windows qualification as
  native CPython 3.12 x64 only; no universal superiority or portability claims.
- Review existing legal/naming notes and make a separate maintainer go/no-go.
- Only after explicit authorization publish an immutable new version and assets.
- Local preparation performs no push, tag, release, PR, upload or write API.

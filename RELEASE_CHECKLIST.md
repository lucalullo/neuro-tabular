# NeuroTabular 1.0.0 release checklist

- Verify exact final local commit, version coherence and clean tree.
- Pass fresh full tests, Ruff check/format, target/schema/weight/RNG/API contracts.
- Require 30/30 binary and 4/4 representative frozen exact equivalence.
- Check current and scoped old 0.3.1 trusted persistence, including fresh processes.
- Execute documentation; review privacy/security/support/legal consistency.
- Build wheel/sdist, Twine, member hygiene and complete source/sdist manifest.
- Install wheel/sdist/source/editable separately; test three tasks, Pipeline/CV.
- Verify performance safety and bounded resources; cleanroom build/install.
- Freeze final source and record artifact SHA256 values.
- Obtain exact stable-commit seven-job remote CI before publication; RC CI alone
  does not validate the changed stable tree.
- Disclose absent independent human pilots as soft risk, macOS out of scope and
  physical CUDA optional/PENDING; no superiority or portability claims.
- Review existing legal/naming notes and make a separate maintainer go/no-go.
- Only after authorization publish immutable v1.0.0 and verified assets.
- Local preparation performs no push, tag, release, PR, upload or write API.

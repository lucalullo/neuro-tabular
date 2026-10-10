# NeuroTabular 1.0.0rc1 release notes

**Unpublished local release candidate.** This is preparation for review, not a
stable 1.0 release or publication authorization.

The classifier now enforces one single-output binary/multiclass target contract
for fit, eval_set and score. Fractional/continuous labels, non-finite numeric
labels and unsupported dimensions fail early with ValueError. Integer, string,
boolean and discrete integer-valued floating labels remain supported. Object
containers receive value-preserving scalar inference; numeric strings are not
converted to numbers. See [the target contract](docs/TARGET_CONTRACT.md).

Regression score uses the same finite numeric-target validation as fit/eval_set.
Zero-weight training rows remain excluded before target discovery. Failed
invalid-target refits preserve the previous fitted model and caller RNG state.

This repair changes no constructor default, training algorithm, preprocessing,
checkpoint selection, prediction or probability math. The candidate also
contains earlier unreleased input-schema diagnostics, actionable schema errors,
support examples, packaging hygiene and maintenance policies from the
0.3.1-based stabilization tree; see [the changelog](CHANGELOG.md).

Validation must be associated with the exact source and artifact hashes.
Frozen valid-input checks require 30/30 binary and 4/4 representative
multiclass/regression equivalence. Historical trusted 0.3.1 model checks have a
pinned Linux CPU dependency scope; no universal serialization portability or
migration of previously unsupported-target models is promised.

**REMOTE_CI_PENDING_FOR_CORRECTED_COMMIT.** The old baseline CI is historical
evidence only. Python 3.10–3.12 intent and dependency bounds remain unchanged;
the new commit still needs its own seven-job remote matrix. Native Windows on
the maintainer PC is blocked by Application Control, a separate environment
limitation; no policy bypass is part of this candidate.

Independent user pilots are still unavailable and remain a soft risk. macOS
has no validated support claim. Physical NVIDIA CUDA validation is optional and
pending. No benchmark superiority or new scientific conclusion is claimed.

No tag, push, release, PR or package publication is performed by local
preparation. Review the local reports and exact-commit CI before any later
maintainer go/no-go decision; see [publishing](PUBLISHING.md).

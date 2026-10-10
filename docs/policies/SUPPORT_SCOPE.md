# Support scope for the unpublished 1.0.0rc1 candidate

The CPU Python intent remains **3.10–3.12** (metadata >=3.10,<3.13);
dependency bounds are unchanged. Local RC evidence must be tied to the exact
corrected source. **REMOTE_CI_PENDING_FOR_CORRECTED_COMMIT**.

| Item | Evidence boundary |
| --- | --- |
| Python 3.10 / 3.11 | Intended CPU support; exact RC CI is pending. |
| Python 3.12 / Linux CPU | Maintainer-local WSL Ubuntu 24.04 software validation; record exact dependency versions in validation receipts. This is not an independent-user pilot or universal distro/hardware promise. |
| Windows CPU | Intended support; exact RC native Windows 3.12 CI pending. Maintainer-local imports are blocked by Application Control; WSL does not close native Windows evidence. |
| macOS CPU | OUT_OF_SCOPE: no validated support claim. OS Independent packaging metadata does not imply platform execution. |
| CUDA Linux / Windows | OPTIONAL/BEST_EFFORT. Existing selection/fallback interface retained; physical NVIDIA validation PENDING. |
| PyTorch >=2.0 | Bound unchanged. Exact RC minimum-family CI pending; the configured job tests 2.0.1, not exact 2.0.0 or every pairing. |
| NumPy >=1.24 / pandas >=2.0 / scikit-learn >=1.3 | Bounds unchanged. Local receipts specify tested combinations; untested minimum/all-future combinations are not guaranteed. |

The earlier 0.3.1-based commit had a passing seven-job remote matrix. That result
is historical evidence for its own source only; it is not remote validation of
this corrected RC. Separate clean environments exercise local artifact installs.
Independent users and another real machine remain outstanding evidence and a
disclosed soft risk. No local emulation can supply those observations.

CPU is the stabilization target. CUDA's existing cuda/auto selection and
diagnostics remain available unchanged. Physical CUDA is optional, not a hard
RC blocker. A stronger NVIDIA/platform claim requires actual hardware evidence.

Review the platform/dependency matrix before each planned minor and stable 1.0.
Add support with evidence; do not silently drop advertised scope in a patch.
Planned drops follow notice and at least one intervening minor series; post-1.0
incompatible install/API scope also needs major review. Document urgent
correctness/security exceptions explicitly.

Persistence checks have same-build/pinned CPU scope. Patch compatibility is best
effort; minor/major, dependency, OS and device portability are unguaranteed.
Installing a dependency does not imply serialized-artifact portability. There
is no migration/loader promise for historical unsupported-target artifacts.

DataFrame-only, single-output, comparable classification labels and the
[target contract](../TARGET_CONTRACT.md) remain explicit. No blanket sparse,
ndarray, date, metadata-routing or full sklearn compliance claim; no calibrated
probability/interval or parallel same-process fit guarantee.

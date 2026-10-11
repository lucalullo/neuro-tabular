# Support scope for NeuroTabular 1.0.0

CPU Python support is **3.10–3.12** (metadata >=3.10,<3.13); dependency bounds
are unchanged. The exact published RC1 passed seven CI jobs. Stable source
needs its own exact-commit remote CI before publication.

| Item | Evidence boundary |
| --- | --- |
| Linux CPU / Python 3.10, 3.11 | CI_SUPPORTED on published RC1; stable-commit CI pending. No unavailable local interpreter result is claimed. |
| Linux CPU / Python 3.12 | Maintainer-local Ubuntu 24.04 WSL2 tests in recorded current/pinned CPU environments. No universal distro/hardware or independent-user claim. |
| Windows CPU / Python 3.12 | Published RC1 remote CI PASS and fresh maintainer-local normal native CPU install/tests. Historical Application Control blocking is not a current product defect; no bypass used. Stable-commit CI remains separate. |
| macOS CPU | OUT_OF_SCOPE: no validated execution claim. OS-independent wheel metadata is not platform evidence. |
| CUDA Linux / Windows | BEST_EFFORT, physical NVIDIA validation PENDING. Selection/fallback/error interface retained; mocks do not prove hardware. |
| PyTorch >=2.0 | Bound retained; RC minimum-family CI tests 2.0.1, not exact 2.0.0/every pairing. |
| NumPy >=1.24 / pandas >=2.0 / scikit-learn >=1.3 | Bounds retained; receipts identify tested combinations, not all minimum/future combinations. |

Independent human/other-machine pilots remain absent and a disclosed soft risk.
Local emulation and maintainer testing cannot supply that evidence. CPU is the
stabilization target; physical CUDA remains optional.

Review the matrix before minors/releases. Planned support drops require notice
and at least one intervening minor series; incompatible post-1.0 scope also
requires major review. Document urgent correctness/security exceptions.

Persistence is trusted same-build/pinned CPU scope. Patch compatibility is best
effort; minor/major, dependency, OS and device portability are unguaranteed.
DataFrame-only, single-output, comparable classification labels and the
[target contract](../TARGET_CONTRACT.md) are explicit. No blanket sparse/ndarray,
date, routing, full sklearn, calibrated probabilities/intervals or parallel
same-process fit guarantee is made.

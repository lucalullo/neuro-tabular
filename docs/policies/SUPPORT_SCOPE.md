# Support scope and qualification evidence

NeuroTabular 1.0.3 is the maintenance release derived from the qualified RC2.
The historical 1.0.2 baseline and RC1/RC2 evidence remain distinct. Local stable
validation and exact-stable-commit CI bind the new version and artifacts;
a successful RC2 run is not automatically a run of the stable bytes.
The maintainer waived final professional independent attestation for this release.
That waiver is not independent certification. Registry publication remains staged.

Metadata is `Requires-Python >=3.10`, with no upper bound. First-class
conventional CPython targets are 3.11–3.14. These classifications describe executed
Linux CPU evidence, not every OS, hardware configuration or dependency pairing.

| Item | Evidence boundary |
| --- | --- |
| Linux CPU / CPython 3.11–3.14 | FIRST_CLASS_SUPPORTED in maintainer-local Ubuntu 24.04 WSL2. Ordinary production PyPI dependency installs, full tests with warnings as errors, minimum/current stacks and builds. Exact 3.13.15 included. |
| Linux CPU / CPython 3.10 | LEGACY_COMPATIBLE in its executed current mutually compatible stack. Python 3.10 is upstream EOL and is not preferred. |
| Linux CPU / CPython 3.15 | UPSTREAM_LIMITED: ordinary PyPI lacks Torch wheels. CODE_COMPATIBLE with full tests on the official Torch CPU channel. No one-command PyPI guarantee. |
| Windows CPU / CPython 3.12 x64 | Executed native Windows 11 Sandbox, CPython 3.12.10, pinned dependency stack: 1,174 passed / 12 explained skips before version assignment, plus separate fresh-install automated replay. Short environment/test-cache paths and official Microsoft Visual C++ runtime required. Local host Application Control failures are preserved separately; no protection was disabled. |
| Windows CPU / CPython 3.11, 3.13-3.14 | RC2 GitHub CI #17 passed complete suites on these versions. This is executed hosted-runner evidence, not every Windows/Python/dependency combination; stable-commit CI remains required. |
| macOS CPU | RC2 CI #17 passed the full suite on a GitHub macos-15 / CPython 3.14 runner. This is one executed stack; it does not certify all macOS architectures or older Intel/Torch combinations. |
| CUDA Linux / Windows | BEST_EFFORT, physical NVIDIA validation PENDING. CPU tests, mocks and CUDA dependency installation do not prove hardware operation. |
| NumPy >=1.24 / pandas >=2.0 / sklearn >=1.3 / Torch >=2.0 | Bounds retained. Python-specific binary minima and current stacks are recorded in the compatibility report. NumPy 1.x/2.x and pandas 2.x/3.x are exercised in coherent combinations, not all Cartesian pairings. |
| joblib / Python >=3.12 | Require >=1.5 on 3.12; >=1.4.2 excluding 1.5.0/1.5.1 on 3.13; >=1.5.2 on 3.14; >=1.6 on >=3.15. Covers removed distutils, cached hashing and loky worker/resource tracking. Python 3.10–3.11 retains its working implicit floor. |
| Free-threaded CPython | Not executed or advertised. |

Independent human/other-machine pilots and real Kaggle/Colab sessions are separate
evidence. Local emulation cannot supply them. RC2 CI #17 is preserved as executed
remote evidence; the stable 1.0.3 commit still requires its own exact-commit run.

Python 3.15 users can explicitly install Torch from its official CPU index first,
then NeuroTabular from PyPI after publication. Prefer Python 3.11–3.14 for ordinary
installation. No closure date for the upstream wheel gap is promised.

The executed Python 3.13 / Torch 2.5 minimum stack has a documented upstream
TemporaryDirectory shutdown ResourceWarning under `-W error`, reproduced without
NeuroTabular. Functional checks pass, and current stacks do not show this warning.
The Torch floor is retained; the audit does not label that old stack warning-free.

Review the matrix before releases. Planned support drops require notice and at
least one intervening minor series; incompatible post-1.0 scope requires major
review. Document urgent correctness/security exceptions.

Persistence is trusted same-build/pinned CPU scope. Patch compatibility is best
effort; dependency, Python, OS and device portability are unguaranteed.
DataFrame-only, single-output, comparable classification labels and the
[target contract](../TARGET_CONTRACT.md) are explicit. No blanket sparse/ndarray,
date, metadata routing, complete sklearn, calibrated probabilities/intervals or
parallel same-process fit guarantee is made.

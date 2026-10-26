# Support scope and qualification evidence

NeuroTabular 2.0.0 retains the 1.x estimator API and default numerical behavior.
The table below preserves historical 1.0.2/1.0.3 qualification boundaries; it is
not a claim that all those tests have been repeated on every 2.0.0 platform.
Exact-release local and CI receipts determine current coverage. Automated checks
and maintainer approval must not be described as independent certification.

Before final numerical-input hardening, the frozen 2.0.0 source passed 1,303 software tests with 12 explained
skips in each executed Linux WSL2 and Windows Sandbox environment, with separate
dtype, persistence, equivalence and Python dependency checks. These are functional
results, not dedicated native Linux/Windows performance qualification. No CUDA
hardware or new local macOS validation is claimed. Required 2.0.0 Windows/Linux
CI must pass on the exact final commit before publication.

The maintainer explicitly accepts incomplete native cost qualification as a
nonblocking risk for 2.0.0 only. Historical results and frozen performance gates
remain unchanged and no new training-speed or RAM gain is claimed. Smart App
Control blocked an unsigned pandas extension on one Windows host; that environment
failure is distinct from a general NeuroTabular defect and was not bypassed.

Metadata is `Requires-Python >=3.10`, with no upper bound. First-class
conventional CPython targets are 3.11â€“3.14. These classifications describe executed
Linux CPU evidence, not every OS, hardware configuration or dependency pairing.

| Item | Evidence boundary |
| --- | --- |
| Linux CPU / CPython 3.11â€“3.14 | FIRST_CLASS_SUPPORTED in maintainer-local Ubuntu 24.04 WSL2. Ordinary production PyPI dependency installs, full tests with warnings as errors, minimum/current stacks and builds. Exact 3.13.15 included. |
| Linux CPU / CPython 3.10 | LEGACY_COMPATIBLE in its executed current mutually compatible stack. Python 3.10 is upstream EOL and is not preferred. |
| Linux CPU / CPython 3.15 | UPSTREAM_LIMITED: ordinary PyPI lacks Torch wheels. CODE_COMPATIBLE with full tests on the official Torch CPU channel. No one-command PyPI guarantee. |
| Windows CPU / CPython 3.12 x64 | Executed native Windows 11 Sandbox, CPython 3.12.10, pinned dependency stack: 1,174 passed / 12 explained skips before version assignment, plus separate fresh-install automated replay. Short environment/test-cache paths and official Microsoft Visual C++ runtime required. Local host Application Control failures are preserved separately; no protection was disabled. |
| Windows CPU / CPython 3.11, 3.13â€“3.14 | Expanded CI proposed; not newly executed for this candidate. Native 3.12 success does not certify all Windows/Python/dependency combinations. |
| macOS CPU | NOT_LOCALLY_VALIDATED. Apple Silicon wheel availability audited through 3.14 and CI proposed. Intel Torch wheels stop at 2.2.2 / Python 3.12; old stacks require coherent NumPy ABI versions. |
| CUDA Linux / Windows | BEST_EFFORT, physical NVIDIA validation PENDING. CPU tests, mocks and CUDA dependency installation do not prove hardware operation. |
| NumPy >=1.24 / pandas >=2.0 / sklearn >=1.3 / Torch >=2.0 | Bounds retained. Python-specific binary minima and current stacks are recorded in the compatibility report. NumPy 1.x/2.x and pandas 2.x/3.x are exercised in coherent combinations, not all Cartesian pairings. |
| joblib / Python >=3.12 | Require >=1.5 on 3.12; >=1.4.2 excluding 1.5.0/1.5.1 on 3.13; >=1.5.2 on 3.14; >=1.6 on >=3.15. Covers removed distutils, cached hashing and loky worker/resource tracking. Python 3.10â€“3.11 retains its working implicit floor. |
| Free-threaded CPython | Not executed or advertised. |

Independent human/other-machine pilots and real Kaggle/Colab sessions are separate
evidence. Local emulation cannot supply them. Expanded CI is proposed locally;
this audit does not establish a remote result for the current candidate.

Python 3.15 users can explicitly install Torch from its official CPU index first,
then NeuroTabular from PyPI after publication. Prefer Python 3.11â€“3.14 for ordinary
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

The 2.0 release line retains the tested 1.x API/default contracts. Earlier qualification counts above are historical. Exact 2.0 local/CI receipts and the release readiness matrix determine current executed coverage; do not infer untested platforms or independent certification from a major version.

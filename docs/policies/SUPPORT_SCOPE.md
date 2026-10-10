# P43 support scope decision

2026-10-10. Canonical future product location: docs/policies/SUPPORT_SCOPE.md.
This identical roadmap copy records the local decision; no actual 1.0 release.
Intended CPU Python range remains **3.10–3.12**, current metadata >=3.10,<3.13;
no convenience narrowing or dependency changes. Intent is not remote PASS.

| Item | Current evidence class | Intended 1.0 scope / evidence boundary |
| --- | --- | --- |
| Python 3.10 | NOT_VALIDATED | Intended CPU support; SUPPORTED_VIA_CI only after exact candidate job/minimum-family/install evidence. |
| Python 3.11 | NOT_VALIDATED | Intended CPU support; exact candidate CI evidence pending. |
| Python 3.12 | SUPPORTED | Locally validated 3.12.3 WSL Linux pinned CPU; Windows pairing remains separate. |
| Linux CPU | SUPPORTED | Executed WSL Ubuntu24.04 modern pinned runtime; other matrix combinations conditional on evidence, no universal distro/hardware promise. |
| Windows CPU | NOT_VALIDATED | Intended support, exact-source native Windows3.12 CI + independent installation required; WSL not native evidence. |
| macOS CPU | OUT_OF_SCOPE | Untested, no supported claim; OS Independent packaging classifier does not change this. |
| CUDA Linux | BEST_EFFORT | Optional current device interface; no physically validated support claim. |
| CUDA Windows | BEST_EFFORT | Optional interface, native NVIDIA execution unvalidated. |
| PyTorch >=2.0 declared minimum | NOT_VALIDATED at minimum | Installed 2.14.0+cpu tested; configured minimum-family CI pins2.0.1, not exact2.0.0; lower bound unchanged, no all-future-Torch guarantee. |
| NumPy >=1.24 | SUPPORTED only tested2.5.3; NOT_VALIDATED minimum | Existing lower bound unchanged; configured Torch-min job NumPy<2 is one pairing, not full lower-bound coverage. |
| pandas >=2.0 | SUPPORTED only tested3.0.1; NOT_VALIDATED minimum | Existing bound, exact other combinations need their own evidence. |
| scikit-learn >=1.3 | SUPPORTED only tested1.8.0; NOT_VALIDATED minimum | Tested subset clone/params/Pipeline/CV; general metadata routing experimental. |

No SUPPORTED_VIA_CI result is asserted now. The existing seven-job matrix and
product history justify retaining intent3.10–3.12; this task has no oldest-runtime
or independent dependency-resolution execution. Dependency bounds are installer
constraints, not exhaustive compatibility warranties. Test exact advertised
candidate combinations, record gaps and do not label unresolved combinations PASS.

**CUDA decision B: OPTIONAL/BEST_EFFORT FOR_1_0.** CPU is the supported
stabilization target; existing cuda/auto selection and diagnostics remain
available unchanged. Physical CUDA validation stays PENDING on this AMD PC and
is not a 1.0 hard blocker. Promoting a validated NVIDIA/OS claim requires the
P42 physical procedure on that claimed platform, separately authorized.

Compatibility cadence: review Python/dependency matrix before each planned
minor and each eventual1.0 release. Add support only with evidence; do not drop
advertised support in a patch. Planned drops follow FutureWarning where an
operation can reasonably warn, changelog/install/docs notice and at least one
full intervening minor series; post1.0 incompatible install/API scope also needs
major review. Urgent correctness/security exceptions explicitly documented.
No automatic upper-bound addition or latest-version upgrade is authorized here.

Persistence: same-build/pinned CPU tested, patch best effort, minor/major and
dependency/OS/device changes unguaranteed. Support for installing a dependency
does not imply serialized-artifact portability. No migration/loader promise.
DataFrame-only/single-output, comparable classifier labels, in-memory categorical
costs, no blanket sparse/ndarray/date inputs or full sklearn compliance; no
calibrated probabilities/intervals/parallel same-process fit guarantee.

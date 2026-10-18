# NeuroTabular 1.0.3 maintenance release

Prepared on 2026-10-18 from the qualified 1.0.3rc2. The release preserves the
stable API, defaults, architecture, dependency bounds and predictive algorithms.
Historical 1.0.2 releases and RC1/RC2 artifacts remain immutable.

## Changes and measured scope

Conservative categorical preprocessing, schema validation and numerical-buffer
reuse reduce repeated work. CPU inference avoids unnecessary temporary output
allocations. No persistent cache or new estimator state is introduced.
Distribution hygiene excludes private workspace material. LF source identity and
a complete SHA256 manifest support reproducible Git checkouts.

The frozen paired Linux CPU comparison against original 1.0.2 measured a
14.14% median paired reduction for a 1,200-row multiclass predict_proba workload:
six of six pairs were faster, with a median paired decrease of 0.737 ms.
This is a local workload result, not a universal inference speedup.
Medium/large diagnostic arrays used about 33-48% fewer temporary array bytes;
a small-input case used about 1.9 KiB more. No general process peak-RAM reduction
is demonstrated.

Complete-fit workloads did not meet the frozen general training improvement
criterion. No general training acceleration, accuracy gain or new research
architecture is claimed. Same-runtime predictions, probabilities and trained
numerical state remain equivalent in the qualified comparisons.

## Qualification and release process

RC2 commit d969836a63b0bacacc19ed8f1f04ee75581144ff passed CI #17:
17 jobs, including Linux Python 3.10-3.14, Windows 3.11-3.14, macOS 15 /
Python 3.14, PyTorch 2.0-family and useful binary dependency minima, Ruff and
package builds/install-smokes. Full-suite jobs reported 1,174 passed and
12 motivated skips; four minimum-dependency jobs reported 110 passed / 2 skips.
These are RC2 results, not new executions of the stable version.

The stable 1.0.3 version has its own source/artifact hashes and qualification.
Its exact new commit must pass CI before the immutable v1.0.3 tag is authorized.
Publication is staged: TestPyPI, actual-file SHA256 download and clean-install
verification, then separately authorized PyPI promotion of the same wheel/sdist.
The promotion workflow does not rebuild those distributions.

The maintainer explicitly waived final professional independent attestation for
this release. This is a maintainer risk decision, not independent certification.
Technical evidence, limitations and original protocols remain preserved.

The computational implementation was also qualified in a nine-lane Linux
matrix and native Windows 11 / CPython 3.12.10 x64 before version assignment;
a separate fresh-install automated replay passed. Historical counts are not
presented as runs of newly versioned artifacts. No protection was disabled.
Trusted persistence is same-runtime; cross-OS, dependency, Python and device
portability is not guaranteed. Physical CUDA and real notebook sessions remain
outside the executed evidence.

## Historical 1.0.2 compatibility patch audit

The compatibility patch removed `Requires-Python <3.13`, which prevented installation
on an actual Kaggle CPython 3.13.15 notebook. Proposed metadata is `>=3.10`.
Conventional CPython 3.11–3.14 is the preferred scope; 3.10 remains legacy compatible.
These paragraphs describe the historical compatibility audit; they do not
certify later maintenance changes by themselves.

Linux CPU validation covers ordinary PyPI dependency installation, full tests with
warnings as errors, binary/multiclass/regression, mixed pandas dtypes, sklearn
Pipeline/CV/clone/tags, persistence and builds. Useful minimum and current mutually
compatible dependency stacks are tested. Exact 3.13.15 is reproduced on Ubuntu 24.04
WSL2, not inside Kaggle. Notebook handoffs require a real post-publication check.

Python 3.15 passes full code tests using the official PyTorch CPU index. Standard
PyPI has no compatible Torch distribution at this audit snapshot, classified as
UPSTREAM_DISTRIBUTION_LIMITATION. No arbitrary Python upper bound is added.
Older Torch/NumPy ABI combinations need coherent versions; the metadata is not a
promise that every independent dependency version pairing works.

The Python 3.13 minimum stack with Torch 2.5 also emits an upstream
TemporaryDirectory ResourceWarning during process shutdown under `-W error`.
It is reproduced without importing NeuroTabular. Functional checks pass; the
current Torch stack has no such observed shutdown warning. The audit retains
this warning explicitly rather than treating exit status alone as warning-free.

At the historical compatibility audit, runtime implementation bytes matched
public 1.0.1 except the version constant. The present performance changes are
qualified separately. The original four runtime bounds are retained. Python-specific joblib
bounds prevent stdlib distutils, cached-Pipeline hashing and parallel resource
tracker failures: >=1.5 on 3.12; >=1.4.2 excluding 1.5.0/1.5.1 on 3.13;
>=1.5.2 on 3.14; >=1.6 on >=3.15. Cached Pipeline and parallel CV have
permanent regression tests. API,
architecture, training, predictive defaults and valid-input behavior are unchanged.
Frozen 30/30 binary and 4/4 representative
comparisons pass bit for bit in the previously required pinned CPU runtime.

Windows portable interpreters and native dependency imports encounter Application
Control restrictions locally; these are retained as environment failures.
RC2 CI #17 subsequently executed Windows Python 3.11-3.14 and macOS 15 /
Python 3.14 successfully. These historical results are not runs of the new stable commit. Physical NVIDIA CUDA remains best effort with
validation pending. Free-threaded Python is not advertised.

Same-runtime trusted persistence does not guarantee dependency, Python, OS or device
portability. This patch makes no SOTA or universal superiority claim. Exact-commit
stable-commit CI, optional independent-user checks and package publication remain
separate maintainer decisions.
Historical published tags/releases remain immutable.

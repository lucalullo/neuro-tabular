# NeuroTabular 1.0.3rc2 — local maintenance release candidate

Prepared on 2026-10-18; unpublished. The historical 1.0.2 release and its
artifacts remain immutable. This patch prerelease retains the stable API,
defaults, architecture, dependency bounds and predictive algorithms.

## RC2 source identity

This candidate supersedes the frozen, unpublished 1.0.3rc1 for local review.
Five Markdown documents now use LF as required by Git attributes, and source
hashes are regenerated from repository bytes. RC1 artifacts and evidence remain
immutable. RC2 changes no computational implementation, dependency bounds or
performance claims; only the version diagnostic changes. Independent review
and exact-commit CI remain open gates.

## Changes and measured scope

Conservative categorical preprocessing, schema validation and numerical-buffer
reuse reduce repeated work. CPU inference avoids unnecessary temporary output
allocations. No persistent cache or new estimator state is introduced.
Distribution hygiene explicitly excludes private workspace material.

The frozen paired Linux CPU comparison against original 1.0.2 measured a
14.14% median paired reduction for a 1,200-row multiclass `predict_proba`
workload: six of six pairs were faster, with a median paired decrease of
0.737 ms. This is a local workload result, not a universal inference speedup.
Medium/large diagnostic array workloads used about 33–48% fewer temporary
array bytes; a small-input case used about 1.9 KiB more. These measurements
do not demonstrate a general reduction in process peak RAM.

The complete-fit workloads did not meet the frozen general training improvement
criterion. No general training acceleration, accuracy gain or ensemble/research
architecture is claimed. Qualified same-runtime comparisons preserve exact
predictions, probabilities and trained numerical state.

## Qualification and remaining review

Before version assignment, the identical computational implementation passed
the nine-lane Linux CPU matrix (CPython 3.10–3.14; coherent minimum/current
dependency stacks) and a native Windows 11 / CPython 3.12.10 x64 run with
1,174 passed tests and 12 explained skips. A separate fresh-install automated
Windows replay also passed; it is not an independent human review.
The local maintainer packet binds the newly versioned source, wheel and sdist
to renewed version, packaging, installation and numerical-equivalence receipts.
Historical test counts are not presented as new runs of the versioned artifacts.

Independent final review by another evaluator or a second approved physical
host remains a separate prepublication gate. No GitHub or registry publication
has been authorized. Broader Windows/Python CI, macOS and physical CUDA have
the limits documented in [support scope](docs/policies/SUPPORT_SCOPE.md).
Windows qualification used short environment/cache paths and the official
Microsoft Visual C++ runtime; it required no security-policy changes.

Trusted persistence is tested within pinned runtimes; cross-OS, dependency,
Python and device portability remains unguaranteed.

## Historical 1.0.2 compatibility patch audit

The compatibility patch removed `Requires-Python <3.13`, which prevented installation
on an actual Kaggle CPython 3.13.15 notebook. Proposed metadata is `>=3.10`.
Conventional CPython 3.11–3.14 is the preferred scope; 3.10 remains legacy compatible.
These paragraphs describe the historical compatibility audit; they do not
certify the current unreleased performance changes.

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
Expanded Windows CI is proposed and unexecuted. macOS is NOT_LOCALLY_VALIDATED;
Apple Silicon CI is proposed. Physical NVIDIA CUDA remains best effort with
validation pending. Free-threaded Python is not advertised.

Same-runtime trusted persistence does not guarantee dependency, Python, OS or device
portability. This patch makes no SOTA or universal superiority claim. Exact-commit
remote CI, independent-user checks and package publication remain maintainer gates.
Historical published tags/releases remain immutable.

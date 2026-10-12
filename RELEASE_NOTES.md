# NeuroTabular 1.0.2 — compatibility patch candidate

This local candidate removes `Requires-Python <3.13`, which prevented installation
on an actual Kaggle CPython 3.13.15 notebook. Proposed metadata is `>=3.10`.
Conventional CPython 3.11–3.14 is the preferred scope; 3.10 remains legacy compatible.
The candidate has not been published by this audit.

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

Runtime implementation bytes are unchanged from public 1.0.1 except the version
constant. The original four runtime bounds are retained. Python-specific joblib
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

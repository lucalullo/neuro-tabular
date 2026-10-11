# NeuroTabular 1.0.0

The first stable 1.0 release stabilizes a deliberately compact public API for
binary classification, multiclass classification and single-output regression
on heterogeneous pandas DataFrames. Numerical and categorical preprocessing
remains inside each estimator and is learned only from training data.

The release includes sklearn clone, Pipeline and cross-validation workflows;
transactional refits; caller Python/NumPy/PyTorch CPU RNG restoration; sample
weights; trusted pickle/joblib persistence; and the product diagnostics,
support guidance and distribution hardening developed since 0.3.1.

The pre-1.0 target repair is retained: fit, eval_set and score reject fractional
classification labels, nonfinite targets and unsupported target dimensions.
Discrete integer, boolean, string and integer-valued floating classification
labels remain supported; regression uses finite real numeric targets and
predicts in original units. Invalid refits preserve the previous fitted model.

Promotion from 1.0.0rc1 changes release metadata and documentation only. It adds
no feature, algorithm, predictive default or dependency-bound change. Frozen
30-case binary and four representative checks must remain exact bit-for-bit.

Python 3.10–3.12 is the declared scope. The published RC passed Linux and
Windows CI, including the PyTorch 2.0 family (2.0.1), and fresh local Linux/
Windows CPU checks. The exact stable 1.0.0 commit also passed the seven-job
remote CI matrix before publication; RC CI is retained as separate historical
evidence rather than relabeled as stable-commit evidence.

Limitations remain explicit: DataFrame-only, single-output, in-memory workloads;
no full sklearn check_estimator or automatic external eval_set transformation;
no probability calibration/interval guarantee; and no cross-version, dependency,
OS or device pickle portability promise. Historical 0.3.1 fixture compatibility
is restricted to its tested pinned Linux CPU runtime. Non-scalar numerical
representations, feature gating, detailed diagnostics and routing remain
experimental as listed in [API_BOUNDARY](docs/API_BOUNDARY.md).

Physical NVIDIA CUDA validation remains PENDING / BEST_EFFORT. macOS is outside
validated support. Independent human pilots remain unavailable and a disclosed
soft risk. Local synthetic tests are not external-user or production evidence.
There is no SOTA, universal superiority or tree-model superiority claim.

NeuroTabular 1.0.0 was published from the validated stable commit after the
exact-commit CI gate passed. The `v1.0.0` tag and release artifacts are immutable
release identities; future changes belong to later versions.

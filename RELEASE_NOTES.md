# NeuroTabular 0.2.0 release notes

NeuroTabular 0.2.0 is a measured architecture and efficiency update to the
final 0.1.1 baseline. It keeps the compact residual MLP and scikit-learn binary
classifier interface while improving the categorical representation and
making alternative numerical representations explicitly testable.

This file prepares release content. It does not assert that tag `v0.2.0`, a
GitHub release, or a package-index publication already exists.

## Release default

The 0.2.0 default adds one training-only aggregate log-frequency feature for
each categorical column. Missing, unknown, rare, and frequent values retain
separate encoding semantics; rare values use their aggregate bucket count and
unknown values receive zero. Categorical embedding widths now adapt to dataset
size, cardinality, feature count, and a compact bounded memory budget.

Scalar numerical input remains the default. Affine, periodic, and
piecewise-linear quantile representations are available through
`numerical_embedding`, and a lightweight gate is available through
`feature_gating`, but release ablations did not justify enabling them by
default.

The selected configuration is:

```text
numerical_embedding="scalar"
use_category_frequency=True
feature_gating=False
full_data_refit=False
```

## Training and execution

Preprocessing state is fitted only on training rows. Numerical quantile knots,
category counts, adaptive widths, and vocabularies cannot observe validation or
test rows. The categorical frequency transform reuses encoded IDs and a NumPy
lookup rather than mapping the pandas column twice.

Loss-only validation avoids collecting probabilities and targets or computing a
sigmoid when those arrays are not needed. Processed tables remain device
resident, and batching continues to use index tensors.

The final 0.1.1 compatibility behavior is preserved in full:

- automatic CUDA requires a successful synchronized kernel probe;
- incompatible automatic CUDA warns and falls back to CPU;
- explicit CUDA requests raise detailed errors instead of falling back;
- AMP is enabled only on a verified compatible CUDA device;
- CPU/FP32 execution with AMP disabled uses `nullcontext` and never constructs
  `torch.autocast`;
- AdamW does not force an optimizer implementation unsupported by PyTorch 2.0.

## Measured results

The principal paired, warm, interleaved comparison covered seven synthetic
dataset families, two seeds, and three timing repeats:

- mean ROC-AUC: 0.874559 (0.1.1) to 0.878408 (0.2.0), +0.003849;
- median paired fit-time ratio: 0.950, about 4.96% lower;
- median paired prediction-time ratio: 0.986, about 1.41% lower;
- mean selected/run epochs: 14.143 to 13.714.

Two scikit-learn public numerical datasets provided a no-regression holdout:
both versions achieved mean ROC-AUC 0.998113 with identical predictions and
selected epochs. The release host had no CUDA-capable PyTorch build, so VRAM,
GPU throughput, AMP speed, pinned memory, and non-blocking transfer performance
are explicitly unmeasured.

The project brief also contained an external Kaggle reference favoring
LightGBM. Its data and environment were unavailable, so NeuroTabular 0.2.0 does
not claim the referenced 0.945 target or a reproduced tree comparison.

## Ablation decisions

- Category frequency: selected and enabled.
- Adaptive embedding widths: selected and enabled.
- Affine, periodic, and piecewise numerical modes: retained opt-in, not default.
- Input gating: retained opt-in, not default.
- Full-data refit: retained opt-in; about 68% slower median fit and lower mean
  AUC in screening.
- Categorical and embedding dropout: not enabled after tiny AUC regressions.
- Vocabulary cap/hash: research-only; reduced parameters but produced only
  small inconsistent AUC changes.
- `torch.compile`: not enabled; the host lacked a supported C++ compiler and
  the attempt failed after additional cold time.
- Forced foreach/fused AdamW: not enabled because evidence was noisy and not
  portable across the PyTorch 2.0+ range.
- Parameter-efficient neural ensembling: researched but not implemented in
  this release because its cost and API surface lacked supporting ablation.

## Verification

The release candidate gate includes:

- warnings-as-errors pytest for the final 0.1.1 baseline;
- warnings-as-errors pytest for the 0.2.0 candidate;
- Ruff lint and format check;
- Python bytecode compilation;
- source distribution and wheel build;
- Twine metadata/README validation;
- clean-wheel import and fit/predict smoke test;
- source and archive scans for caches, environments, build output, secrets, and
  local benchmark data;
- a final content fingerprint confirming the 0.1.1 baseline was not modified.

Exact final counts and artifact SHA-256 are recorded in the external GitHub
update report produced with the release archive.

## Compatibility and limitations

NeuroTabular requires Python 3.10+, PyTorch 2.0+, pandas 2.0+, NumPy 1.24+, and
scikit-learn 1.3+. It remains binary-classification-only and
pandas-DataFrame-only. Multiclass, regression, calibration, model persistence,
attention, and automated hyperparameter optimization are not part of 0.2.0.

This is a pre-1.0 minor release, so the new constructor parameters intentionally
evolve the experimental API. Existing 0.1.1 parameter names and core estimator
methods remain available.

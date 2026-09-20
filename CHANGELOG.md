# Changelog

All notable public changes to NeuroTabular are recorded here. Published tags and
release artifacts are immutable.

## [0.3.0] - 2026-09-20

### Added

- Automatic multiclass classification with ordered labels, cross entropy,
  normalized softmax probabilities, weights and loss/accuracy stopping.
- `NeuroTabularRegressor` with train-only weighted target standardization,
  inverse transformation and loss/RMSE/MAE/R² evaluation.
- Shared preprocessing, neural backbone and multi-task training core.
- Pickle/joblib validation in fresh processes for all three tasks.
- Expanded schema, weighting, target, device, sklearn and packaging validation,
  with documentation and frozen real-data benchmark evidence.

### Changed

- Refactor internal estimator/training code while preserving exact v0.2.0 binary
  behavior on the frozen 30-case equivalence panel and retaining public defaults.

## [0.2.0] - 2026-09-06

### First public release

- Add `NeuroTabularClassifier`, a scikit-learn-compatible neural binary
  classifier for heterogeneous pandas DataFrames.
- Add automatic numerical/categorical detection, missing-value handling,
  training-only preprocessing, categorical embeddings, rare/unknown IDs, and
  categorical log-frequency side features.
- Add scalar numerical inputs by default plus explicit affine, periodic, and
  piecewise experimental representations.
- Add adaptive categorical embedding widths, residual MLP backbone, optional
  feature gating, and optional full-data refit.
- Add AdamW, cosine scheduling, automatic batching, internal/external validation,
  early stopping, strict best-checkpoint restoration, sample weights, and
  `class_weight="balanced"`.
- Add transactional re-fitting and restoration of caller Python/NumPy/PyTorch
  CPU RNG states.
- Treat declared-but-unobserved pandas categorical levels as unknown.
- Reject complex values and unsafe float32 conversions explicitly.
- Ignore zero-weight rows before target discovery and preprocessing.
- Add CPU/CUDA diagnostics, synchronized CUDA probing, safe automatic fallback,
  and AMP only on compatible CUDA devices.
- Add Python 3.10-3.12, PyTorch 2.0+, Linux/Windows CI, wheel/sdist verification,
  and clean-wheel smoke tests.
- Add benchmark, performance, ablation, research, usage, API, security,
  contribution, publishing, and release documentation.

### Status

NeuroTabular 0.2.0 is experimental and pre-1.0. No state-of-the-art or universal
performance claim is made.

# Changelog

All notable public changes to NeuroTabular are recorded here. The project uses
the categories Added, Changed, Fixed, Removed, and Documentation when
applicable. Published releases and tags are immutable.

## [0.2.0] - 2026-08-28

### Added

- Add leakage-safe aggregate log-frequency side features for categorical
  columns and enable them in the release default.
- Add dataset-aware categorical embedding widths bounded by cardinality,
  sample count, feature count, and a compact memory budget.
- Add tested scalar, affine, periodic, and piecewise-linear numerical
  representations; scalar remains the default after ablation.
- Add optional lightweight input gating and optional full-data refit after
  internal epoch selection; both remain disabled by default.
- Add numerical training-only quantile knots and missing embeddings required by
  non-scalar numerical modes.
- Add paired synthetic and public-data comparisons, detailed stage profiling,
  compile/optimizer experiments, categorical regularization studies, and
  high-cardinality cap/hash research ablations.
- Add tests for new representations, adaptive embeddings, frequency leakage,
  full-data refit, categorical overflow controls, and CPU autocast regression.

### Changed

- Optimize categorical frequency transform by looking up learned frequencies
  with already encoded category IDs instead of performing a second pandas map.
- Avoid probability accumulation, sigmoid, and host-array construction during
  validation when early stopping uses loss only.
- Expose categorical embedding dimensions as `embedding_dimensions_` and
  expand profiling for preprocessing, training, prediction, and optional refit.
- Set package and distribution metadata to 0.2.0.

### Fixed

- Preserve all final 0.1.1 device and training fixes, including the synchronized
  CUDA probe, correct automatic fallback, explicit CUDA failure diagnostics,
  supported PyTorch optimizer selection, and no CPU `torch.autocast` creation
  when AMP is disabled.
- Keep the package compatible with the declared PyTorch 2.0 minimum and retain
  the final 0.1.1 CUDA fallback regression tests.

### Performance

- Improve paired mean ROC-AUC from 0.874559 to 0.878408 across seven synthetic
  datasets and two seeds. Median paired fit and prediction ratios were 0.950
  and 0.986 respectively; absolute timings remain host-dependent.
- Preserve identical mean ROC-AUC (0.998113) on two numerical scikit-learn
  public datasets used as an external no-regression check.
- Do not enable compile, forced optimizer backends, categorical dropout,
  embedding dropout, feature gating, full-data refit, or high-cardinality
  cap/hash because their release evidence was negative, inconsistent, or not
  portable enough.

### Documentation

- Update README, API, usage, benchmark, performance, ablation, research,
  publishing, release, and GitHub-update documentation for 0.2.0.
- Record CPU-only hardware limits explicitly; no GPU, AMP-throughput, or VRAM
  result is inferred.

## [0.1.1] - 2026-08-28

### Fixed

- Require a successful synchronized CUDA kernel probe before automatic CUDA
  selection, preventing late `no kernel image is available` failures on GPUs
  unsupported by the installed PyTorch build.
- Fall back to CPU with a detailed warning for incompatible automatic CUDA,
  while explicit CUDA requests fail before model/training setup with device and
  build diagnostics.
- Validate indexed CUDA requests against the visible device count.
- Seed CUDA generators only after the selected CUDA path is verified usable;
  CPU-only and fallback fits no longer touch CUDA seeding.
- Avoid constructing disabled float16 autocast contexts on CPU, preserving
  warnings-as-errors compatibility with the declared PyTorch 2.0 minimum.

### Changed

- Add fitted `device_info_` diagnostics without changing the public constructor.
- Enable CUDA AMP only on verified devices with compute capability 7.0 or newer;
  CPU and older/unknown-capability CUDA paths use FP32.
- Stop forcing fused AdamW and allow the supported PyTorch default to choose its
  optimizer implementation.

### Performance

- Add target, scheduler, best-state restoration, device, and optimizer-path
  profiling information.
- Add a warm, interleaved 0.1.0/0.1.1 regression harness. Local ROC-AUC and
  epoch counts were identical across all 14 dataset/seed pairs. Timing was too
  noisy and unfavorable to claim a speedup, so speculative hot-path changes
  were removed from the candidate.

### Documentation

- Document exact automatic and explicit device behavior, CUDA compatibility
  inspection, official repository URLs, the candidate install command, and the
  reproducible Kaggle CUDA smoke script.
- Refresh pre-publication wording now that the official repository and 0.1.0
  release exist, while making no claim that 0.1.1 is already published.

## [0.1.0] - 2026-08-27

### Added

- `NeuroTabularClassifier`, a scikit-learn-compatible binary neural classifier.
- Automatic pandas dtype detection and explicit integer categorical features.
- Leakage-safe numerical imputation, standard scaling, and missing indicators.
- Per-feature categorical embeddings with missing, unknown, and rare IDs.
- A compact SiLU residual MLP with LayerNorm and one binary logit.
- Direct in-memory tensor batching, deterministic automatic batch sizing,
  CPU/CUDA selection, CUDA mixed precision, and fused-optimizer fallback.
- Internal stratified validation, explicit validation, best-weight restoration,
  and early stopping by loss, ROC-AUC, or accuracy.
- `min_delta`, evaluation frequency, sample weights, and balanced class weights.
- Fit, preprocessing, training, validation, parameter-count, and phase profiling
  attributes.
- Tests for input types, missing values, categoricals, weights, early stopping,
  sklearn compatibility, reproducibility, device behavior, and inference.
- Reproducible benchmark, ablation, and performance profiling harnesses.
- Python 3.10–3.12 packaging and continuous integration.

### Documentation

- Initial README, API reference, usage guide, benchmark reports, research
  notes, contribution policies, security guidance, legal notes, dependency
  notices, versioning policy, and publishing checklist.

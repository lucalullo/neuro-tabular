# Changelog

All notable public changes to NeuroTabular are recorded here. The project uses
the categories Added, Changed, Fixed, Removed, and Documentation when
applicable. Published releases and tags are immutable.

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

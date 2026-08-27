# Changelog

All notable public changes to NeuroTabular are recorded here. The project uses
the categories Added, Changed, Fixed, Removed, and Documentation when
applicable. Published releases and tags are immutable.

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

# NeuroTabular

NeuroTabular is an experimental neural-network library for binary
classification on heterogeneous pandas DataFrames. It provides a compact
scikit-learn-style estimator while handling numerical missing values,
categorical vocabularies, learned embeddings, device selection, batching, and
early stopping automatically.

```python
from neurotabular import NeuroTabularClassifier

model = NeuroTabularClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)
```

## Project status

NeuroTabular 0.1.1 is the recommended 0.1.x release. It is a maintenance and
robustness patch for the experimental, pre-1.0 binary-classification
foundation—not a claim of state-of-the-art performance. It keeps the 0.1.0
architecture, public constructor, and defaults while making automatic CUDA
selection safer and explicit CUDA failures clearer.

## Features

- One sklearn-compatible `NeuroTabularClassifier` estimator.
- pandas `DataFrame` input with strict schema validation and automatic column
  reordering at inference.
- Automatic categorical detection for object, pandas string, category, and
  boolean columns.
- Explicit integer-ID categoricals through `categorical_features`.
- Training-only median imputation, standard scaling, and missing indicators.
- Per-column neural embeddings without default one-hot encoding.
- Separate categorical IDs for missing, unseen, and rare values.
- Internal stratified validation or one explicit validation set.
- Early stopping with `min_delta`, configurable validation frequency, and best
  weight restoration.
- Automatic full/large-batch training through direct tensor indexing.
- Sample weights and balanced binary class weights.
- CPU and verified CUDA paths, safe automatic CPU fallback, local training,
  and bounded-memory inference.
- Profiling attributes for fit, preprocessing, training, and validation.

## Installation

The official repository is
[lucalullo/neuro-tabular](https://github.com/lucalullo/neuro-tabular). After
the 0.1.1 candidate is reviewed and tag `v0.1.1` exists, the reproducible Git
installation is:

```bash
pip install git+https://github.com/lucalullo/neuro-tabular.git@v0.1.1
```

For local candidate review, from the repository root:

```bash
python -m pip install .
```

For development:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -W error
```

The supported Python versions are 3.10, 3.11, and 3.12. Runtime dependencies
are NumPy, pandas, PyTorch, and scikit-learn.

## Data handling

`X` must be a non-empty pandas `DataFrame` with at least one feature and unique
column names. Fitting stores the exact schema. Prediction accepts a different
column order and restores training order automatically; missing, extra, or
duplicate columns raise an error.

Numerical NaNs require no manual preprocessing. NeuroTabular learns medians,
means, and standard deviations from training rows only, appends one missing
indicator per numerical feature, and safely handles constant or entirely
missing columns. Positive and negative infinity are rejected.

Object, string, category, and boolean features are categorical automatically.
Integer IDs can be added without disabling automatic detection:

```python
model = NeuroTabularClassifier(categorical_features=["store_id", "postal_code"])
```

Each categorical column has a compact vocabulary and its own neural embedding.
Missing values use ID 0, categories unseen during fitting use ID 1, and rare
training categories use ID 2. The default `min_category_count=2` can be
overridden.

## Validation and class imbalance

Without `eval_set`, fitting creates a stratified internal validation subset
using `random_state`. An explicit validation set does not change the fitted
preprocessing state or vocabularies:

```python
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

Early stopping supports validation `"loss"`, `"roc_auc"`, and `"accuracy"`.
The default validates every epoch because optimized in-memory validation was
measured to terminate earlier than reduced-frequency validation on the release
ablation matrix. `min_delta=1e-4` prevents negligible changes from resetting
patience.

For imbalanced data:

```python
model = NeuroTabularClassifier(class_weight="balanced")
model.fit(X_train, y_train, sample_weight=row_weights)
```

Class and sample weights are multiplicative and can be combined.

## Devices and performance

`device="auto"` selects CUDA only after PyTorch reports it available, the
device exists, metadata is collected, and a tiny synchronized CUDA kernel
probe succeeds. A visible GPU that is incompatible with the installed PyTorch
build produces a clear warning and falls back to CPU. `device="cuda"` and
indexed CUDA requests never fall back silently: they fail before model or
training tensor creation with the available GPU, compute-capability, PyTorch,
compiled-architecture, and probe diagnostics. `device="cpu"` performs no CUDA
probe.

For small and medium in-memory datasets, arrays are converted to tensors once.
Training uses direct tensor indexing, full batches for very small datasets, and
large deterministic batches otherwise. DataLoader worker processes are not
used for already-materialized arrays. CUDA attempts one-time device residency
when memory permits and otherwise uses pinned CPU staging. Verified CUDA
training uses float16 automatic mixed precision on compute capability 7.0 or
newer. AdamW keeps PyTorch's supported implementation selection instead of
forcing the newer fused path. `torch.compile` remains disabled because compile
cold start was not justified for the target workloads.

After fitting, inspect:

```python
print(model.fit_time_)
print(model.preprocessing_time_)
print(model.training_time_)
print(model.validation_time_)
print(model.batch_size_)
print(model.n_parameters_)
print(model.device_)
print(model.device_info_)
```

## scikit-learn compatibility

The constructor follows estimator conventions and has no training or device
side effects. `clone`, `get_params`, `set_params`, and `cross_val_score` are
supported.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    NeuroTabularClassifier(max_epochs=10),
    X,
    y,
    cv=3,
    scoring="roc_auc",
)
```

## Benchmarks

The release benchmark covers small numerical, mixed, missing-value,
categorical-heavy, moderate-cardinality, imbalanced, and medium synthetic
binary datasets. The measured CPU matrix records accuracy, log loss, fit time,
prediction time, parameter count, preprocessing time, and memory observations.
Results vary materially by dataset; tree baselines remained stronger on
several tasks. See
[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md),
[PERFORMANCE_PROFILE.md](PERFORMANCE_PROFILE.md), and
[ABLATION_REPORT.md](ABLATION_REPORT.md) for methods, hardware, raw summaries,
and limitations. The 0.1.1 maintenance comparison preserves identical local
ROC-AUC and epoch counts across all 14 dataset/seed pairs. Timing was noisy and
did not justify a speedup claim, so attempted hot-path changes were discarded.

These synthetic measurements are engineering diagnostics, not evidence of
universal superiority. External benchmark libraries are not runtime
dependencies.

## Documentation

- [API reference](docs/API.md)
- [Usage guide](docs/USAGE.md)
- [Research and design notes](RESEARCH_NOTES.md)
- [Versioning policy](VERSIONING.md)
- [Publishing policy](PUBLISHING.md)
- [Changelog](CHANGELOG.md)

## Limitations

- Binary classification and pandas `DataFrame` input only.
- One explicit validation set and no validation sample-weight argument.
- No public serialization, calibration, interpretability, or feature-importance
  API.
- Rare bucketing is deliberately simple; extremely high-cardinality columns
  may need domain-specific handling.
- The standard numerical representation does not include learned numerical
  embeddings.
- GPU behavior is implemented and conditionally tested, but the 0.1.1 candidate
  environment had a CPU-only PyTorch build; no CUDA speed or VRAM claim is made.
- Exact CUDA reproducibility can depend on GPU, driver, CUDA, PyTorch, and
  kernel selection.

## Roadmap

Possible future work includes multiclass classification, a regressor,
serialization, calibration, numerical embeddings, feature interactions,
interpretability, and parameter-efficient neural ensembling. This roadmap has
no promised order or dates; changes require tests, documentation, and
reproducible benchmarks.

## Privacy, license, and project information

Training and inference are local. NeuroTabular performs no telemetry, upload,
account access, model download, or dataset download.

NeuroTabular is created by Luca Lullo and developed as an open-source project.
It is licensed under the [MIT License](LICENSE). Dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Naming review is preliminary
engineering due diligence, not professional legal advice or trademark
clearance; see [LEGAL.md](LEGAL.md).

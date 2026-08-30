# NeuroTabular

NeuroTabular is a compact PyTorch binary classifier for heterogeneous pandas
DataFrames. Version 0.2.0 adds leakage-safe categorical frequency features,
dataset-aware categorical embedding widths, optional numerical embeddings and
feature gating, while preserving the scikit-learn estimator interface.

NeuroTabular 0.2.0 is an experimental pre-1.0 release. It is suitable for
controlled experiments and reproducible evaluation; it is not a claim that a
neural model will outperform gradient-boosted trees on every tabular dataset.

## Highlights in 0.2.0

- one estimator API: `fit`, `predict`, and `predict_proba`;
- automatic numerical, object, string, categorical, and boolean handling;
- training-only median imputation, scaling, missing indicators, vocabularies,
  rare buckets, quantile knots, and log-frequency side features;
- adaptive categorical embedding widths bounded by dataset size, cardinality,
  feature count, and a compact memory budget;
- scalar numerical inputs by default, with affine, periodic, and piecewise
  representations available for explicit ablation;
- residual MLP with optional lightweight input gating;
- stratified validation, early stopping, best-weight restoration, sample
  weights, and balanced class weights;
- deterministic automatic batching, CPU/CUDA device diagnostics, AMP only on
  compatible CUDA hardware, and no `torch.autocast` construction on the CPU
  path when AMP is disabled;
- scikit-learn cloning, pipelines, and cross-validation compatibility;
- Python 3.10-3.12 and PyTorch 2.0+ compatibility gates.

The release default is deliberately conservative: scalar numerical inputs,
categorical frequency features enabled, no feature gate, and no full-data
refit. Those choices were selected by the release ablations rather than by
architectural preference.

## Installation

From a local checkout:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests, lint, profiling, and package verification:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
```

Runtime requirements are NumPy 1.24+, pandas 2.0+, scikit-learn 1.3+, and
PyTorch 2.0+. LightGBM and CatBoost are optional benchmark-only comparisons;
they are not runtime dependencies.

## Quick start

```python
import pandas as pd
from neurotabular import NeuroTabularClassifier

X = pd.DataFrame(
    {
        "age": [22, 45, 31, 54, 28, 61],
        "income": [32_000, 78_000, None, 91_000, 46_000, 105_000],
        "city": ["Rome", "Milan", "Rome", "Turin", None, "Milan"],
    }
)
y = [0, 1, 0, 1, 0, 1]

model = NeuroTabularClassifier(
    max_epochs=30,
    eval_metric="roc_auc",
    random_state=42,
)
model.fit(X, y)

labels = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

Integer-coded categorical columns must be named explicitly:

```python
model = NeuroTabularClassifier(categorical_features=["postal_code"])
```

For a user-controlled holdout:

```python
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

Preprocessing is fitted only on training rows. When an external validation set
is supplied, its categories and numeric distribution never affect learned
statistics or vocabularies.

## Optional 0.2 representations

The release default should be the starting point. Alternatives are explicit:

```python
affine = NeuroTabularClassifier(numerical_embedding="affine")
piecewise = NeuroTabularClassifier(numerical_embedding="piecewise")
gated = NeuroTabularClassifier(feature_gating=True)
```

`periodic` and `piecewise` numerical embeddings are implemented and tested but
did not improve the release benchmark mean, so they are not defaults. Likewise,
`full_data_refit=True` repeats training on every row for the selected number of
epochs; the measured quality/cost trade-off did not justify enabling it by
default.

## Validation and imbalance

Without `eval_set`, NeuroTabular creates a deterministic stratified validation
split. `eval_metric` accepts `"loss"`, `"roc_auc"`, or `"accuracy"`. The best
model state is restored after early stopping.

Use `class_weight="balanced"` for inverse-frequency class weighting or pass
non-negative per-row `sample_weight` values to `fit`. If both are supplied,
their effects are multiplied. Binary targets may use any two mutually
comparable labels; predictions return the original labels.

## Devices and performance

`device="auto"` performs a real synchronized CUDA kernel probe. It falls back
to CPU with a diagnostic warning if CUDA is reported but unusable. An explicit
CUDA request fails with device, PyTorch, CUDA, architecture, and probe details
instead of silently changing devices.

The training table is kept resident on the selected device, batches are formed
with index tensors, and inference is batched. AMP is enabled only for a probed
CUDA device with compute capability 7.0 or newer. CPU execution uses a plain
null context when AMP is disabled, which preserves PyTorch 2.0 compatibility
and avoids CPU autocast overhead.

The detailed 5,000-row CPU profile measured 2.772 s end-to-end fit and 0.024 s
prediction in a cold candidate process. A separate paired, warm, interleaved
comparison is the appropriate release comparison: over seven synthetic
datasets and two seeds, 0.2.0 improved mean ROC-AUC from 0.87456 to 0.87841,
with a median paired fit-time ratio of 0.950 and prediction-time ratio of 0.986.
Absolute timings depend strongly on host load and are not universal throughput
claims.

Two bundled scikit-learn datasets provided an external no-regression check:
both versions produced the same mean ROC-AUC, 0.99811. CUDA, AMP throughput,
and VRAM were not measured on the release host because its PyTorch build was
CPU-only.

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md),
[PERFORMANCE_PROFILE_0_2.md](PERFORMANCE_PROFILE_0_2.md), and
[ABLATION_REPORT.md](ABLATION_REPORT.md) for methods, raw-result filenames,
hardware, uncertainty, and rejected experiments.

## scikit-learn use

The constructor stores parameters without fitting side effects, so standard
scikit-learn composition works:

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

base = NeuroTabularClassifier(max_epochs=10, random_state=7)
copy = clone(base)
scores = cross_val_score(copy, X, y, cv=3, scoring="roc_auc")
```

Input to fitting and prediction must be a pandas DataFrame with the same unique
column names. Prediction may reorder columns, but missing or unexpected columns
are rejected rather than guessed.

## Documentation

- [Usage guide](docs/USAGE.md)
- [API reference](docs/API.md)
- [Benchmark report](BENCHMARK_REPORT.md)
- [Performance profile](PERFORMANCE_PROFILE_0_2.md)
- [Ablation report](ABLATION_REPORT.md)
- [Research notes](RESEARCH_NOTES.md)
- [Release notes](RELEASE_NOTES.md)
- [Publishing guide](PUBLISHING.md)

## Limitations

- binary classification only;
- pandas DataFrame input only;
- no calibrated-probability or multiclass interface;
- no claim of state-of-the-art accuracy or tree-model superiority;
- internal validation reduces the rows used for the selected model unless
  optional full-data refit is enabled;
- very high-cardinality categoricals can still increase model size;
- performance and memory results are hardware- and workload-specific;
- CUDA behavior is covered by fallback/probe regression tests, but this release
  was not benchmarked on physical GPU hardware.

## License and privacy

NeuroTabular is created by Luca Lullo and developed as an open-source project.
It is licensed under the [MIT License](LICENSE). Dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Naming review is preliminary
engineering due diligence, not professional legal advice or trademark
clearance; see [LEGAL.md](LEGAL.md).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

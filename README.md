# NeuroTabular

NeuroTabular is an experimental PyTorch library for binary classification on
heterogeneous pandas DataFrames. It exposes a compact scikit-learn-style API,
handles numerical and categorical columns automatically, and keeps preprocessing
inside the estimator so training and inference share the same learned state.

**NeuroTabular 0.2.0 is the first public release in the current public history.**
It is an alpha, pre-1.0 release intended for controlled experiments and practical
use on small and medium in-memory tabular datasets. It does not claim universal
superiority over gradient-boosted trees.

## Highlights

- `NeuroTabularClassifier` with `fit`, `predict`, and `predict_proba`;
- automatic numerical, object, string, categorical, and boolean handling;
- training-only median imputation, standard scaling, missing indicators,
  categorical vocabularies, rare buckets, quantile knots, and log-frequency
  side features;
- declared-but-unobserved pandas `Categorical` levels remain truly unseen;
- adaptive categorical embedding widths with a compact aggregate width budget;
- scalar numerical inputs by default, with affine, periodic, and piecewise
  representations available explicitly for experiments;
- compact residual MLP backbone with optional lightweight input gating;
- AdamW, cosine learning-rate scheduling, automatic batching, stratified
  validation, early stopping, and strict best-checkpoint restoration;
- `sample_weight` and `class_weight="balanced"`, with zero-weight rows ignored
  before target discovery, splitting, and preprocessing;
- transactional re-fitting: a failed refit preserves a previously fitted model;
- caller Python, NumPy, and PyTorch CPU RNG states are restored after fitting;
- explicit checks for unsafe float32 conversion and complex-valued inputs;
- CPU/CUDA device diagnostics, synchronized CUDA usability probing, and AMP only
  on compatible CUDA hardware;
- scikit-learn cloning, pipelines, and cross-validation compatibility;
- Python 3.10-3.12 and PyTorch 2.0+ compatibility gates;
- Linux and Windows CI, source/wheel build checks, and external wheel smoke tests.

The default configuration is intentionally conservative: scalar numerical
representation, categorical frequency features enabled, no feature gate, and no
full-data refit. Experimental research ideas are not enabled by default without
broader evidence.

## Installation

From the GitHub release tag:

```bash
python -m pip install "git+https://github.com/lucalullo/neuro-tabular.git@v0.2.0"
```

From a local checkout:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and verification:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
```

Runtime requirements are NumPy 1.24+, pandas 2.0+, scikit-learn 1.3+, and
PyTorch 2.0+. LightGBM and CatBoost are optional benchmark-only comparisons and
are not runtime dependencies.

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

model = NeuroTabularClassifier(max_epochs=30, eval_metric="roc_auc", random_state=42)
model.fit(X, y)

labels = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

Integer-coded categorical columns should be named explicitly:

```python
model = NeuroTabularClassifier(categorical_features=["postal_code"])
```

For a user-controlled holdout:

```python
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

Preprocessing is fitted only on the training rows. Categories and numerical
statistics from an external validation frame never enter the learned training
preprocessor.

## Optional representations

The release default should be the starting point. Alternatives are explicit:

```python
affine = NeuroTabularClassifier(numerical_embedding="affine")
piecewise = NeuroTabularClassifier(numerical_embedding="piecewise")
gated = NeuroTabularClassifier(feature_gating=True)
```

`periodic` and `piecewise` numerical embeddings are implemented and tested but
were not selected as defaults in the internal ablation suite. Likewise,
`full_data_refit=True` retrains on all primary rows for the selected number of
epochs; it remains opt-in because its quality/cost trade-off was not consistently
better in development experiments.

## Validation and imbalance

Without `eval_set`, NeuroTabular creates a deterministic stratified validation
split. `eval_metric` accepts `"loss"`, `"roc_auc"`, or `"accuracy"`. The strictly
best observed model state is restored after early stopping; `min_delta` controls
patience rather than blocking restoration of a smaller strict improvement.

Use `class_weight="balanced"` for inverse-frequency class weighting or pass
non-negative per-row `sample_weight` values to `fit`. If both are supplied,
their effects are multiplied. Rows with zero effective sample weight are ignored
before target validation and preprocessing.

## Devices and performance

`device="auto"` performs a real synchronized CUDA kernel probe. It falls back to
CPU with a diagnostic warning if CUDA is reported but unusable. An explicit CUDA
request fails with diagnostic details instead of silently changing devices.

Training data is converted once and kept on the selected device, batches are
formed with index tensors, and inference is batched. AMP is enabled only after a
compatible CUDA probe. CPU execution uses a plain null context when AMP is
disabled.

The bundled engineering benchmark is deliberately modest. On the documented
seven-dataset synthetic development matrix, the current default configuration
recorded mean ROC-AUC `0.878408` and mean log loss `0.408868`. A contextual
`HistGradientBoostingClassifier` baseline recorded mean ROC-AUC `0.873459` and
mean log loss `0.487927` on the same matrix. Two bundled scikit-learn numerical
datasets gave mean ROC-AUC `0.998113` for NeuroTabular in the documented check.
These are workload-specific engineering measurements, not claims of universal
superiority. See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) and
[PERFORMANCE_PROFILE.md](PERFORMANCE_PROFILE.md).

## scikit-learn use

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

base = NeuroTabularClassifier(max_epochs=10, random_state=7)
copy = clone(base)
scores = cross_val_score(copy, X, y, cv=3, scoring="roc_auc")
```

Input to fitting and prediction must be a pandas DataFrame with unique column
names. Prediction may reorder columns by learned schema, but missing or
unexpected columns are rejected rather than guessed.

## Documentation

- [Usage guide](docs/USAGE.md)
- [API reference](docs/API.md)
- [Benchmark report](BENCHMARK_REPORT.md)
- [Performance profile](PERFORMANCE_PROFILE.md)
- [Ablation report](ABLATION_REPORT.md)
- [Research notes](RESEARCH_NOTES.md)
- [Release notes](RELEASE_NOTES.md)
- [Publishing guide](PUBLISHING.md)

## Limitations

- binary classification only;
- pandas DataFrame input only;
- no built-in multiclass, regression, or calibrated-probability API yet;
- no claim of state-of-the-art accuracy or tree-model superiority;
- internal validation reduces the rows used for the selected model unless
  optional full-data refit is enabled;
- the embedding-width budget is not a byte-level RAM/VRAM cap; very
  high-cardinality categoricals can still increase model size substantially;
- concurrent `fit` calls in the same process are not guaranteed thread-safe
  because fitting temporarily controls process-global RNG state;
- performance and memory measurements are hardware- and workload-specific;
- CUDA behavior is covered by fallback/probe regression tests, but this release
  has not been performance-benchmarked on physical GPU hardware.

## Status

NeuroTabular 0.2.0 is experimental and pre-1.0. The public API, architecture,
and defaults may evolve as broader real-world validation continues.

## License

NeuroTabular is licensed under the [MIT License](LICENSE). Dependency notices
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See [LEGAL.md](LEGAL.md)
for the project's legal/naming note.

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

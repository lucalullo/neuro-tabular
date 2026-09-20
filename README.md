# NeuroTabular

NeuroTabular is an experimental PyTorch library for binary classification,
multiclass classification, and single-output regression on heterogeneous pandas
DataFrames. It exposes a compact scikit-learn-style API, handles numerical and
categorical columns automatically, and keeps preprocessing inside the estimator
so training and inference share the same learned state.

**NeuroTabular 0.3.0 adds multiclass classification and regression while
preserving the validated binary behavior of the 0.2 line.** It remains an alpha,
pre-1.0 release intended for controlled experiments and practical use on small
and medium in-memory tabular datasets. It does not claim universal superiority
over gradient-boosted trees.

## Highlights

- `NeuroTabularClassifier` automatically handles binary and multiclass targets;
- `NeuroTabularRegressor` learns a continuous target and predicts in original units;
- automatic numerical, object, string, categorical, and boolean handling;
- training-only median imputation, standard scaling, missing indicators,
  categorical vocabularies, rare buckets, quantile knots, and log-frequency
  side features;
- declared-but-unobserved pandas `Categorical` levels remain truly unseen;
- adaptive categorical embedding widths with a compact aggregate width budget;
- scalar numerical inputs by default, with affine, periodic, and piecewise
  representations available explicitly for experiments;
- compact residual MLP backbone with optional lightweight input gating;
- AdamW, cosine learning-rate scheduling, automatic batching, deterministic
  validation, early stopping, and strict best-checkpoint restoration;
- `sample_weight` across all tasks and `class_weight="balanced"` for
  classification, with zero-weight rows ignored before target discovery,
  splitting, and preprocessing;
- transactional re-fitting: a failed refit preserves a previously fitted model;
- caller Python, NumPy, and PyTorch CPU RNG states are restored after fitting;
- explicit checks for unsafe float32 conversion and complex-valued inputs;
- CPU/CUDA device diagnostics, synchronized CUDA usability probing, and AMP only
  on compatible CUDA hardware;
- scikit-learn cloning, pipelines, and cross-validation compatibility;
- pickle/joblib round trips validated in fresh processes for all three tasks;
- Python 3.10-3.12 and PyTorch 2.0+ compatibility gates;
- Linux and Windows CI, source/wheel build checks, and installed-artifact smoke tests.

The default configuration is intentionally conservative: scalar numerical
representation, categorical frequency features enabled, no feature gate, and no
full-data refit. Experimental research ideas are not enabled by default without
broader evidence.

## Installation

From GitHub after the `v0.3.0` tag is published:

```bash
python -m pip install "git+https://github.com/lucalullo/neuro-tabular.git@v0.3.0"
```

In Kaggle:

```python
!pip install -qq git+https://github.com/lucalullo/neuro-tabular.git@v0.3.0
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

Binary classification:

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

model = NeuroTabularClassifier()
model.fit(X, y)

labels = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

Multiclass classification uses the same estimator, without a task parameter:

```python
from sklearn.datasets import load_iris
from neurotabular import NeuroTabularClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
clf = NeuroTabularClassifier().fit(X, y)
probabilities = clf.predict_proba(X)  # shape (150, 3), columns follow clf.classes_
labels = clf.predict(X)
```

Regression:

```python
from sklearn.datasets import load_diabetes
from neurotabular import NeuroTabularRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
reg = NeuroTabularRegressor().fit(X, y)
prediction = reg.predict(X)  # original target units, shape (442,)
```

These examples demonstrate the API; evaluate quality on a separate test set.

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

## Validation, metrics, and weights

Without `eval_set`, NeuroTabular creates a deterministic validation split,
stratified for classification and shuffled without stratification for regression.
Classifier `eval_metric` accepts `"loss"` and `"accuracy"`; `"roc_auc"` is
binary-only. Regressor metrics are `"loss"` (MSE), `"rmse"`, `"mae"`, and `"r2"`.
Loss is the default for every task. The strictly best observed model state is
restored after early stopping; `min_delta` controls patience rather than blocking
restoration of a smaller strict improvement.

Use `class_weight="balanced"` for inverse-frequency class weighting or pass
non-negative per-row `sample_weight` values to `fit`. If both are supplied for
classification, their effects are multiplied. Rows with zero effective sample
weight are ignored before target validation and preprocessing.

Regression uses train-only weighted population mean and standard deviation for
the target, with MSE on the standardized scale and an automatic inverse transform
at prediction. A constant training target produces the exact constant. See the
[API reference](docs/API.md) for task-specific details.

## Persistence

Pickle and joblib round trips are tested in fresh processes for binary,
multiclass, and regression models. Use matching package/dependency versions when
loading persisted models; cross-version and cross-device persistence
compatibility is not guaranteed. Only load trusted pickle/joblib artifacts.
See the [usage guide](docs/USAGE.md#persistence).

## Devices and performance

`device="auto"` performs a real synchronized CUDA kernel probe. It falls back to
CPU with a diagnostic warning if CUDA is reported but unusable. An explicit CUDA
request fails with diagnostic details instead of silently changing devices.

Training data is converted once and kept on the selected device, batches are
formed with index tensors, and inference is batched. AMP is enabled only after a
compatible CUDA probe. CPU execution uses a plain null context when AMP is
disabled.

The frozen 0.3 development benchmark contains 330 fits across 22 task/dataset
cases and three seeds, comparing NeuroTabular with four fixed tree recipes. Mean
binary ROC-AUC is `0.8461`, multiclass accuracy is `0.8275`, and regression R² is
`0.5320`. These are workload-specific engineering measurements, not a claim of
universal superiority. See [0.3 benchmark evidence](docs/BENCHMARK_0_3.md) for
scope, exact aggregates, comparisons, and known failure cases. Historical 0.2
binary measurements remain in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

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

* [Usage guide](docs/USAGE.md)
* [API reference](docs/API.md)
* [0.3 benchmark evidence](docs/BENCHMARK_0_3.md)
* [0.3 architecture overview](docs/architecture_0_3.md)
* [Historical 0.2 benchmark report](BENCHMARK_REPORT.md)
* [Historical 0.2 performance profile](PERFORMANCE_PROFILE.md)
* [Historical 0.2 ablation report](ABLATION_REPORT.md)
* [Research notes](RESEARCH_NOTES.md)
* [Release notes](RELEASE_NOTES.md)
* [Publishing guide](PUBLISHING.md)

## Limitations

* pandas DataFrame input only;
* single-output targets; no multilabel, multioutput regression, or calibrated-probability API;
* multiclass ROC-AUC is not an early-stopping metric; use loss or accuracy;
* credit-g and Titanic show weaker binary results and split sensitivity;
* car can collapse to its majority class; yeast/glass have weak rare-class coverage;
* quake has negative R² and triazines is weak and split-sensitive; see the
  [benchmark limitations](docs/BENCHMARK_0_3.md#known-limitations);
* no claim of state-of-the-art accuracy or tree-model superiority;
* internal validation reduces the rows used for the selected model unless
  optional full-data refit is enabled;
* the embedding-width budget is not a byte-level RAM/VRAM cap; very
  high-cardinality categoricals can still increase model size substantially;
* concurrent `fit` calls in the same process are not guaranteed thread-safe
  because fitting temporarily controls process-global RNG state;
* performance and memory measurements are hardware- and workload-specific;
* CUDA behavior is covered by fallback/probe regression tests, but physical GPU
  validation was not completed for this release.

## Status

NeuroTabular 0.3.0 is experimental and pre-1.0. The public API, architecture,
and defaults may evolve as broader real-world validation continues.

## License

NeuroTabular is created by Luca Lullo and released under the [MIT License](LICENSE). Dependency notices
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See [LEGAL.md](LEGAL.md)
for the project's legal/naming note.

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

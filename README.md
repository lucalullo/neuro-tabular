# NeuroTabular

NeuroTabular is a PyTorch library for binary classification,
multiclass classification, and single-output regression on heterogeneous pandas
DataFrames. It exposes a compact scikit-learn-style API, handles numerical and
categorical columns automatically, and keeps preprocessing inside the estimator
so training and inference share the same learned state.

**Version 2.0.0 consolidates the stable neural product and its maintenance policies.**
The major number is an explicit maintainer choice for project maturity; it does
not denote a newly validated predictive architecture or a general accuracy gain.
The public estimator API and defaults retain the 1.x behavior. Complex numerical
columns are rejected before lossy conversion, including beside pandas nullable
columns. Conservative preprocessing/inference optimizations already present in
1.0.3 are retained. Their qualified local benefits are not new 2.0 speedups or a
general training/RAM advantage. See the [release notes](RELEASE_NOTES.md),
[support scope](docs/policies/SUPPORT_SCOPE.md), [API boundary](docs/API_BOUNDARY.md)
and [target contract](docs/TARGET_CONTRACT.md).

PyPI availability and publication approval are separate from the source version.
Use `python -m pip install neurotabular` for the published package; an explicit
`neurotabular==2.0.0` pin requires that the maintainer has published that version.
Previous releases remain historical reproducibility references rather than
alternative recommended release lines after 2.0 publication.

The long-term research mission is an original neural tabular algorithm that
learns more accurately, generalizes reliably, and reduces complete training,
inference and memory costs. These are objectives, not implemented or demonstrated
advantages of the current release. The [canonical vision](docs/NEUROTABULAR_VISION.md)
defines that direction; [AGENTS.md](AGENTS.md) provides contributor-agent working
instructions. Research prototypes remain separate from the stable product.

## Highlights

- retain CPU prediction-path overhead reductions from 0.3.1;
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
- explicit checks for unsafe float32 conversion, overflowing numerical training
  statistics, complex-valued inputs and complex sample weights;
- CPU/CUDA device diagnostics, synchronized CUDA usability probing, and AMP only
  on compatible CUDA hardware;
- scikit-learn cloning, pipelines, and cross-validation compatibility;
- pickle/joblib round trips validated in fresh processes for all three tasks;
- conventional Python 3.11–3.14 coverage and Python 3.10 legacy compatibility;
- Linux CPU software checks, historical Windows 11 Sandbox qualification,
  installed-artifact checks, and exact-version CI requirements; see the support scope.

The default configuration is intentionally conservative: scalar numerical
representation, categorical frequency features enabled, no feature gate, and no
full-data refit. Experimental research ideas are not enabled by default without
broader evidence.

## Installation

Use conventional CPython 3.11–3.14 for the documented installation path:

```bash
python -m pip install neurotabular
```

In a fresh Kaggle or Colab notebook using a supported Python runtime:

```python
!pip install neurotabular
from neurotabular import NeuroTabularClassifier
```

The historical 1.0.2 audit tested exact CPython 3.13.15, including normal
PyPI dependency resolution. It has not been executed inside Kaggle or Colab.
When reproducing the published baseline, install `neurotabular==1.0.3` and check
that `neurotabular.__version__` is `1.0.3`. A development checkout can retain
different version metadata; see [Status](#status) before installing from source.
Check `neurotabular.__version__` after installation. Restart an existing notebook
session if it already imported another version.

Python 3.10 works in the tested legacy stack but is upstream EOL. Python 3.15
passes the full Linux CPU code suite using the official PyTorch CPU channel;
ordinary PyPI installation is currently limited by missing Torch wheels. If you
need 3.15, install Torch using its documented channel before NeuroTabular:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install neurotabular
```

This is an explicit alternative installation path. CPython 3.11–3.14 is preferred
for the ordinary one-command experience. The upstream distribution gap has no
promised resolution date. See the [PyTorch 2.14 announcement](https://pytorch.org/blog/pytorch-2-14-release-blog/).

An initial host Windows audit was blocked by Application Control; subsequent
native Windows 11 Sandbox qualification passed in its pinned CPython 3.12 x64
environment without disabling protections. That does not certify every Windows
configuration. CI coverage and local-host validation are distinct; macOS is not
locally validated. Physical NVIDIA CUDA remains best effort and unvalidated.
The support scope records the environments and limits of executed evidence.

From a local checkout:

```bash
python -m venv .venv
```

Activate that environment before installing (choose the command for your shell):

```bash
source .venv/bin/activate  # Linux/macOS bash
```

```powershell
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

Then use its Python:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

If your shell prevents activation, use `.venv/bin/python` on Linux/macOS or
`.\.venv\Scripts\python.exe` on Windows in place of `python`; no policy change
is needed. Confirm `python -c "import neurotabular; print(neurotabular.__version__)"`.

For development and verification:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
```

Runtime requirements are NumPy 1.24+, pandas 2.0+, scikit-learn 1.3+, and
PyTorch 2.0+. Pip also resolves Python-specific joblib floors for stdlib, caching and
parallel-worker compatibility. LightGBM and CatBoost are optional benchmark-only comparisons and
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
[API reference](https://github.com/lucalullo/neuro-tabular/blob/main/docs/API.md) for task-specific details.

## Persistence

Pickle and joblib round trips are tested in fresh processes for binary,
multiclass, and regression models. Use matching package/dependency versions when
loading persisted models; cross-version and cross-device persistence
compatibility is not guaranteed. Only load trusted pickle/joblib artifacts.
See the [usage guide](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md#persistence).

## Devices and performance

`device="auto"` performs a real synchronized CUDA kernel probe. It falls back to
CPU with a diagnostic warning if CUDA is reported but unusable. An explicit CUDA
request fails with diagnostic details instead of silently changing devices.

Training data is converted once and kept on the selected device, batches are
formed with index tensors, and inference is batched. AMP is enabled only after a
compatible CUDA probe. CPU execution uses a plain null context when AMP is
disabled.

For the 0.3.1 patch, the internal P35 CPU/WSL software panel measured median
latency reductions of about 31% for `predict` and 36% for `predict_proba` versus
the 0.3.0 path, with bit-for-bit outputs on the frozen equivalence checks. These
figures are hardware- and workload-specific; see
[0.3.1 performance validation](https://github.com/lucalullo/neuro-tabular/blob/main/docs/PERFORMANCE_0_3_1.md).

The frozen 0.3 development benchmark contains 330 fits across 22 task/dataset
cases and three seeds, comparing NeuroTabular with four fixed tree recipes. Mean
binary ROC-AUC is `0.8461`, multiclass accuracy is `0.8275`, and regression R² is
`0.5320`. These are workload-specific engineering measurements, not a claim of
universal superiority. See [0.3 benchmark evidence](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md) for
scope, exact aggregates, comparisons, and known failure cases. Historical 0.2
binary measurements remain in [BENCHMARK_REPORT.md](https://github.com/lucalullo/neuro-tabular/blob/main/BENCHMARK_REPORT.md).

## scikit-learn use

```python
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from neurotabular import NeuroTabularClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
base = NeuroTabularClassifier(max_epochs=10, random_state=7)
copy = clone(base)
scores = cross_val_score(copy, X, y, cv=3, scoring="roc_auc")
```

For a complete DataFrame-preserving Pipeline, CV and small grid-search example,
run `python examples/pipeline_workflow.py` from the local development checkout.

Input to fitting and prediction must be a pandas DataFrame with unique column
names. Prediction may reorder columns by learned schema, but missing or
unexpected columns are rejected rather than guessed.

## Documentation

### Maintenance policies and stable scope

The stable 1.0 API follows the [API maintenance](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/API_STABILITY_POLICY.md),
[deprecation](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/DEPRECATION_POLICY.md),
[persistence](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/PERSISTENCE_POLICY.md),
[warning](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/WARNING_POLICY.md) and
[versioning](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/VERSIONING_POLICY.md) policies. The reviewed
[1.0 API boundary](https://github.com/lucalullo/neuro-tabular/blob/main/docs/API_BOUNDARY.md) is the stable public contract; items
explicitly classified as experimental or internal remain outside that guarantee.
[Support scope](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/SUPPORT_SCOPE.md) documents the validated
Linux Python 3.11–3.14 CPU matrix, 3.10 legacy support and remaining evidence limits. CUDA remains optional
best effort with physical NVIDIA validation pending; macOS remains outside the
validated support scope. Historical reference models and private validation
evidence are not packaged. See the [canonical policy index](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/README.md)
for sources of truth.

### Complete mixed-data workflows

Run the same executable example for each supported task:

```bash
python examples/mixed_dataframe_workflow.py --task binary
python examples/mixed_dataframe_workflow.py --task multiclass
python examples/mixed_dataframe_workflow.py --task regression
```

It includes a train/test split, nullable and categorical features, missing and
unseen values, reordered prediction columns and scores. Add `--output-dir ./models`
for a same-runtime joblib round-trip and optional metadata JSON. The small
synthetic examples demonstrate usage; they are not benchmark results or tuning
recommendations. See [usage](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md#complete-task-and-device-examples).
NeuroTabular 1.0.2 includes the optional `input_schema_` diagnostic. The
examples exercise the stable task APIs while keeping diagnostic-only fields
outside the core prediction contract.

* [Usage guide](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md)
* [API reference](https://github.com/lucalullo/neuro-tabular/blob/main/docs/API.md)
* [Minimal reproducer and safe environment report](https://github.com/lucalullo/neuro-tabular/blob/main/docs/support/MINIMAL_REPRODUCER_GUIDE.md)
* [0.3 benchmark evidence](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md)
* [0.3.1 performance validation](https://github.com/lucalullo/neuro-tabular/blob/main/docs/PERFORMANCE_0_3_1.md)
* [0.3 architecture overview](https://github.com/lucalullo/neuro-tabular/blob/main/docs/architecture_0_3.md)
* [Historical 0.2 benchmark report](https://github.com/lucalullo/neuro-tabular/blob/main/BENCHMARK_REPORT.md)
* [Historical 0.2 performance profile](https://github.com/lucalullo/neuro-tabular/blob/main/PERFORMANCE_PROFILE.md)
* [Historical 0.2 ablation report](https://github.com/lucalullo/neuro-tabular/blob/main/ABLATION_REPORT.md)
* [Research notes](https://github.com/lucalullo/neuro-tabular/blob/main/RESEARCH_NOTES.md)
* [Release notes](https://github.com/lucalullo/neuro-tabular/blob/main/RELEASE_NOTES.md)
* [Publishing guide](https://github.com/lucalullo/neuro-tabular/blob/main/PUBLISHING.md)

## Limitations

* pandas DataFrame input only;
* single-output targets; no multilabel, multioutput regression, or calibrated-probability API;
* multiclass ROC-AUC is not an early-stopping metric; use loss or accuracy;
* credit-g and Titanic show weaker binary results and split sensitivity;
* car can collapse to its majority class; yeast/glass have weak rare-class coverage;
* quake has negative R² and triazines is weak and split-sensitive; see the
  [benchmark limitations](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md#known-limitations);
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

These are the NeuroTabular 2.0.0 release sources. Published 1.0.3 artifacts and
earlier tags remain immutable historical baselines. Version 2.0 consolidates
the product and its validation; it does not introduce a new predictive
architecture or establish general accuracy, training-speed or RAM gains.

Dedicated native Linux/Windows performance qualification of the additional
complex-plus-nullable validation remains incomplete. The maintainer explicitly
approved proceeding without making that qualification a prerequisite for this
release, accepting the unquantified performance risk. Historical inconclusive
results remain unchanged; this decision is not a performance PASS or independent
certification. Functional tests and exact-commit release CI remain mandatory.

Registry availability is determined by the actual published artifacts, not this
source version. Release actions require separate maintainer authorization and
the exact-source checks described in [publishing](PUBLISHING.md).
Current capabilities and [limitations](#limitations) remain distinct from the
[research vision](docs/NEUROTABULAR_VISION.md).

## License

NeuroTabular is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/neuro-tabular/blob/main/LICENSE). Dependency notices
are in [THIRD_PARTY_NOTICES.md](https://github.com/lucalullo/neuro-tabular/blob/main/THIRD_PARTY_NOTICES.md). See [LEGAL.md](https://github.com/lucalullo/neuro-tabular/blob/main/LEGAL.md)
for the project's legal/naming note.

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

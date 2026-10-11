# NeuroTabular 1.0.0 usage guide

## Minimal binary classification

```python
from neurotabular import NeuroTabularClassifier

model = NeuroTabularClassifier(random_state=42)
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
prediction = model.predict(X_test)
```

`X_train` and `X_test` must be pandas DataFrames. Targets may be NumPy arrays,
pandas Series, or other one-dimensional array-like objects. This example uses
two classes; the same estimator detects multiclass targets automatically.

## Multiclass classification

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from neurotabular import NeuroTabularClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
clf = NeuroTabularClassifier(eval_metric="loss").fit(X_train, y_train)
p = clf.predict_proba(X_test)
print(clf.classes_, p.shape)
print(accuracy_score(y_test, clf.predict(X_test)))
print(f1_score(y_test, clf.predict(X_test), average="macro"))
print(log_loss(y_test, p, labels=clf.classes_))
print(roc_auc_score(y_test, p, multi_class="ovr", average="macro"))
```

Probability columns follow sorted `classes_`. String labels and nonconsecutive
integer labels work identically. Multiclass early stopping supports loss and
accuracy. The ROC-AUC above is an external macro one-versus-rest evaluation,
defined when all classes are represented in the test target.

## Regression

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from neurotabular import NeuroTabularRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
reg = NeuroTabularRegressor(eval_metric="loss").fit(X_train, y_train)
prediction = reg.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, prediction)))  # RMSE, original units
print(mean_absolute_error(y_test, prediction))
print(r2_score(y_test, prediction))
print(reg.target_mean_, reg.target_scale_)
scores = cross_val_score(make_pipeline(NeuroTabularRegressor()), X, y, cv=3)
```

The regressor standardizes its target using training-only weighted population
moments, then reverses the transform automatically. Pass unscaled original y to
both fit and eval_set. Negative values and integer-valued continuous outcomes
are supported. Constant training targets return the exact constant. Regression
validation is a random split without stratification; loss/RMSE/MAE are minimized,
R² is maximized. Validation history uses standardized units, while predictions
and the metrics above use original units.

## Persistence

```python
import pickle
import joblib

with open("model.pkl", "wb") as stream:
    pickle.dump(reg, stream)
joblib.dump(reg, "model.joblib")

# In a new Python process with the same package/dependency versions:
loaded = joblib.load("model.joblib")
prediction = loaded.predict(X_test)
```

Both formats are tested in new processes with identical predictions for all
three tasks. Store a dependency lock or environment record alongside artifacts.
Cross-version loading and moving CUDA models to CPU are not guaranteed. Load
only artifacts from a trusted source. Persistence does not replace storing the
original DataFrame schema and an independent validation dataset.

## Recommended evaluation

Use a holdout that is never supplied to `fit` for final metrics:

```python
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

model = NeuroTabularClassifier(eval_metric="roc_auc", random_state=42)
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, probability))
print(log_loss(y_test, probability))
```

The estimator still creates an internal split from `X_train` for early
stopping. The held-out test frame remains untouched.

## Explicit validation

```python
model = NeuroTabularClassifier(eval_metric="roc_auc")
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

With external validation, preprocessing is fitted on positive-weight `X_train` rows and
never on `X_valid`. Both target classes must be present in validation when
`eval_metric="roc_auc"`.

## Missing numerical values

No external imputer is required:

```python
X = X.copy()
X.loc[5, "income"] = float("nan")
model.fit(X, y)
```

The training partition supplies a median, center, scale, and quantile knots for
each numerical feature. The transformed matrix contains scaled values plus a
separate missing indicator. All-missing and constant training columns are
stabilized. Positive and negative infinity are rejected.

## Categorical values

Object, pandas string, pandas categorical, and boolean columns are detected
automatically:

```python
X = X.assign(
    city=X["city"].astype("string"),
    subscribed=X["subscribed"].astype(bool),
)
model.fit(X, y)
```

Integer-coded categories require an explicit declaration:

```python
model = NeuroTabularClassifier(
    categorical_features=["postal_code", "store_id"],
)
```

Missing, unseen, and rare values use separate IDs. Values with fewer than
`min_category_count` training occurrences use the aggregate rare bucket.
Category frequency side features are enabled by default and are calculated
from training counts only. Unknown values receive frequency zero.

Disable the frequency side channel for an ablation or a frequency-feature ablation:

```python
model = NeuroTabularClassifier(use_category_frequency=False)
```

## Numerical representation experiments

Scalar input was selected by the release ablation. Other tested modes remain
available when a dataset-specific validation protocol supports them:

```python
for mode in ["scalar", "affine", "periodic", "piecewise"]:
    model = NeuroTabularClassifier(
        numerical_embedding=mode,
        random_state=42,
    )
```

`piecewise` uses training-only quantile knots. `periodic` uses learned
sinusoidal projections. These modes can increase parameters and compute; do not
assume they improve a particular dataset without repeated validation.

The optional input gate is independent:

```python
model = NeuroTabularClassifier(feature_gating=True)
```

It is disabled by default because it did not improve the release ablation mean.

## Sample and class weights

```python
model.fit(X_train, y_train, sample_weight=row_weights)
```

Weights must be finite, one-dimensional, non-negative, and aligned to rows.
At least one weight must be positive. Rows whose weight is zero are ignored
completely, including for preprocessing statistics and target-class discovery.

For inverse-frequency binary or multiclass training class weights:

```python
model = NeuroTabularClassifier(class_weight="balanced")
model.fit(X_train, y_train, sample_weight=row_weights)
```

When both mechanisms are used, per-class and per-row weights multiply.

## Cross-validation and pipelines

This standalone example defines a binary target; do not reuse regression y or
use binary ROC-AUC on a multiclass target.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from neurotabular import NeuroTabularClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = NeuroTabularClassifier(max_epochs=20, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
```

The classifier performs its own DataFrame preprocessing, so a separate encoder
is usually unnecessary. Any upstream transformer in a scikit-learn pipeline
must preserve or reconstruct a DataFrame because NumPy matrix input is rejected.

## Optional full-data refit

```python
model = NeuroTabularClassifier(full_data_refit=True)
model.fit(X_train, y_train)
print(model.full_data_refit_)
```

After internal epoch selection, a fresh preprocessor and network are fitted on
all rows for `best_epoch_` epochs. The release default is `False`: the measured
ablation increased median fit time by roughly 68% and reduced mean ROC-AUC on
the screening workloads. When `eval_set` is explicit, refit is skipped and
`full_data_refit_` remains false.

## CPU and CUDA

Force CPU:

```python
model = NeuroTabularClassifier(device="cpu")
```

Try CUDA with a safe fallback:

```python
model = NeuroTabularClassifier(device="auto")
model.fit(X_train, y_train)
print(model.device_)
print(model.device_info_)
```

Require CUDA and fail if incompatible:

```python
model = NeuroTabularClassifier(device="cuda:0")
```

Automatic selection launches and synchronizes a small kernel. A failed probe
warns and falls back to CPU. Explicit selection raises a detailed error. AMP is
enabled only on a successfully probed CUDA device of sufficient capability.
CPU training never constructs an autocast context when AMP is disabled.

## Batch sizing and fast experiments

`batch_size="auto"` is deterministic for a given processed schema, model, and
device. To compare architectures under a fixed batch:

```python
model = NeuroTabularClassifier(
    batch_size=256,
    max_epochs=10,
    patience=2,
    eval_frequency=1,
)
```

For a quick smoke test, lower `max_epochs`; for a fair model comparison, keep
the split, seeds, metric, epoch budget, and preprocessing protocol fixed.

## Reproducibility

```python
first = NeuroTabularClassifier(random_state=19).fit(X, y)
second = NeuroTabularClassifier(random_state=19).fit(X, y)
```

The seed controls the internal split, Python, NumPy, PyTorch initialization, and
batch shuffling. Fitting restores the caller's Python, NumPy, and PyTorch CPU RNG
states before returning. Exact cross-hardware floating-point identity is not
promised, especially across CUDA architectures or library versions.

## Diagnostics

After a successful fit, call prediction before inspecting its duration:

```python
model.predict(X_test)
print(model.best_epoch_, model.n_iter_, model.best_score_)
print(model.n_parameters_, model.embedding_dimensions_)
print(model.fit_time_, model.last_prediction_time_)
print(model.profile_["preprocessing"])
print(model.profile_["training"])
```

`history_` contains only validation checkpoints. `profile_` separates
preprocessing, tensor conversion, transfer, batch construction, forward,
backward, optimizer, validation, metric, checkpoint, restoration, and setup
where applicable.

## Prediction schema

### Inspecting a fitted model

Version 1.0.0 includes a compact schema snapshot; the published v0.3.1
tag and its saved models do not yet expose it:

```python
import pandas as pd

print(pd.DataFrame(model.input_schema_))
print(model.device_, model.device_info_)
print(model.numeric_features_, model.categorical_features_)
print(pd.DataFrame(model.history_))
```

Schema counts are learned only from the final training partition. They include
rare categories but exclude missing, validation-only and declared-but-unobserved
values. They describe the final encoder after optional full-data refit; history
describes the earlier validation fit. Treat the snapshot as read-only.
History and device diagnostics already existed before P37; no new metric,
prediction tensor or training pass is introduced. See the
[API schema contract](API.md#unreleased-p37-fitted-schema-diagnostic).

### Keeping the prediction schema

The prediction frame may present fitted columns in another order. NeuroTabular
reorders them to the fitted schema. Missing or extra columns raise an error:

```python
model.predict_proba(X_test[model.feature_names_in_[::-1]])  # accepted
model.predict_proba(X_test.drop(columns=["income"]))  # rejected
```

This strictness prevents silent feature misalignment.

Prediction uses the fitted numeric/categorical roles. A numeric column can still
use the existing numeric-string conversion, but a non-numeric value fails with
the offending column name. Clean the values; if the feature really represents
categories, declare it in `categorical_features` and refit. Changing a prediction
dtype does not train a new vocabulary. New categorical levels remain unknown.
For unsafe numeric magnitudes, rescale input consistently and refit instead of
discarding imaginary components, clipping overflow, or suppressing errors.

For an upstream `ColumnTransformer`, use `.set_output(transform="pandas")`
and ensure the transformed frame has stable unique column names. Default NumPy
output is rejected. An executable mixed-data pipeline, cross-validation and
small grid-search workflow is in `examples/pipeline_workflow.py`.

CPU and mocked device paths are covered by tests, but physical CUDA hardware
validation was not completed for 0.3.1. See the
[benchmark limitations](BENCHMARK_0_3.md#known-limitations).

## Complete task and device examples

`examples/mixed_dataframe_workflow.py` is a complete synthetic workflow for
binary, multiclass and single-output regression, with a stratified test split
for classification and a random split for regression. All three use mixed
DataFrames with numeric missing values, nullable integers/booleans and
categoricals. Prediction shows new categories and reordered columns. `score`
uses the original test frame; its value is illustrative, not a quality claim.

```bash
python examples/mixed_dataframe_workflow.py --task binary --device cpu
python examples/mixed_dataframe_workflow.py --task multiclass --device cpu
python examples/mixed_dataframe_workflow.py --task regression --device cpu --output-dir ./models
```

`--device auto` uses the existing checked selection/fallback path. On a machine
with supported NVIDIA hardware and a CUDA-enabled PyTorch runtime,
`--device cuda` explicitly requests CUDA and raises if unavailable. P39 was
validated on CPU; the local AMD adapter and CPU Torch do not validate physical
CUDA. No CUDA equivalence or memory claim follows from this example.

With `--output-dir`, the example saves joblib and an optional human-readable
metadata sidecar, loads the trusted local artifact and checks identical CPU
predictions in the same runtime. P39 separately tests fresh-process pickle and
joblib with rare, missing, unseen and declared-but-unobserved categories for all
three tasks. Cross-version and device migration remain unsupported guarantees.
Pin the full environment and keep original export metadata with the artifact;
P40 additionally records private version-at-fit provenance; older artifacts
without it remain unknown. Never load untrusted
pickle/joblib files.

For Pipeline, DataFrame-preserving `ColumnTransformer`, CV and grid-search,
use the existing `examples/pipeline_workflow.py`. Keep external `eval_set`
already transformed consistently if using preprocessing: ordinary pipeline
parameter passing does not transform that set automatically.

At prediction, new feature categories are accepted as unknown; missing or extra
column names are errors. New target classes in a classification `eval_set` are
an error. `input_schema_` describes roles/counts, not an editable vocabulary.

## Refit and parameter changes

`model.set_params(hidden_dim=12)` changes the settings for the next fit. An
already fitted model keeps its current network, encoder and resolved device
until a successful `fit`. `get_params` and repr show pending settings, so they
are not a historical training receipt. Prediction keeps using the fitted state;
set_params does not move a saved CUDA model to CPU. Clone copies settings and
is unfitted. A successful refit can learn new data/schema; a failed refit leaves
the previous fitted snapshot and predictions usable, with the new pending
settings still in place. Do not modify fitted attributes to reconfigure a model.

Python/NumPy/PyTorch CPU RNG states are restored after fit. This does not make
same-process concurrent fits thread-safe because RNG handling is process-global.
Use isolated processes when fitting concurrently and account for resource use.

## Persistence versions and warnings

Use the same pinned CPU runtime/source build for supported round-trips.
Patch upgrades are best effort; minor/major, dependency and device migrations
need explicit validation. Keep a trusted probe frame and original outputs with
environment/source hashes. Public 0.3.1 artifacts lack the later input_schema_
and P40 internal version metadata; loading does not fabricate them. Refit under
new code only when intended, with original training data.

Schema failures raise existing standard exceptions; ordinary column reordering,
missing-value handling and unseen feature categories do not warn. Explicit CUDA
fails if unavailable; auto compatibility failure warns with a recorded fallback.
No current parameter is deprecated, and P40's tested internal FutureWarning
helper is dormant. Physical NVIDIA CUDA validation remains optional and pending.

## Reporting and recovering from mistakes

Use the [minimal reproducer guide](support/MINIMAL_REPRODUCER_GUIDE.md) for a
small synthetic example, safe environment report and recovery table. Run
`python examples/support_environment.py` from a checkout to print only version,
OS family and CUDA availability fields; it does not choose or probe a device.
An import failure should be reported with the traceback after reviewing it for
personal paths. Do not upload private frames, model files, tokens or full
environment-variable listings.


## Canonical maintenance and stable scope

Current behavior above is unchanged. See [the policy index](policies/README.md),
[1.0 API boundary](API_BOUNDARY.md) and [support scope](policies/SUPPORT_SCOPE.md).
The reviewed boundary is the stable 1.0 public contract; experimental
routing/schema/timing/optional mechanisms remain explicitly bounded outside that
guarantee.

## Target validation contract

Classification supports only single-output sklearn binary/multiclass targets.
Fractional continuous targets and missing/non-finite numeric labels raise
ValueError consistently in fit, eval_set and score. Regression keeps finite real
single-output targets. See [target contract](TARGET_CONTRACT.md), including the
existing zero-weight row and validation-class rules.

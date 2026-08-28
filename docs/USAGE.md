# NeuroTabular 0.2.0 usage guide

## Minimal classification

```python
from neurotabular import NeuroTabularClassifier

model = NeuroTabularClassifier(random_state=42)
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
prediction = model.predict(X_test)
```

`X_train` and `X_test` must be pandas DataFrames. Targets may be NumPy arrays,
pandas Series, or other one-dimensional array-like objects with exactly two
classes.

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

With external validation, preprocessing is fitted on all `X_train` rows and
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

Disable the frequency side channel for an ablation or strict 0.1.x-style input:

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
At least one weight must be positive.

For inverse-frequency binary class weights:

```python
model = NeuroTabularClassifier(class_weight="balanced")
model.fit(X_train, y_train, sample_weight=row_weights)
```

When both mechanisms are used, per-class and per-row weights multiply.

## Cross-validation and pipelines

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
batch shuffling. Exact cross-hardware floating-point identity is not promised,
especially across CUDA architectures or library versions.

## Diagnostics

```python
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

The prediction frame may present fitted columns in another order. NeuroTabular
reorders them to the fitted schema. Missing or extra columns raise an error:

```python
model.predict_proba(X_test[model.feature_names_in_[::-1]])  # accepted
model.predict_proba(X_test.drop(columns=["income"]))  # rejected
```

This strictness prevents silent feature misalignment.

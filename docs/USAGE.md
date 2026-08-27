# NeuroTabular 0.1.0 usage guide

This guide shows common binary-classification workflows. See
[API.md](API.md) for exact parameter validation and fitted attributes.

## Minimal usage

```python
from neurotabular import NeuroTabularClassifier

model = NeuroTabularClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)
```

`X_train` and `X_test` must be pandas DataFrames.

## Evaluate ROC-AUC

```python
from sklearn.metrics import roc_auc_score
from neurotabular import NeuroTabularClassifier

model = NeuroTabularClassifier(eval_metric="roc_auc")
model.fit(X_train, y_train)

score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(score)
```

## Cross-validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from neurotabular import NeuroTabularClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    NeuroTabularClassifier(random_state=42),
    X,
    y,
    cv=cv,
    scoring="roc_auc",
)
```

Each fold fits a separate neural network and creates its own internal
validation split. For fast experiments, reduce `max_epochs` explicitly.

## Explicit validation

```python
model = NeuroTabularClassifier(
    eval_metric="roc_auc",
    patience=4,
    min_delta=1e-4,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
```

No additional split is created. Statistics and vocabularies remain fitted on
`X_train` only.

## Numerical NaNs

```python
import numpy as np
import pandas as pd

X = pd.DataFrame(
    {
        "age": [25.0, 41.0, np.nan, 33.0],
        "income": [32_000.0, np.nan, 51_000.0, 44_000.0],
    }
)
```

Manual imputation and scaling are unnecessary. Training medians fill missing
values, standard scaling is fitted on training rows, and a missing indicator is
added for every numerical column.

## Categorical features and missing values

```python
X = pd.DataFrame(
    {
        "age": [25, 41, 33, 29],
        "city": ["Rome", "Milan", None, "Rome"],
        "member": [True, False, True, False],
    }
)

model = NeuroTabularClassifier()
model.fit(X, y)
```

Strings and booleans are detected automatically. No one-hot encoder is needed.

## Integer categorical IDs

Integer columns are numerical by default. Mark IDs explicitly:

```python
model = NeuroTabularClassifier(categorical_features=["store_id", "postal_code"])
```

Explicit columns are added to automatic string/category/boolean detection.

## Unknown and rare categories

An inference category absent during fitting maps to the unknown embedding ID.
It does not cause an error. A training category seen fewer than
`min_category_count` times maps to the rare ID.

```python
model = NeuroTabularClassifier(min_category_count=3)
```

Set `min_category_count=1` to give every observed training category its own ID.
Extremely high-cardinality identifiers can still require domain-specific
feature design.

## Sample weights

```python
model.fit(X_train, y_train, sample_weight=row_weights)
```

Weights must be finite, non-negative, and include at least one positive value.

## Balanced classes

```python
model = NeuroTabularClassifier(class_weight="balanced")
model.fit(X_train, y_train)
```

Balanced class weights and row weights can be combined:

```python
model.fit(X_train, y_train, sample_weight=row_weights)
```

## CPU and CUDA

Automatic selection:

```python
model = NeuroTabularClassifier(device="auto")
```

Force CPU:

```python
model = NeuroTabularClassifier(device="cpu")
```

Request CUDA:

```python
model = NeuroTabularClassifier(device="cuda")
```

An unavailable requested CUDA device raises an error. Check the resolved path
after fitting with `model.device_`.

## Automatic and explicit batches

The default chooses full or large batches from dataset width, row count, device,
and CUDA memory:

```python
model = NeuroTabularClassifier(batch_size="auto")
```

Override it when a controlled memory bound is needed:

```python
model = NeuroTabularClassifier(batch_size=512)
```

The resolved training value is `model.batch_size_`.

## Fast experiments

```python
model = NeuroTabularClassifier(
    hidden_dim=32,
    n_blocks=1,
    max_epochs=8,
    patience=2,
    random_state=42,
)
```

These settings reduce capacity and training budget; they are not universal
accuracy defaults.

## Reproducibility

```python
model = NeuroTabularClassifier(random_state=42, device="cpu")
```

Use the same seed for train/test or cross-validation splitters. Record Python,
NumPy, pandas, PyTorch, scikit-learn, and hardware versions for reproducible
experiments. CUDA results may vary across GPU stacks even with the same seed.

## Inspect training and profiling

```python
print(model.best_epoch_)
print(model.best_score_)
print(model.n_iter_)
print(model.n_parameters_)
print(model.fit_time_)
print(model.preprocessing_time_)
print(model.training_time_)
print(model.validation_time_)
print(model.history_)
```

Fine-grained measured phases are in `model.profile_`. CPU operation timings are
the reliable release path. CUDA fine-grained timings intentionally avoid a
synchronization after every operation and are therefore approximate.

## Prediction schema

Prediction columns may be reordered:

```python
proba = model.predict_proba(X_test[model.feature_names_in_[::-1]])
```

Columns must otherwise match training exactly. Missing, extra, and duplicate
columns raise errors.

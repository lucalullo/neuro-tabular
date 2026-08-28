# NeuroTabular 0.2.0 API reference

The supported public package surface is:

```python
from neurotabular import NeuroTabularClassifier, __version__
```

`TabularPreprocessor`, `TabularNetwork`, and training helpers are implementation
modules. Their signatures may change without a public deprecation cycle.

## `NeuroTabularClassifier`

```python
NeuroTabularClassifier(
    hidden_dim=64,
    n_blocks=2,
    dropout=0.1,
    lr=0.003,
    weight_decay=1e-5,
    batch_size="auto",
    max_epochs=30,
    validation_fraction=0.2,
    patience=4,
    min_delta=1e-4,
    eval_frequency=1,
    eval_metric="loss",
    class_weight=None,
    categorical_features=None,
    min_category_count=2,
    numerical_embedding="scalar",
    use_category_frequency=True,
    feature_gating=False,
    full_data_refit=False,
    device="auto",
    random_state=42,
    verbose=0,
)
```

The constructor is side-effect free and stores every argument verbatim, as
required by scikit-learn cloning and parameter inspection.

## Constructor parameters

### Architecture

`hidden_dim` is the positive width of the input projection and each residual
block. `n_blocks` is the positive number of residual blocks. `dropout` is a
probability in `[0, 1)` applied inside residual blocks.

`numerical_embedding` selects the numerical representation:

- `"scalar"`: scaled value and missing indicator; the release default;
- `"affine"`: a learned per-feature affine expansion;
- `"periodic"`: learned projection of sinusoidal features;
- `"piecewise"`: a piecewise-linear basis over training-only quantile knots.

All modes preserve a distinct missingness signal. Quantile knots are fitted
only on training rows. The non-scalar modes are supported for controlled
experiments but were not selected as release defaults.

`use_category_frequency=True` appends one numerical log-frequency feature for
each categorical column. Frequencies, including aggregate rare-bucket
frequency, are learned from training rows only; unknown values receive zero.

`feature_gating=True` replaces the plain initial projection with a lightweight
learned value/gate projection. It increases model complexity and is disabled by
default because the release ablation did not show a robust gain.

Categorical embedding dimensions are calculated automatically. They are
bounded and adapt to training sample count, cardinality, number of categorical
features, and a compact aggregate embedding budget. The learned dimensions are
available as `embedding_dimensions_` after fitting.

### Optimization

`lr` is the positive initial AdamW learning rate. `weight_decay` is a
non-negative AdamW decay. Training uses a cosine schedule.

`batch_size` accepts a positive integer or `"auto"`. Automatic batching uses
sample count, processed input width, categorical cardinalities, hidden width,
block count, and selected device. The resolved value is `batch_size_`.

`max_epochs` is a positive integer. `patience` is the positive number of
validation checks without a qualifying improvement. `min_delta` is the
non-negative absolute improvement threshold. `eval_frequency` is the positive
number of epochs between checks; the first and final epochs are always checked.

`eval_metric` accepts `"loss"`, `"roc_auc"`, or `"accuracy"`. Loss is minimized;
ROC-AUC and accuracy are maximized. The best model state is restored.

`full_data_refit=True` uses the internally selected `best_epoch_` to train a new
model on all supplied rows. It is skipped when `eval_set` is supplied because
external validation already leaves all primary training rows available. The
default is `False`: the release ablation found a substantial fit-time increase
and lower mean ROC-AUC for the tested workloads.

### Data and imbalance

`categorical_features` is `None` or a sequence of unique DataFrame column
labels. Named columns are forced categorical in addition to automatic object,
string, pandas categorical, and boolean detection. This is required for
integer-coded categories.

`min_category_count` is a positive integer. Categories seen fewer times use a
shared rare ID. Missing, unknown, and rare IDs are distinct.

`class_weight` accepts `None` or `"balanced"`. Balanced weights are calculated
from the training target. They multiply any `sample_weight` passed to `fit`.

### Validation, device, and reproducibility

`validation_fraction` must be in `(0, 1)` and controls the internal stratified
split when `eval_set` is absent.

`device` accepts `"auto"`, `"cpu"`, `"cuda"`, or an indexed CUDA string such as
`"cuda:1"`. Automatic CUDA selection requires a successful synchronized kernel
probe and otherwise warns before falling back to CPU. Explicit CUDA failures
raise `RuntimeError` with diagnostic metadata.

AMP is enabled only after a compatible CUDA probe. When AMP is disabled,
including all CPU runs, training uses `contextlib.nullcontext`; no
`torch.autocast` context is constructed. This behavior is regression-tested for
PyTorch 2.0 compatibility.

`random_state` is an integer seed used for Python, NumPy, PyTorch, validation
splitting, and batch order. `verbose` accepts `0` or `1`.

## `fit`

```python
model.fit(X, y, sample_weight=None, eval_set=None)
```

`X` must be a non-empty pandas DataFrame with unique column names. `y` must
contain exactly two mutually comparable classes and have the same length.

`sample_weight`, when supplied, is a finite one-dimensional non-negative array
with at least one positive value. `eval_set` is either `None` or `(X_valid,
y_valid)`. Validation weights are not accepted separately.

The method returns `self`. All preprocessing state is learned from primary
training rows only. With internal validation this means only the training side
of the split; with external validation it means all rows in `X`, never the
validation frame.

## `predict_proba`

```python
probabilities = model.predict_proba(X)
```

Returns a finite `float` array of shape `(n_samples, 2)`. Columns follow
`classes_`; each row sums to one. Prediction validates the DataFrame schema and
uses batched inference.

## `predict`

```python
labels = model.predict(X)
```

Returns original target labels using a 0.5 positive-class threshold.

## Fitted attributes

- `classes_`: the two original target labels in sorted order;
- `n_features_in_`: number of input columns;
- `feature_names_in_`: string column names, when every name is a string;
- `numeric_features_`, `categorical_features_`: inferred/declared partitions;
- `embedding_dimensions_`: categorical embedding width per categorical column;
- `device_`, `device_info_`: resolved device and diagnostic metadata;
- `batch_size_`, `inference_batch_size_`: resolved training/inference batches;
- `n_parameters_`: trainable network parameter count;
- `best_epoch_`, `best_score_`, `best_validation_loss_`, `n_iter_`;
- `history_`: one dictionary per validation check;
- `full_data_refit_`: whether the optional refit actually ran;
- `preprocessing_time_`, `training_time_`, `validation_time_`, `fit_time_`;
- `last_prediction_time_`: most recent end-to-end prediction duration;
- `profile_`: nested preprocessing, transfer, training, validation, optimizer,
  checkpoint, device, and optional refit timings.

Timing attributes use `time.perf_counter` seconds and describe the current host
and workload; they are diagnostics, not service-level guarantees.

## Errors and warnings

The estimator raises clear `TypeError` or `ValueError` exceptions for invalid
hyperparameters, targets, weights, frames, non-finite values, or schema changes.
An automatic CUDA compatibility failure emits `RuntimeWarning` and records the
reason in `device_info_`; an explicit CUDA request raises instead.

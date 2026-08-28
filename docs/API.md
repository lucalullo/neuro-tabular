# NeuroTabular 0.1.1 API reference

NeuroTabular 0.1.1 exposes one public estimator:

```python
from neurotabular import NeuroTabularClassifier
```

It supports binary classification with pandas `DataFrame` inputs.

## Constructor

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
    device="auto",
    random_state=42,
    verbose=0,
)
```

The constructor stores parameters only. It does not inspect data, initialize a
network, access a GPU, or change PyTorch settings.

## Parameters

### `hidden_dim=64`

- Type: positive integer; booleans are rejected.
- Purpose: width of the input projection and residual representation.
- Effect: larger values increase capacity, parameter count, activation memory,
  and compute.

### `n_blocks=2`

- Type: positive integer; booleans are rejected.
- Purpose: number of residual feed-forward blocks.
- Each block uses pre-LayerNorm, a two-times-width feed-forward branch, SiLU,
  dropout, and a residual connection.

### `dropout=0.1`

- Type: finite real number in `[0, 1)`.
- Purpose: dropout probability inside residual branches.

### `lr=0.003`

- Type: finite positive real number.
- Purpose: initial AdamW learning rate.
- The internal scheduler applies cosine decay to 5% of the initial learning
  rate over `max_epochs`.

### `weight_decay=1e-5`

- Type: finite non-negative real number.
- Purpose: AdamW weight decay.

### `batch_size="auto"`

- Accepted values: the string `"auto"` or a positive integer.
- `"auto"` uses a deterministic heuristic based on training rows, processed
  width, embedding width, hidden width, residual depth, device, and available
  CUDA memory.
- Datasets with at most 2,048 training rows use a full batch. Larger CPU
  datasets use conservative batches between 2,048 and 8,192 according to
  width. CUDA uses a memory-limited power-of-two batch capped at 65,536.
- An explicit size is capped at the number of training rows.
- The resolved value is exposed as `batch_size_`.

### `max_epochs=30`

- Type: positive integer.
- Purpose: upper bound on training epochs; early stopping may finish sooner.

### `validation_fraction=0.2`

- Type: finite real number strictly between 0 and 1.
- Purpose: fraction used for the internal stratified split when `eval_set` is
  absent.
- Ignored when an explicit validation set is supplied.

### `patience=4`

- Type: positive integer.
- Purpose: number of consecutive validation checks without a significant
  improvement before stopping.
- Patience counts checks, not necessarily epochs when `eval_frequency > 1`.

### `min_delta=1e-4`

- Type: finite non-negative real number.
- Purpose: minimum absolute improvement required to update the checkpoint and
  reset patience.
- For loss, a score must be lower than `best_score - min_delta`. For ROC-AUC or
  accuracy, it must be higher than `best_score + min_delta`.

### `eval_frequency=1`

- Type: positive integer.
- Purpose: validation interval in epochs.
- The first epoch, every divisible epoch, and the final epoch are evaluated.

### `eval_metric="loss"`

- Accepted values: `"loss"`, `"roc_auc"`, or `"accuracy"`.
- Purpose: early-stopping score and best-checkpoint selection.
- The training objective is always binary cross-entropy with logits.
- ROC-AUC requires both target classes in validation.

### `class_weight=None`

- Accepted values: `None` or `"balanced"`.
- `"balanced"` computes inverse-frequency weights from the training subset
  only.
- Class weights multiply `sample_weight` when both are supplied.
- The resolved mapping is exposed as `class_weight_`; it is `None` when class
  weighting is disabled.

### `categorical_features=None`

- Type: `None` or an iterable of unique existing column names.
- Explicit columns are added to automatic categorical detection; they do not
  replace it.
- Use this parameter for integer IDs or codes that should use embeddings.
- Unknown names, duplicate entries, strings passed as the entire iterable, and
  non-iterable values are rejected.

Automatic detection covers object, pandas string, pandas category, and boolean
dtypes. Other columns are treated as numerical.

### `min_category_count=2`

- Type: positive integer.
- Purpose: minimum training frequency for a category to receive its own ID.
- Less frequent observed values use the rare ID `2`.
- Missing ID `0` and unseen ID `1` remain separate.
- Set to `1` to disable rare bucketing.

### `device="auto"`

- Type: string.
- Accepted forms: `"auto"`, `"cpu"`, `"cuda"`, or an indexed CUDA string such
  as `"cuda:0"`.
- `"auto"` uses CPU when CUDA is unavailable. When CUDA is visible, it checks
  the device index, collects available GPU/build metadata, launches a minimal
  CUDA kernel, and synchronizes it. CUDA is selected only if that probe succeeds.
- If `"auto"` sees a GPU but the installed PyTorch build cannot execute the
  probe, fitting emits `RuntimeWarning`, records the reason, and continues on CPU.
- `"cpu"` selects CPU directly and performs no CUDA query or probe.
- `"cuda"` requests the current CUDA device; `"cuda:N"` requests index `N`.
  Explicit CUDA never falls back. Unavailable, nonexistent, or probe-failing
  devices raise `RuntimeError` before network creation, CUDA training tensor
  creation, AMP/GradScaler setup, or optimizer setup.
- Explicit errors include the requested device and, when available, GPU name,
  `sm_XY` capability, PyTorch/CUDA versions, compiled architectures, device
  count, and original probe error.
- The resolved device is exposed as `device_`; successful CUDA resolution is
  indexed, such as `"cuda:0"`.
- Verified CUDA devices with compute capability 7.0 or newer use float16
  autocast plus gradient scaling. Older or metadata-unknown verified devices
  use FP32. CPU never enables AMP by default.

### `random_state=42`

- Type: integer; booleans are rejected.
- Seeds Python, NumPy, the PyTorch CPU generator, the validation split, and
  batch shuffling. CUDA generators are seeded only after a CUDA device passes
  the compatibility probe.
- CPU execution is reproducible for the same inputs and software environment in
  the tested path. Exact CUDA results can depend on the GPU stack and kernels.

### `verbose=0`

- Accepted values: `0` or `1`.
- `0` is silent.
- `1` prints one line per validation check and a final best-epoch summary.

## `fit`

```python
model.fit(
    X,
    y,
    sample_weight=None,
    eval_set=None,
)
```

Fits the estimator and returns `self`.

### `X`

A pandas `DataFrame` with at least one row, at least one feature, and unique
column names. Fitting stores the exact schema. Numerical infinity is rejected;
numerical and categorical missing values are supported.

### `y`

A one-dimensional target with the same number of rows as `X`, exactly two
mutually comparable classes, and no missing values. Original labels, including
strings, are preserved in `classes_` and returned by `predict`.

### `sample_weight=None`

An optional one-dimensional numeric array with one value per training row.
Values must be finite, non-negative, and include at least one positive value.

With internal validation, weights are split with the corresponding rows and
used for training and validation metrics. With explicit validation,
`sample_weight` applies to training rows only because 0.1.1 does not expose an
explicit validation-weight argument. ROC-AUC additionally requires positive
validation weight for both classes.

### `eval_set=None`

An optional single `(X_valid, y_valid)` pair. When present:

- no additional internal split is created;
- preprocessing statistics and vocabularies are fitted on `X` only;
- validation must match the training feature schema;
- validation labels must belong to the two training classes; and
- unseen validation categories use the unknown ID.

## `predict`

```python
pred = model.predict(X)
```

Returns a one-dimensional array of original class labels. The second entry in
`classes_` is selected when its probability is at least 0.5. Calling before
`fit` raises `NotFittedError`.

Prediction accepts the training columns in any order, restores their order,
and rejects missing, extra, or duplicate columns.

## `predict_proba`

```python
proba = model.predict_proba(X)
```

Returns a finite float array with shape `(n_samples, 2)`. Columns follow
`classes_`; the positive-class probability is `proba[:, 1]`. Inference uses
`torch.inference_mode()` and batches of `inference_batch_size_`.

## Fitted public attributes

- `classes_`: sorted original binary target classes.
- `n_features_in_`: number of fitted feature columns.
- `feature_names_in_`: object array of feature names when all names are strings.
- `numeric_features_`: columns treated as numerical.
- `categorical_features_`: automatically detected plus explicit categoricals.
- `class_weight_`: resolved balanced mapping or `None`.
- `device_`: resolved device string.
- `device_info_`: device decision diagnostics with `requested_device`,
  `resolved_device`, `fallback_used`, PyTorch/CUDA build information, and GPU
  metadata when available. `fallback_reason` and `probe_error` appear when
  relevant.
- `batch_size_`: resolved training batch size.
- `inference_batch_size_`: prediction batch bound.
- `n_parameters_`: trainable neural parameter count.
- `best_epoch_`: one-based epoch of the selected checkpoint.
- `best_score_`: selected `eval_metric` value.
- `best_validation_loss_`: validation loss at `best_epoch_`.
- `n_iter_`: total epochs executed.
- `history_`: list of evaluated epoch records containing `epoch`, `train_loss`,
  `validation_loss`, `validation_score`, and `learning_rate`.
- `fit_time_`: total wall-clock fit time in seconds.
- `preprocessing_time_`: fit and transform preprocessing time in seconds.
- `training_time_`: measured training compute time in seconds.
- `validation_time_`: cumulative validation wall time in seconds.
- `profile_`: nested preprocessing and training phase timings. Fine-grained
  operation timings are reliable on the measured CPU path; CUDA operation
  timings are marked as approximate to avoid synchronization in every batch.
  It also includes target-preparation timing and device diagnostics.
- `last_prediction_time_`: wall-clock time of the most recent `predict` or
  `predict_proba` preprocessing-plus-inference call; available after prediction.

## Neural architecture and objective

```text
scaled numerical values + missing indicators
                 +
per-feature categorical embeddings
                 |
                 v
            concatenate
                 |
                 v
     Linear -> SiLU input projection
                 |
                 v
 [LayerNorm -> Linear(2x) -> SiLU -> Dropout
  -> Linear -> Dropout -> residual add] x n_blocks
                 |
                 v
          Linear -> one logit
```

Training uses `BCEWithLogitsLoss`, AdamW, and cosine learning-rate decay.
AdamW leaves implementation selection to the supported PyTorch default instead
of forcing the newer fused implementation.
Validation and prediction use `torch.inference_mode()`. `torch.compile` is not
enabled in 0.1.1.

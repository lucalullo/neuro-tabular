# NeuroTabular 0.3 architecture

NeuroTabular 0.3 uses one shared tabular pipeline with task-specific target and
output semantics.

```mermaid
flowchart LR
    X[DataFrame and effective rows] --> P[PREPROCESSOR]
    P --> B[SHARED BACKBONE]
    B --> H[TASK HEAD]
    H --> L[TASK LOSS / METRICS]
    L --> C[Best validation checkpoint]
    C --> O[Task prediction and target decoding]
```

## Shared estimator boundary

`_base.py` owns the transactional fit lifecycle, feature-schema checks,
preprocessing, splitting, sample weights, device selection, training/refit
orchestration, and batched inference. `NeuroTabularClassifier` and
`NeuroTabularRegressor` are sibling scikit-learn estimators with task-specific
target and prediction semantics.

`TabularPreprocessor` is task-neutral. It learns numerical statistics,
missingness signals, categorical vocabularies, rare buckets, optional numerical
representations, and frequency side features from training rows only.

`TabularNetwork` provides the shared compact residual MLP backbone. A task spec
selects the output width, loss, prediction transform, metric semantics, and
checkpoint direction while leaving the shared training engine unchanged.

## Task semantics

| Task | Output | Training loss | Prediction |
| --- | --- | --- | --- |
| Binary classification | one logit | weighted BCE with logits | sigmoid, then class mapping |
| Multiclass classification | `n_classes` logits | weighted cross entropy | softmax / argmax |
| Regression | one scalar | weighted MSE on standardized target | identity, then inverse target transform |

Binary keeps the one-logit formulation rather than switching to two logits. The
0.3 refactor was validated against frozen 0.2 binary behavior before multiclass
and regression were added.

Regression standardizes the target using weighted train-only population moments
so one fixed optimization recipe can operate across target scales. Prediction
restores the original units. Constant targets are handled explicitly.

## Training and checkpointing

The shared trainer uses AdamW, cosine scheduling, deterministic batching,
validation, early stopping, strict best-checkpoint restoration, and optional
full-data refit. Classification uses stratified internal validation; regression
uses a deterministic shuffled split without stratification.

Rows with zero effective sample weight are removed before target discovery,
splitting, and preprocessing. Failed refits are transactional: an already fitted
estimator keeps its previous valid state.

## Device and persistence behavior

Device selection performs an actual CUDA compatibility probe. Automatic mode
falls back to CPU with diagnostics when CUDA is visible but unusable; explicit
CUDA requests fail rather than silently changing device. AMP is enabled only on
compatible CUDA hardware.

Pickle/joblib persistence includes preprocessing state, target encoding or
target statistics, device metadata, and network parameters. Round trips are
tested in fresh processes for all three tasks. Cross-version or cross-device
compatibility is not guaranteed.

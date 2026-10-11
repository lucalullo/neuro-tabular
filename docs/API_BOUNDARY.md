# Stable and experimental API boundary for NeuroTabular 1.0.0

FREEZE_SAFE defines the reviewed stable scope; EXPERIMENTAL and INTERNAL retain
their explicit limits. Publication remains a separate maintainer decision.
RC1 target hardening is retained; valid-input training/prediction math and
constructor defaults are unchanged.

## Export and method contract

| Symbol | Classification | Exact scope / evidence |
| --- | --- | --- |
| NeuroTabularClassifier | FREEZE_SAFE | DataFrame-only binary/multiclass; sklearn binary/multiclass single-output finite labels, sorted original labels, same self from fit; classifier/Pipeline/clone/persistence tests. |
| NeuroTabularRegressor | FREEZE_SAFE | Separate single-output finite real target, original-unit float64 predictions; no predict_proba/class_weight parameter. |
| __version__ | FREEZE_SAFE | Version string matches package metadata; unpublished same-version builds require source identity. |
| fit(X,y,sample_weight=None,eval_set=None) | FREEZE_SAFE | Positional 1-D y/weights; task target validation shared with eval_set/score; nonnegative finite weights with positive total, zero rows excluded; train-only preprocessing; new eval target classes rejected; transactional committed state. |
| predict(X) | FREEZE_SAFE | Shape(n,), classifier original-label dtype/threshold0.5 or argmax; regression float64 original units/constant path; schema names aligned, row order preserved. |
| predict_proba(X), classifier only | FREEZE_SAFE | float64(n,C), sorted classes_ columns, normalized finite probabilities; no calibration guarantee. |
| score(X,y,sample_weight=None) | FREEZE_SAFE | Sklearn accuracy/R2 with existing weights, after the shared task target validation; signature unchanged. |
| get_params(deep=True), set_params(**params), clone conventions | FREEZE_SAFE | U below; invalid names ValueError, clone unfitted. Exact repr pretty-print not frozen. |
| get_metadata_routing, set_fit_request, set_score_request | EXPERIMENTAL | Installed sklearn behavior; no general routing or automatic external eval_set transformation. |
| Private modules/underscore methods/network/encoder/optimizer/tensors/sklearn implementation tags | INTERNAL | Importability is not a public stability promise. |
| _fitted_neurotabular_version_, _warn_deprecated/_warnings | INTERNAL | Diagnostic fit provenance/dormant notice infrastructure; neither a loader nor migration guarantee. |
| Examples and support helper | INTERNAL to estimator API | Maintained examples, not additional package exports. |

## Constructor contract

U applies independently to **every row**: stored verbatim in __init__, returned
by get_params, copied by unfitted clone; set_params changes pending next-fit
configuration, keeps the fitted predictor unchanged, and fit applies it. Failed
refit preserves the old fitted state, not earlier user parameter edits. Domain
validation occurs at fit; no eager coercion/retrain. Finite real/integer domains
use existing validators; this is no new promise about accidental bool aliases.
No names removed/renamed; no redundant knob established by current evidence.

| Name | Documented domain | Current default | Behavior | clone/get/set | Classification |
| --- | --- | --- | --- | --- | --- |
| `hidden_dim` | positive integer | `64` | Projected/residual width | U | FREEZE_SAFE |
| `n_blocks` | positive integer | `2` | Residual block count | U | FREEZE_SAFE |
| `dropout` | finite real [0,1) | `0.1` | Training dropout probability | U | FREEZE_SAFE |
| `lr` | finite positive real | `0.003` | Initial AdamW cosine-schedule rate | U | FREEZE_SAFE |
| `weight_decay` | finite nonnegative real | `1e-05` | AdamW decay | U | FREEZE_SAFE |
| `batch_size` | positive integer or auto | `'auto'` | Explicit training batch or current automatic resolution; heuristic not frozen | U | FREEZE_SAFE |
| `max_epochs` | positive integer | `30` | Training epoch limit | U | FREEZE_SAFE |
| `validation_fraction` | finite real (0,1) | `0.2` | Internal stratified classifier/random regression split; external eval_set bypasses split | U | FREEZE_SAFE |
| `patience` | positive integer | `4` | Validation checks without min_delta improvement | U | FREEZE_SAFE |
| `min_delta` | finite nonnegative real | `0.0001` | Patience reset threshold; strict best checkpoint still restored | U | FREEZE_SAFE |
| `eval_frequency` | positive integer | `1` | Validation interval, with existing first/final checks | U | FREEZE_SAFE |
| `eval_metric` | classifier loss/accuracy/binary roc_auc; regressor loss/rmse/mae/r2 | `'loss'` | Existing task-specific metric direction/scale | U | FREEZE_SAFE |
| `class_weight` | None or balanced; classifier only | `None` | Train-only class weighting combined with sample weights | U | FREEZE_SAFE |
| `categorical_features` | None or sequence of valid column labels | `None` | Additional explicit categories; automatic detection retained | U | FREEZE_SAFE |
| `min_category_count` | positive integer | `2` | Train-only rare-value threshold | U | FREEZE_SAFE |
| `numerical_embedding` | scalar, affine, periodic, piecewise | `'scalar'` | Scalar/default FREEZE_SAFE; non-scalar values EXPERIMENTAL | U | FREEZE_SAFE scalar / EXPERIMENTAL non-scalar |
| `use_category_frequency` | bool | `True` | Existing train-only log-frequency side features | U | FREEZE_SAFE |
| `feature_gating` | bool | `False` | Opt-in initial projection gate, default off | U | EXPERIMENTAL |
| `full_data_refit` | bool | `False` | Fresh selected-epoch final fit after internal split; skipped for external eval_set | U | FREEZE_SAFE |
| `device` | auto, cpu, cuda, cuda:N | `'auto'` | Selection/fallback interface; physical CUDA remains optional best effort | U | FREEZE_SAFE |
| `random_state` | integer accepted by NumPy seed (0..2**32-1), excluding bool | `42` | Fixed-runtime split/initialization/shuffle and caller RNG restoration | U | FREEZE_SAFE |
| `verbose` | 0 or 1 | `0` | Progress control; exact console text not frozen | U | FREEZE_SAFE |

Classifier has all 22 rows; regressor has 21, omitting class_weight. Numerical
embedding's parameter/default scalar interface stays stable; non-scalar behavior
is explicitly experimental. device selection interface is stable; actual CUDA
hardware/portability remains outside validated support. Exact signatures are in
the tested constructor inventory and docs/API.md.

## Fitted attribute contract

L applies to every row unless its condition narrows presence: absent before
successful fit/on clones; created/replaced by successful fit; preserved on failed
refit; successful new-schema refit replaces all fitted observations. M: all
read-only by convention; mutable arrays/dicts/lists and scalar assignments are
not configuration interfaces, some live fields affect prediction if edited.
S: preserved by trusted same-build/pinned CPU pickle/joblib round-trip when
present; never implies cross-version/device/dependency portability. These
common lifetime/mutability/serialization rules apply to **each listed attribute**.

| Attribute | Type | Presence / lifetime L | Mutability | Serialization | Classification | Meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `classes_` | ndarray of original comparable labels | classifier; L | M | S | FREEZE_SAFE | Sorted probability-column/label order; do not mutate |
| `n_classes_` | int | classifier; L | M | S | FREEZE_SAFE | Class count |
| `class_weight_` | None or dict | classifier; L | M | S | FREEZE_SAFE | Fitted class-weight observation |
| `n_features_in_` | int | both; L | M | S | FREEZE_SAFE | Committed feature count |
| `feature_names_in_` | ndarray | both, all labels strings only; L | M | S | FREEZE_SAFE | Original fitted column order; removed on successful non-string refit |
| `numeric_features_` | list of column labels | both; L | M | S | FREEZE_SAFE | Committed numeric roles |
| `categorical_features_` | list of column labels | both; L | M | S | FREEZE_SAFE | Committed categorical roles |
| `embedding_dimensions_` | list of int | both; L | M | S | EXPERIMENTAL | Allocation heuristic |
| `device_` | str | both; L | M | S | FREEZE_SAFE | Resolved committed device; assignment is not a supported migration |
| `device_info_` | dict | both; L | M | S | EXPERIMENTAL | Probe/resource diagnostic subkeys |
| `batch_size_` | int | both; L | M | S | FREEZE_SAFE | Resolved training batch observation |
| `inference_batch_size_` | int | both; L | M | S | EXPERIMENTAL | Engineering batching state |
| `n_parameters_` | int | both; L | M | S | FREEZE_SAFE | Fitted network parameter count |
| `best_epoch_` | int | both; L | M | S | FREEZE_SAFE | Selected checkpoint epoch, existing numbering |
| `best_score_` | floating scalar | both; L | M | S | FREEZE_SAFE | Selected eval_metric score with documented task scale/direction |
| `best_validation_loss_` | floating scalar | both; L | M | S | FREEZE_SAFE | Task validation loss |
| `n_iter_` | int | both; L | M | S | FREEZE_SAFE | Selection epochs, excludes extra full_data_refit |
| `n_effective_samples_` | int | both; L | M | S | FREEZE_SAFE | Positive-weight primary rows before internal split |
| `n_ignored_samples_` | int | both; L | M | S | FREEZE_SAFE | Excluded zero-weight primary rows |
| `history_` | list of dict | both; L | M | S | FREEZE_SAFE documented fields | Validation checkpoint metrics only; unspecified new keys not frozen |
| `full_data_refit_` | bool | both; L | M | S | FREEZE_SAFE | Whether final fresh refit occurred |
| `preprocessing_time_` | float seconds | both; L | M | S | EXPERIMENTAL | Timing observation |
| `training_time_` | float seconds | both; L | M | S | EXPERIMENTAL | Training/profile aggregate, not inclusive fit |
| `validation_time_` | float seconds | both; L | M | S | EXPERIMENTAL | Timing observation |
| `fit_time_` | float seconds | both; L | M | S | EXPERIMENTAL | Inclusive successful fit |
| `last_prediction_time_` | float seconds | after prediction only; L | M | S | EXPERIMENTAL | Last inference duration |
| `profile_` | nested dict | both; L | M | S | EXPERIMENTAL | Detailed clocks/resources |
| `input_schema_` | tuple of detached dict | new fits only, both; L | M | S | EXPERIMENTAL | Observational format; old public models lack it; not prediction input |
| `target_mean_` | floating scalar | regressor; L | M | S | FREEZE_SAFE | Weighted final-training target mean |
| `target_scale_` | floating scalar | regressor; L | M | S | FREEZE_SAFE | Weighted population scale, 1 for constant target |
| `target_is_constant_` | bool | regressor; L | M | S | FREEZE_SAFE | Exact constant-target path |


## Warnings and errors

FREEZE_SAFE: NotFittedError before inference validation; TypeError for wrong
container/type paths; ValueError for existing invalid values/schema/target/
weights; RuntimeError for explicit unavailable/incompatible CUDA. Freeze the
documented path/family, not every low-level exception or exact prose. RuntimeWarning
for recovered auto CUDA probe failure; no CUDA available -> quiet CPU; explicit
CUDA failure -> error. Normal reorder/missing values/unseen feature levels quiet.
FutureWarning is adopted for future public notices; no feature currently deprecated.
UserWarning/DeprecationWarning are policy categories, no new emitter/API.

## Evidence and remaining limits

The target repair closes the final-audit B1/B2 continuous/non-finite target
defects through shared fit/eval_set/score validation. See
[TARGET_CONTRACT.md](TARGET_CONTRACT.md). Focused/randomized checks, transactional
refit, valid-input frozen equivalence, persistence, schema and sklearn workflow
tests must pass again on the exact candidate source; historical phase counts do not
substitute for that run.

No full sklearn check_estimator compliance is claimed. External CV splitters
and scorers can validate y before the estimator; use ordinary inferred numeric
arrays for object-backed numeric classification labels in those workflows.
Routing, schema format, clocks/probe keys, allocation heuristics and optional
scientific paths remain experimental.

Published RC1 exact-commit CI passed all seven jobs. This stable commit requires
its own remote CI before publication. Fresh local native Windows CPU execution
now succeeds without security workarounds. Independent human pilots remain an
outstanding soft risk; physical CUDA remains optional/PENDING. See
[SUPPORT_SCOPE.md](policies/SUPPORT_SCOPE.md).

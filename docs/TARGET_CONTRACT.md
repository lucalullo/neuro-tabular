# Classification and regression target contract

Version 1.0.0 retains the deliberate invalid-input hardening introduced in RC1. Earlier 0.3.1 code could
accept fractional binary labels in fit/predict even though sklearn scoring
rejected them. That inconsistent acceptance is not preserved.

## Classification

Targets are single-output, one-dimensional arrays with the same number of
samples as X. After normal array conversion, sklearn target typing must infer
binary or multiclass. Integers, strings, booleans and integer-valued floating
labels work when sklearn assigns one of these types. Dtype alone is not an
acceptance rule. Object-backed scalar arrays first use pandas infer_objects
(value-preserving inference, with no string-to-number coercion), then sklearn
target typing. Classes and predicted labels retain the inferred scalar dtype.
This keeps ignored labels from imposing a spurious object target type.

Continuous/fractional, continuous-multioutput, multilabel and other multioutput
targets are rejected with ValueError. Missing numeric labels and positive or
negative infinity are rejected before splitting, encoding or metrics, regardless
of warning filters. The textual labels "inf" or "NaN" are ordinary strings,
not numeric infinity or a missing numeric value.

Training requires at least two comparable classes. Primary rows with zero
sample weight are excluded before target validation, class discovery or
preprocessing. Object scalar inference is shared across fit, eval_set and score, including
manually filtered input; ignored labels cannot change the effective target type.
Explicit eval_set targets use the same target validator and
must contain only training classes; a single known validation class is allowed
unless the chosen metric requires both. Score uses the same validator but
permits a single class and new true labels, as ordinary sklearn accuracy does.
Invalid scoring targets are rejected before prediction and metric evaluation.

## Regression

Targets must be one-dimensional, finite real numeric values. Existing numeric
conversion and scale/constant-target behavior are retained. Fit, eval_set and
score share the existing numeric target validator. Boolean, complex and
non-numeric regression targets are unsupported. R2 keeps sklearn's normal
small-sample behavior; no new statistical guarantee is made.

## Compatibility

Constructor signatures, defaults, valid-model fitted state, preprocessing,
training, checkpoint selection, prediction and probability math are unchanged.
Failed refit remains transactional and preserves the caller's RNG state.
Only invalid/previously inconsistent input acceptance is narrowed, including
score validation before metrics. Valid historical pickle/joblib fixtures remain
subject to the existing trusted-artifact, same-runtime persistence policy.
Loading models from previously inconsistent target domains is not a migration
contract. External sklearn splitters/scorers type targets independently before
delegation; use ordinary numeric arrays for numeric object-backed labels in
those workflows. This does not add general sklearn coercion or routing support.

The target typing behavior depends on the installed supported sklearn version.
This contract does not promise full sklearn check_estimator or metadata-routing
compliance. It adds no task parameter, public validator or new model feature.

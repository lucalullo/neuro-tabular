# Optional internal K-fold cross-validation

NeuroTabular 2.1.0 provides optional internal K-fold diagnostics.
The default remains cv=None. CPU functional qualification covers binary
classification, multiclass classification and regression on artificial
fixtures; it does not establish better predictive accuracy or GPU scope.

Use `NeuroTabularClassifier(cv=3)` or `NeuroTabularRegressor(cv=3)` and ordinary
`fit(X_train, y_train)`. `cv=None` is the unchanged default. Nothing runs K-fold
CV unless it is explicitly enabled. There is no tuning, ensemble or calibrator.

Each outer fold has its own FIT/STOP within its training partition. Checkpoint
selection and early stopping use only that STOP. The outer validation is scored
once after training. Preprocessing, category counts and target statistics are
estimated in that fold's FIT only, then re-estimated within the fold's full
training partition if full_data_refit=True. Outer validation never enters those
statistics. Finally a new estimator is fitted on the supplied TRAIN procedure;
the fold results do not choose its recipe or checkpoint. No external test is
accepted or used by the CV procedure.

For K folds the cost is K+1 training passes, and 2(K+1) if full_data_refit=True.
The final estimator uses the native holdout/early stopping unless full_data_refit
requests training on all supplied positive-weight rows. Thus cv=3 costs four passes,
or eight with refit. Exactly one network is retained for prediction. Inference uses
the unchanged native network. Fit profiling includes all folds, scoring and the
final fit. Planning stores O(K*n) indices, not K networks; plan_index_bytes is
index storage, not a measurement of process RSS.

Pass `fit(X_train, y_train, groups=train_groups)` for grouped internal CV. Groups
are positional, one-dimensional, nonmissing hashable IDs. Zero-weight rows are
removed before group and target validation. Group boundaries are respected in
outer folds and inner FIT/STOP. Classification uses StratifiedKFold or
StratifiedGroupKFold; regression uses KFold or GroupKFold. The grouped inner
classification STOP uses the first fold of a fixed TRAIN-only stratified-group
partition with min(number_of_groups,max(2,round(1/validation_fraction))) folds.
Its actual fraction can differ from the requested one because groups cannot be
split. The actual fraction is reported. No lucky partition search or retry is
performed. Infeasible class/group/metric support raises before neural training.

Without groups the guarantee is row-level only. Related rows and duplicates
must have the same supplied group ID, including transitive relationships. The
estimator does not infer subject IDs, temporal order or causal independence.
Integer cv is not a time-series splitter; use an explicitly designed external
temporal validation protocol for ordered data. groups with cv=None is rejected
rather than silently ignored. cv and eval_set are mutually exclusive. The argument
eval_sample_weight also requires cv=None plus eval_set.

sample_weight is propagated inside each FIT/STOP and outer scoring. Native
class_weight affects training only. cv_results_ contains fold scores, timing,
actual row counts, epochs, plan hash and total passes; it stores no fold model,
raw group IDs, vocabulary copies or OOF predictions. Classification reports NLL,
accuracy or binary AUC according to eval_metric. Regression reports RMSE in
original units for loss/rmse, original MAE or R2; its native inner loss/history
continues to use standardized MSE. Fold mean/std are descriptive, not an
independence-adjusted confidence interval or an independent confirmation.

CV may be composed with external sklearn CV but multiplies its cost. With outer
M and inner K folds, M*(K+1) passes are needed, doubled with full_data_refit.
Group metadata must be routed explicitly according to the sklearn version.
Use cv=None when only the external CV is wanted. Pipelines still must preserve
DataFrame names, and do not automatically transform eval_set.

Successful refit replaces CV metadata; failed refit preserves the old fitted
model and report. clone is unfitted. Same-runtime trusted pickle/joblib retain
the single final model and small report. Old states lacking cv receive None;
this does not establish arbitrary cross-version or CPU/GPU portability.

There is no claim that CV improves accuracy or that any new classification
recipe is qualified. Default regression patience8/max30, robust standardization,
weighted mean, classifier recipes and mandatory dependencies remain unchanged.

# Frozen six-family CPU protocol

Protocol `P6-2026-11-19` is specified by [protocol.json](protocol.json), the
source/split manifest, software lock and runtime module hashes. It replaces
an unexecuted twelve-family proposal with a smaller representative pilot.
Selection preceded model results and excludes known prior predictive tuning
exposure found in the available source history. Legacy coverage is incomplete;
this is not a claim that every source has never been seen anywhere, nor a
formal independent confirmation of a new learned recipe.

## Roles and comparisons

All eight methods share each family's rows, targets, feature schema, split
and seed. IDs and co-targets are excluded. Exact input duplicates are grouped;
fixed temporal roles purge earlier duplicate exposure instead of moving future
records into training. Source-defined study and year roles override random
splits. Official OpenML roles are used only if group-disjoint; otherwise the
predeclared grouped fallback is recorded before training. No redraw follows
a poor score. Missing class support stops that cell.

**Default:** one final fit on the TRAIN pool (FIT plus validation), constructor
defaults except the documented seed/resource/adapter controls. In particular,
NeuroTabular is instantiated with `random_state=seed` only.

**hpo4:** four fixed candidates, each fitted only on FIT. Selection minimizes
validation NLL for classification or RMSE for regression; exact ties choose
the lowest candidate index. One separately counted final fit uses the TRAIN
pool. TEST is absent from every fitting/selection job. An incomplete four-trial
search is labeled partial and excluded from complete-search aggregates, while
its outcome and full cost remain reported. There is no threshold tuning.

Every final model must be closed and hash-sealed before TEST predictions.
The reporting stage reads TEST labels only after predictions are sealed. No
TEST score may change a model, search space, threshold, dataset or budget.

## Technical gates

Before real-source fits, each method/task has one mixed synthetic-data fit and
one fresh-process replay. Predictions must be finite, probability columns must
match classes and sum to one, serialization must work, and the replay must be
bit-identical in this pinned CPU environment. Failure closes only that
method/task. No replacement method or retry is added.

Independent real-source cells continue within their predeclared budgets even
when another method loses or fails. This benchmark has no quality-based gate
that removes unfavorable datasets. It does not promote a new library default.

## Measurements

Windows native, CPU logical core 2 for model work, core 4 for the controller,
one BLAS/PyTorch/estimator thread. No CUDA. Fit workers are fresh, one at a
time. Complete fit CPU/wall includes process startup, imports, conversion,
preprocessing, validation, training and model serialization. API fit time is
recorded separately. Tuning totals include unsuccessful trials.

For each family/seed/track, compare every alternative with 2.1.0 in a dedicated
two-worker pair. A separate pair compares 2.1.0 with itself. Both processes
perform the same cold prediction, probability export where applicable, five
warm-ups and twelve blocks. Blocks alternate A/B then B/A; each workload appears
four times, twice in each order. Workloads are 1 row ×256 calls, 32 rows ×64,
and the full TEST ×8, repeated four times. Workers import serially, acknowledge
READY before prediction, and confirm DONE plus normal process-tree exit.

Warm CPU and wall ratios use an order-stratified paired bootstrap of log
ratios (5,000 resamples, seed 417). All A/A 95% intervals must lie inside
[0.90, 1.10] for timing validity. Zero-resolution CPU blocks or a failed A/A
are inconclusive. A/B ratios and intervals are reported regardless of whether
they favor NeuroTabular; no speed win is claimed if uncertainty includes 1.
This gate concerns warm measurement validity, not a general training-speed
qualification. Cold predictions, complete fit and isolated RSS stay separate.

Peak RSS is per process and includes imports, adapters and model load/fit.
Worker descendants are tracked, including Windows virtualenv launchers. The
monitor uses complete archive censuses, conservative unresolved reservations,
atomic receipts and hash-bound process acknowledgements. A temporary rename
race triggers another complete census; persistent errors stop execution.

## Metrics and uncertainty

Classification: accuracy, balanced accuracy, macro-F1, macro-recall, native
class-aligned NLL, binary AUC or macro OVR AUC; Top-5 only when classes exceed
five. Native `predict` labels are retained, including tie behavior. No score
inversion or probability calibration is introduced by the evaluator.

Regression: RMSE, MAE and R². All tasks: full fit, tuning, inference CPU/wall,
peak RSS, model bytes, replay outcome and error/resource-stop status.

Conditional unit-bootstrap intervals cover balanced accuracy/NLL or RMSE.
The same declared units support paired comparisons; seed repetitions are not
independent source families. Rare-class bootstrap support is reported; an
interval is unavailable rather than fabricated when support is inadequate.
Climate has one held-out simulation study, so no independent-study interval.
AUC and macro-F1 are point estimates in this pilot. Summary ratios use median
seed results and equal family weights; two families per task give descriptive
evidence, not a universal ranking or a population-level confidence interval.

## Finite budget and stops

Maximum 24 synthetic-data backend fits, 144 real default fits, 576 search
trials and 144 selected final fits: **888 fit attempts**, zero internal refits.
Up to 576 metrology workers and 312 cold replays, 288 TEST scorings,
576 validation scorings and 24 synthetic scorings. At most 27,750,000 optimizer
steps and 760,752 boundary prediction API calls. Native validation/forward
work inside a fit is included in its CPU/step limit, not misreported as free
training. Tree boosting rounds are preserved in each estimator configuration.

CPU: 273,600 seconds (76 h); active wall: 547,200 seconds (152 h).
Per process: 4 GiB RSS, one thread. At most two resident model processes.
Archives: 34,359,738,368 bytes (32 GiB). Individual defaults/finals allow
300 s CPU; each trial 225 s; synthetic fit 60 s. Each metrology resident
allows 60 s CPU; pair wall 240 s. Cold replay allows 20 s CPU/40 s wall.
Model file cap 64 MiB; optimizer cap 50,000 steps per neural fit.

Run archives include generated reports and temporary worker files. Frozen
preexisting software and source inputs are separately inventoried, and are not
silently charged as new fits. All worker working directories and caches stay
inside their monitored job. Before dispatch, available host RAM must cover the
RSS caps of all proposed resident workers plus 1 GiB for the controller/OS.
Insufficient headroom pauses before starting a ticket, without consuming a fit
or retrying an experiment.

These are proposed caps until the maintainer grants this exact admission.
Resource stops are failures, never successful truncated defaults. A model
that cannot finish within the common cap gets no fabricated score. All costs
and attempts remain recorded. Failed/incomplete tickets are never relaunched
automatically. Shutdown reserves are included; no automatic extension.

## Executed result and resource scope

The exact frozen executable configuration is `protocol.json`, hash `ef629cfa68cde30ba73f1759a9dc3ad470f335c2a2bdcd265ba6a29e1b242c40`. The admitted recovery controller has CPU/wall maxima of 273,000/547,000 seconds within the original 273,600/547,200 cumulative ceilings; it does not reopen the previous grant. All scientific settings and gates are unchanged. The 21,590-second controller wall term is the additional wall margin in the cumulative budget proof, not an independent cap on overlapping controller lifetime.

The completed run used 600 fits (24 artificial, 576 natural), 384 measurement processes, 211 replay attempts and 181 TEST scores. 288 fit slots remained unused because the original qualification gates closed those backend/task branches. All 288 final planned result cells, including 107 unavailable cells, remain visible. Eight preliminary replay CPU stops, five measurement-pair CPU stops and six natural replay CPU stops are retained without retry. A worker exit code of zero does not override a recorded CPU-limit stop. The failure charge includes the frozen two-second closure reservation.

See [hardware](HARDWARE.json), [measurement definitions](MEASUREMENT_SCOPE.md), [availability in the scorecard](SCORECARD.md) and [source and output hashes](../../SOURCE_MANIFEST.sha256).

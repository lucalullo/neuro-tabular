# Measurement scope

All model selection used TRAIN/validation roles. TEST metrics were calculated
after all admissible final fits and prediction replay attempts were closed.
There are 181 measured final endpoints, 96 excluded by the original backend
qualification gates, five stopped measurement pairs and six CPU-limited cold
replays. Failures are retained and never treated as zero quality or infinite cost.

| Field | Meaning and limitation |
|---|---|
| `fit_compute_cpu_s` | Adapter fit, preprocessing and backend construction, after common worker imports. Preprocessing is not independently timed. |
| `fit_cpu_s`, `fit_wall_s` | Complete observed fit-worker cost, including shared startup and persistence. |
| `training_total_*` | Final fit plus all four candidate trials for `hpo4`; one fit for `default`. |
| `cold_predict_wall_ns` | First `predict` on an already loaded resident model, excluding process startup and model loading. |
| `warm_full_predict_*` | Repeated, instrumented native `predict` on the full frozen TEST workload. Probability-quality evaluation is separate; these are not `predict_proba` latency measurements. |
| `fit_peak_rss_bytes`, `inference_peak_rss_bytes` | Isolated process peaks in a shared PyTorch-capable worker. Torch is imported for tree workers too. Not standalone dependency footprint or minimum deployment RAM. |
| `model_bytes` | Actual serialized final model artifact; not installation size or RAM. |
| `warm_full_rows_per_s_point_only` | Frozen TEST rows divided by warm wall latency, only when workload hashes agree and the dedicated A/A is VALID. |
| Forward counters | Selected learned-entry calls, not FLOPs or equal compute. Trees report not applicable. |

Fit cost and RSS are descriptive observations. Warm paired intervals use four
balanced blocks and are conditional on that CPU/process pair; only 2/36 A/A
controls pass all preregistered requirements. Invalid timing is not retried.
All raw timings remain available, including the inconclusive ones.

327 fits preceded a maintainer-initiated reboot; 273 followed it. No fit was
repeated to standardize resource epochs. All paired inference measurements
occurred after the reboot. Training startup, cache, available memory and system
load were not identical across every fit. Do not claim a paired training-time
confidence interval or a general speed/RAM improvement from these point costs.

Quality intervals resample declared TEST units conditional on frozen models.
Climate has one held-out study and therefore no independent-study interval.
AUC, macro-F1, average precision, per-class metrics and local Pareto membership
are descriptive where no interval is provided. A missing or nonsignificant
interval does not establish equivalence. NLL difference intervals are never
silently converted to relative-equivalence intervals.

Each star in the original loss/cost scatter plots marks a local observed-method
point frontier for NLL or RMSE. It is not confidence-qualified dominance or a
global ranking. The paired interval figures reuse existing results without
new scoring, estimation or selection. See the root `SOURCE_MANIFEST.sha256` for hashes.

## Inference timing control audit

A documentation-stage audit of the 36 A/A records in
[results.json](results/results.json) reproduces the original timing gate:
all six workload/clock confidence intervals in a control must be measured
and wholly inside [0.90, 1.10]. No new inference, scoring or statistical
resampling was performed. The 34 inconclusive controls split into 12 with
CPU-only issues, one with wall-only issues and 21 with both. Two pass all six
requirements (`default` Bank Marketing seed 2026 and `hpo4` Bank Marketing
seed 42).

| Workload | CPU intervals outside band | CPU intervals unavailable | Wall intervals outside band |
|---|---:|---:|---:|
| 1 row | 13 / 36 | 0 / 36 | 9 / 36 |
| 32 rows | 28 / 36 | 0 / 36 | 14 / 36 |
| Full TEST | 8 / 36 | 16 / 36 | 11 / 36 |

Rows and clocks overlap; they must not be summed as distinct failed controls.
All 16 unavailable full-TEST CPU intervals have the recorded reason
"Insufficient blocks or clock quantization". The report cannot separate these
two possibilities from the retained interval summaries alone. The other 83
intervals outside the band are measured, not absent. The high rate of
inconclusive controls is a measurement qualification problem. It does not
establish prediction nondeterminism, model failure or a general latency
regression; prediction/replay checks are separate.

### Prospective measurement repair

Before a separately approved timing campaign:

1. Characterize the effective resolution of CPU and wall clocks on the actual
   Windows worker, including zero and quantized elapsed values. Preserve the
   raw block values and both process identities.
2. Use a fit-free preflight to select repeated-call block durations long enough
   relative to the observed clock granularity; freeze the schedule before
   comparative results. Keep identical workload counts for each paired side.
3. Verify the full controller-to-worker affinity and ownership path, thread
   limits, memory availability, process isolation, warm-up, background load and
   balanced order. Separate clock granularity from process-to-process variability.
4. Qualify A/A across all three workload sizes with the prospective protocol
   and sufficient balanced blocks, without relaxing a gate to obtain a PASS.
   Run A/B comparisons only when their required A/A control is valid.
5. Preserve every failed or inconclusive control. Publish new receipts and
   scope separately; never overwrite the original 2/36 result.

The existing frozen measurement code and data remain unchanged. The cause of
the measured interval deviations and the effectiveness of a new timing design
require new controlled execution on the relevant environment. This audit does
not qualify latency, repair the historical measurement run or prove a general
speed advantage. See the [qualification roadmap](../README.md#qualification-roadmap).

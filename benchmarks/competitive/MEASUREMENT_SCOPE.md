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

# NeuroTabular 0.1.0 benchmark report

## Scope and claims

This report is an engineering benchmark for the first release candidate. It is
not a state-of-the-art claim and does not establish universal superiority over
tree models or other neural methods. All datasets are synthetic and all results
are specific to the software and CPU environment below.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-08-27 |
| OS | Windows 11, build 26100 |
| CPU visibility | 2 physical cores / 4 logical CPUs |
| RAM | 11.71 GiB |
| Python | 3.12.13 |
| NumPy | 2.2.6 |
| pandas | 3.0.5 |
| PyTorch | 2.13.0+cpu |
| scikit-learn | 1.6.1 |
| CatBoost | 1.2.10, benchmark-only |
| CUDA / VRAM | Unavailable / not measured |

The environment did not expose a reliable CPU marketing name through the
restricted system interface. LightGBM 4.7.0 installed, but Windows Application
Control blocked its native DLL with `WinError 4551`; no LightGBM score is
reported. CatBoost imported and ran successfully.

## Method

`benchmarks/run_benchmarks.py` generates seven binary tasks:

1. 600-row numerical classification;
2. 800-row mixed numerical/categorical classification;
3. 1,200-row mixed data with numerical and categorical missing values;
4. 1,200-row categorical-heavy data;
5. 1,200-row data with a 300-value identifier and 20-value segment;
6. 1,200-row 90/10 imbalanced classification; and
7. 5,000-row, 24-feature numerical classification.

Every task uses a stratified 75/25 external train/test split. NeuroTabular uses
its default internal validation behavior and no benchmark-specific tuning.
HistGradientBoosting and CatBoost receive a leakage-safe ordinal-encoding
pipeline for heterogeneous frames. External libraries are benchmark-only and
are not NeuroTabular runtime dependencies.

NeuroTabular, HistGradientBoosting, and CatBoost ran with seeds 17 and 23. The
legacy prototype ran seed 17 only with a matched 30-epoch ceiling, patience 4,
64 hidden units, two hidden layers, batch size 256, and CPU execution. The
legacy comparison is a development baseline, not release history.

Fit time includes preprocessing, model construction, optimizer setup, training,
validation, checkpointing, and best-weight restoration. The first neural fit in
the process includes a roughly three-second PyTorch optimizer cold start; later
fits reuse loaded runtime modules.

## Two-seed results

Values are arithmetic means across seeds 17 and 23.

| Dataset | NeuroTabular ROC-AUC | NeuroTabular fit s | HistGradientBoosting ROC-AUC | HistGradientBoosting fit s | CatBoost ROC-AUC | CatBoost fit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small numeric | 0.8978 | 2.1749 | 0.9188 | 1.5426 | 0.9326 | 1.7706 |
| Small mixed | 0.8509 | 0.6281 | 0.8232 | 0.9197 | 0.8558 | 0.7368 |
| Mixed with NaNs | 0.8112 | 0.4696 | 0.8156 | 1.9657 | 0.8478 | 1.1601 |
| Categorical-heavy | 0.7923 | 1.1362 | 0.8122 | 1.3657 | 0.8340 | 0.8487 |
| Moderate high cardinality | 0.8386 | 0.8830 | 0.7968 | 1.5757 | 0.8387 | 0.9682 |
| Imbalanced binary | 0.9435 | 1.2318 | 0.9617 | 1.4466 | 0.9664 | 1.4544 |
| Medium synthetic | 0.9799 | 6.1620 | 0.9871 | 2.3911 | 0.9816 | 3.3880 |
| **All runs** | **0.8735** | **1.8122** | **0.8736** | **1.6010** | **0.8938** | **1.4753** |

Across all two-seed runs, mean log loss was 0.4104 for NeuroTabular, 0.4898
for HistGradientBoosting, and 0.3864 for CatBoost. Mean prediction times were
0.0201 s, 0.0284 s, and 0.0109 s respectively.

These aggregate means combine tasks with different difficulty and should not be
read as a ranking statistic. CatBoost was strongest on average. NeuroTabular
was particularly competitive on the mixed and moderate-cardinality tasks, while
categorical-heavy data remained a clear weakness.

## Paired legacy comparison

The fair paired comparison uses the seven seed-17 rows available for both
neural implementations.

| Metric | Legacy prototype | NeuroTabular | Difference |
| --- | ---: | ---: | ---: |
| Mean ROC-AUC | 0.8717 | 0.8781 | +0.0064 |
| Mean log loss | 0.3977 | 0.4042 | +0.0066 (lower is better; regression) |
| Mean fit time | 3.4732 s | 2.3236 s | 1.49× aggregate speedup |
| Mean predict time | 0.0348 s | 0.0223 s | 1.56× aggregate speedup |

The mean of per-dataset fit-time ratios was 2.24×. NeuroTabular improved
ROC-AUC strongly on the imbalanced task and modestly on small numeric and
moderate-cardinality tasks. It regressed on mixed-with-NaNs and
categorical-heavy tasks. The average ROC-AUC/fit-time tradeoff improved, but the
regressions are important targets for later releases.

The first NeuroTabular run paid the cold optimizer setup cost while the legacy
model ran later in the already-warm process, which biases the time comparison
against NeuroTabular. No adjustment was applied.

## Parameters, RAM, and VRAM

NeuroTabular models ranged from 34,796 to 36,972 trainable parameters across the
benchmark schemas. Mean measured preprocessing time was 0.0640 s.

The process RSS sampler observed a mean 12.10 MiB incremental peak for
NeuroTabular across all runs, including 76.63 MiB on the first cold fit and
roughly 2.5–22.7 MiB on later runs. Incremental RSS is allocator- and run-order
dependent: the legacy models ran after PyTorch was warm and reused allocated
memory, so their small RSS deltas are not a valid cross-model memory comparison.
No absolute peak-RAM superiority claim is made.

CUDA was unavailable, so VRAM is reported as not measured. The harness records
`torch.cuda.max_memory_allocated()` when CUDA exists.

## Reproduction

```bash
python benchmarks/run_benchmarks.py --seeds 17 23
```

An optional legacy implementation can be loaded from an external source path
through the command-line arguments documented by `--help`; no legacy source is
distributed with NeuroTabular.

Results can vary with CPU, BLAS/OpenMP runtime, thread scheduling, PyTorch,
dependency versions, and synthetic split. A broader public-dataset benchmark is
required before making stronger comparative claims.

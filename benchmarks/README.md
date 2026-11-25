# NeuroTabular benchmarks

**Competitive evidence: NeuroTabular 2.1.0, six-family CPU pilot.**

NeuroTabular 2.0.0 is included only as a version-to-version reference; it is excluded from the external-method descriptive overview and retained in the full scorecard and upgrade comparisons.

## Scope and coverage

| Item | Recorded scope |
|---|---|
| Evaluated release | 2.1.0, with 2.0.0 as a version reference |
| External methods attempted | RealMLP, XGBoost, LightGBM, CatBoost, TabM, FT-Transformer |
| RealMLP status | No admitted real-data fit or TEST endpoint (0/36); preliminary replay CPU limits closed its branches. Follow-up priority; see [roadmap](#qualification-roadmap). |
| TabM / FT-Transformer status | 0/36 and 1/36 final endpoints measured, respectively; no complete comparative ranking. |
| Dataset families | 6 (two per task) |
| Regimes | Default and four-candidate tuning, kept separate; three seeds |
| Final endpoints | 181 measured / 288 planned; 107 unavailable |
| Reproducibility | Executable frozen protocol; exact environment, data and frozen wheels required |

### Why 107 endpoints are unavailable

Of the 288 planned final TEST endpoints, 181 (62.8%) were measured and 107
(37.2%) are unavailable. All 107 concern RealMLP, TabM and FT-Transformer;
the other five methods have complete coverage. These counts describe final
endpoints, not the number of failed training attempts.

| Method | Valid / planned | Qualification excluded | Cold replay stopped | Measurement stopped |
|---|---:|---:|---:|---:|
| NeuroTabular 2.1.0 | 36 / 36 | 0 | 0 | 0 |
| NeuroTabular 2.0.0 (version reference) | 36 / 36 | 0 | 0 | 0 |
| XGBoost | 36 / 36 | 0 | 0 | 0 |
| LightGBM | 36 / 36 | 0 | 0 | 0 |
| CatBoost | 36 / 36 | 0 | 0 | 0 |
| RealMLP | 0 / 36 | 36 | 0 | 0 |
| TabM | 0 / 36 | 36 | 0 | 0 |
| FT-Transformer | 1 / 36 | 24 | 6 | 5 |
| **Total** | **181 / 288** | **96** | **6** | **5** |

- **96 qualification exclusions:** the original synthetic-data backend gate closed those method/task branches before real-data fits. RealMLP and TabM account for 36 each; FT-Transformer accounts for 24.
- **6 cold replay stops:** FT-Transformer on Glass reached the replay CPU limit.
- **5 measurement stops:** FT-Transformer on Anneal reached the measurement-pair CPU limit.

RealMLP and TabM therefore have no admitted real-data results; FT-Transformer
has one of 36 final endpoints. None of these exclusions or stops is a predictive
loss. Incomplete methods are not ranked using their surviving results. See the
[per-seed roster](competitive/results/tables/PER_SEED.csv),
[measurement scope](competitive/MEASUREMENT_SCOPE.md) and
[frozen protocol](competitive/PROTOCOL.md).

## Predictive performance

The common overview uses the default regime and medians of all three measured seeds, against the best fully covered external baseline for the primary metric (NLL for classification, RMSE for regression). The historical 2.0.0 version reference is excluded from this descriptive comparison; partial methods are not ranked. The full scorecard retains all fully covered methods and both regimes.

![Primary metric comparison](overview/quality.svg)

[Summary table](overview/SUMMARY.md) · [Complete scorecard](competitive/SCORECARD.md) · [Per-seed results](competitive/results/tables/PER_SEED.csv)

## Computational efficiency

![Serialized size comparison](overview/model_size.svg)

Size is the serialized model, not process RAM or installation size. The summary table compares training CPU, process RSS and stored bytes with the same quality-selected comparator. Only 2 of 36 dedicated inference A/A controls pass. The other 34 are an unresolved measurement qualification issue, not evidence of slow or incorrect predictions. The recorded clock/interval breakdown and a prospective repair plan are in the [timing audit](competitive/MEASUREMENT_SCOPE.md#inference-timing-control-audit). No historical gate or measurement has been changed.

[Measurement definitions](competitive/MEASUREMENT_SCOPE.md) · [Timing controls](competitive/results/tables/TIMING_CONTROLS.csv) · [Detailed figures](FIGURES.md)

## Quality-resource trade-offs

Bank Marketing favors NeuroTabular on default median NLL (0.666704 vs 0.717406 for the best fully covered external baseline, CatBoost; [paired NLL figure](competitive/results/intervals/default-binary-primary-paired-intervals.png)). The held-out Climate study favors it on default median balanced accuracy (77.67% vs 70.53% for LightGBM, the best measured tree baseline; [balanced accuracy figure](competitive/results/default-binary-balanced_accuracy-training_total_cpu_s.png)), but CatBoost has lower NLL. The Climate observation is descriptive for one held-out study. Trees win substantially on Anneal and Glass. Local point frontiers are descriptive, not confidence-qualified dominance. [When to choose NeuroTabular](competitive/WHEN_TO_CHOOSE.md).

## Methodology and reproducibility

The fixed TRAIN/validation/TEST roles, dataset group rules, models, seeds, qualification and measurement gates are documented in the [protocol](competitive/PROTOCOL.md), [datasets](competitive/DATASETS.md), [methods](competitive/METHODS.md) and [reproduction guide](competitive/README.md). The guide explains the frozen Windows CPU environment, wheel identities, permitted data preparation and explicit execution. No new model fitting is needed to regenerate this overview: `python benchmarks/plot_overview.py` reads the recorded CSV only. The original detailed tables and paired figures have their own reproduction commands and custody records.

## Complete results and figures

| Report | Contents |
|---|---|
| [Complete scorecard](competitive/SCORECARD.md) | All complete methods, both regimes, failures and comparisons |
| [Original report](competitive/results/REPORT.md) | Frozen result interpretation |
| [Detailed figures](FIGURES.md) | Original PNG plots, paired intervals and common SVG overview |
| [Numerical data](competitive/results/README.md) | Full precision JSON, CSV and status |
| [Historical evidence](../docs/BENCHMARK_0_3.md) | Earlier protocols and versions, kept separate |
| [Performance evidence](../docs/PERFORMANCE_0_3_1.md) | Historical measurements |

## Qualification roadmap

| Open item | Next verifiable milestone |
|---|---|
| RealMLP competitive coverage | Priority follow-up: qualify its fit and replay under a prospectively declared resource contract, then run the complete same-family, same-split, same-seed default and tuning comparison. No natural-data outcome is currently available. |
| TabM and FT-Transformer coverage | Resolve backend replay and measurement CPU stops in a separately scoped comparison; keep partial results out of aggregate rankings until coverage is complete. |
| Inference timing | Qualify the measurement harness and the effective CPU clock resolution before new timing runs. Require valid A/A controls before interpreting A/B speed ratios. See the [timing audit](competitive/MEASUREMENT_SCOPE.md#inference-timing-control-audit). |
| Extended categorical support | Complete performance qualification of the extended embedding allocator, preserving ordinary predictions and testing resource use at wider categorical widths. The extension is not included in 2.1.0; the current limit is 128 categorical columns. |

These are future qualification milestones, not scheduled releases or promised
results. A complete RealMLP comparison is a priority after the runner and its
resource limits are qualified and a new execution is approved; no date is
committed. Update this page only when verified evidence becomes available.
The delivered 2.1.0 model and the original measured results remain frozen.

## Limitations

Six families and three seeds do not establish universal superiority or independent confirmation. Unavailable comparisons are not losses or wins; measured incomplete methods are not ranked on surviving cells. Raw data, model weights and private laboratory records are excluded. No new fits, scoring or timing were performed to produce this overview.

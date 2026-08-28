# NeuroTabular 0.2.0 benchmark report

## Scope

This report evaluates the 0.2.0 release defaults against the final immutable
0.1.1 baseline. It separates quality, paired timing, public-data regression,
and contextual tree baselines. Results are evidence for this candidate on this
host, not universal performance guarantees.

No private data was used. Synthetic datasets are generated deterministically;
the two public datasets are bundled with scikit-learn and are not committed to
the repository.

## Environment

- Windows 11, Europe/Rome release run;
- Python 3.12.13;
- PyTorch 2.13.0+cpu;
- NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0;
- 2 physical / 4 logical CPU cores;
- 11.71 GiB system RAM;
- CUDA unavailable; VRAM and AMP throughput not measured.

The project declares Python 3.10-3.12 and PyTorch 2.0+ support. CI separately
pins PyTorch 2.0.1 on Python 3.10. The local environment is not a substitute for
the complete CI matrix.

## Paired release comparison

The principal comparison used seven deterministic binary datasets, seeds 17
and 23, three timing repeats, and a stratified 75/25 train/test split. Each
repeat alternated version order in one warm process. Prediction timing is the
median of five calls. Both versions used the same training budget and random
state. The test partition was not used for preprocessing, validation, early
stopping, or model selection.

Datasets cover small numeric, small mixed, missing values, categorical-heavy,
moderately high-cardinality, imbalanced, and 5,000-row medium synthetic cases.

| Aggregate | 0.1.1 | 0.2.0 | Delta or ratio |
|---|---:|---:|---:|
| Mean ROC-AUC | 0.874559 | 0.878408 | +0.003849 |
| Mean selected/run epochs | 14.143 | 13.714 | -0.429 |
| Median paired fit-time ratio | 1.000 | 0.950 | -4.96% |
| Mean paired fit-time ratio | 1.000 | 0.942 | -5.77% |
| Median paired prediction ratio | 1.000 | 0.986 | -1.41% |
| Mean paired prediction ratio | 1.000 | 0.999 | -0.08% |

The quality change comes from categorical workloads. Purely numerical
workloads retain the scalar 0.1.1 path and produced identical probabilities in
this matrix.

| Dataset | ROC-AUC delta | Paired fit ratio | Paired prediction ratio |
|---|---:|---:|---:|
| categorical-heavy | +0.010378 | 0.956 | 1.185 |
| imbalanced binary | 0.000000 | 0.863 | 0.918 |
| medium synthetic | 0.000000 | 0.967 | 1.023 |
| mixed with NaNs | +0.005067 | 1.025 | 0.867 |
| moderate high-cardinality | +0.005200 | 0.878 | 0.919 |
| small mixed | +0.006300 | 0.514 | 0.775 |
| small numeric | 0.000000 | 1.394 | 1.308 |

The small-numeric timing ratios illustrate noise: model outputs were identical,
yet timings moved in both directions. Aggregate paired ratios are therefore
more informative than any single small workload.

## Quality and contextual tree baseline

The separately executed quality matrix produced:

| Model | Mean ROC-AUC | Mean log loss |
|---|---:|---:|
| NeuroTabular 0.1.1 | 0.874559 | 0.410648 |
| NeuroTabular 0.2.0 | 0.878408 | 0.408868 |
| HistGradientBoosting | 0.873459 | 0.487927 |

HistGradientBoosting is a useful installed, reproducible context rather than a
claim that one method dominates. Optional LightGBM and CatBoost runners are
implemented and execute only when those packages are installed; they are not
runtime dependencies and were not installed for this release run.

Per-dataset 0.1.1 to 0.2.0 quality changes were:

| Dataset | 0.1.1 AUC | 0.2.0 AUC | AUC delta | Log-loss delta |
|---|---:|---:|---:|---:|
| categorical-heavy | 0.793289 | 0.803667 | +0.010378 | -0.008842 |
| imbalanced binary | 0.943398 | 0.943398 | 0.000000 | 0.000000 |
| medium synthetic | 0.979658 | 0.979658 | 0.000000 | 0.000000 |
| mixed with NaNs | 0.805356 | 0.810422 | +0.005067 | -0.004960 |
| moderate high-cardinality | 0.838244 | 0.843444 | +0.005200 | +0.004913 |
| small mixed | 0.849800 | 0.856100 | +0.006300 | -0.003570 |
| small numeric | 0.912168 | 0.912168 | 0.000000 | 0.000000 |

The high-cardinality case improved AUC while slightly worsening log loss. This
is why both metrics are reported.

## Public-data no-regression check

Breast Cancer Wisconsin and a binary projection of Wine were loaded from
scikit-learn. Two seeds and three paired timing repeats used the same protocol.
Both are numerical, so the 0.2 categorical feature path is inactive.

| Aggregate | 0.1.1 | 0.2.0 |
|---|---:|---:|
| Mean ROC-AUC | 0.998113 | 0.998113 |
| Mean fit seconds | 0.4377 | 0.4306 |
| Mean prediction seconds | 0.0157 | 0.0156 |

Predictions and selected epochs matched. The small timing difference is treated
as noise-level no-regression evidence, not a speed claim.

## Kaggle reference supplied with the project brief

The original brief reported an external Kaggle-style 0.1.x reference of about
0.940020 ROC-AUC for NeuroTabular and 0.954774 for LightGBM, with roughly 537 s
versus 30 s over six folds. That dataset and execution environment were not
available locally, so those values were not reproduced and are not attributed
to 0.2.0. This release does not claim to reach 0.945 or to close the tree-model
latency gap on that workload.

## Parameters and memory

The 5,000-row detailed profile used 9,401 trainable parameters in 0.1.1 and
9,063 in the 0.2.0 default workload. This difference is schema- and
architecture-dependent; the adaptive embedding rule can increase or decrease a
particular model.

Peak process RSS deltas in isolated quality runs were effectively unchanged:
the maximum was 75.77 MiB for 0.1.1 and 75.34 MiB for 0.2.0. The cold detailed
profiles recorded about 86.7-86.8 MiB. RSS sampling includes Python, native
libraries, allocator behavior, and first-use effects, so sub-MiB differences
are not meaningful. VRAM is `N/A` because CUDA was unavailable.

## Reproduction

From an installed development checkout:

```bash
python benchmarks/run_benchmarks.py --output benchmark.json
python benchmarks/compare_0_2_0.py \
  --baseline-source ../NeuroTabular-0.1.1/src \
  --candidate-source src \
  --output paired.json
python benchmarks/compare_0_2_0.py \
  --baseline-source ../NeuroTabular-0.1.1/src \
  --candidate-source src \
  --datasets breast_cancer wine_binary \
  --output paired-real.json
```

Use an idle host, record package versions, and do not compare absolute timings
from independent busy processes. Raw JSON files are generated artifacts and are
intentionally excluded from the release package; the scripts and complete
aggregate results remain in the repository.

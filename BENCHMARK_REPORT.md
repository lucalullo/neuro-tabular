# NeuroTabular 0.2.0 benchmark report

## Scope

This report records a reproducible engineering benchmark for the release
defaults. It is intended as development evidence, not a universal performance
claim. Synthetic datasets are generated deterministically; the two public
checks are bundled with scikit-learn and are not committed to the repository.

## Recorded environment

- Windows 11;
- Python 3.12.13;
- PyTorch 2.13.0+cpu;
- NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0;
- 2 physical / 4 logical CPU cores;
- 11.71 GiB system RAM;
- CUDA unavailable; VRAM and AMP throughput not measured.

The package declares Python 3.10-3.12 and PyTorch 2.0+ support. CI provides the
compatibility matrix; this host is only the recorded benchmark machine.

## Synthetic development matrix

Seven deterministic binary datasets, seeds 17 and 23, and a stratified 75/25
train/test split were used. The test partition was not used for preprocessing,
validation, early stopping, or model selection.

The families cover small numeric, small mixed, missing values,
categorical-heavy, moderately high-cardinality, imbalanced, and 5,000-row
medium synthetic cases.

| Metric | NeuroTabular 0.2.0 |
|---|---:|
| Mean ROC-AUC | 0.878408 |
| Mean log loss | 0.408868 |
| Mean selected/run epochs | 13.714 |

Per-dataset release results:

| Dataset | ROC-AUC |
|---|---:|
| categorical-heavy | 0.803667 |
| imbalanced binary | 0.943398 |
| medium synthetic | 0.979658 |
| mixed with NaNs | 0.810422 |
| moderate high-cardinality | 0.843444 |
| small mixed | 0.856100 |
| small numeric | 0.912168 |

## Contextual tree baseline

The same quality matrix recorded:

| Model | Mean ROC-AUC | Mean log loss |
|---|---:|---:|
| NeuroTabular 0.2.0 | 0.878408 | 0.408868 |
| HistGradientBoosting | 0.873459 | 0.487927 |

`HistGradientBoostingClassifier` is included only as a reproducible contextual
baseline. Optional LightGBM and CatBoost runners execute only when those
packages are installed; they are not runtime dependencies. This table must not
be read as evidence that NeuroTabular generally dominates tree ensembles.

## Public-data engineering check

Breast Cancer Wisconsin and a binary projection of Wine were loaded from
scikit-learn. Two seeds were used with the same train/test discipline.

| Metric | NeuroTabular 0.2.0 |
|---|---:|
| Mean ROC-AUC | 0.998113 |
| Mean fit seconds | 0.4306 |
| Mean prediction seconds | 0.0156 |

These datasets are small and largely numerical, so the result is a smoke/no-
regression style engineering check rather than a broad benchmark.

## Parameters and memory

A documented 5,000-row mixed profile used 9,063 trainable parameters for that
specific schema. Peak process RSS in the recorded cold profile was about 86.8
MiB. Both values are schema- and environment-dependent; the embedding-width
rule can produce different model sizes for different cardinalities.

VRAM is `N/A` because CUDA was unavailable on the recorded host.

## Reproduction

From an installed development checkout:

```bash
python benchmarks/run_benchmarks.py --output benchmark.json
python benchmarks/profile_0_2.py --output profile.json
python benchmarks/run_0_2_ablations.py --output ablations.json
```

Use an idle host, record package versions, and treat absolute timings as
host-specific. Raw JSON outputs are generated artifacts and intentionally
excluded from the source release.

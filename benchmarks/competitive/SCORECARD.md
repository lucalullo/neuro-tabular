# Competitive scorecard

Medians over three model seeds, within each named family and regime. These are descriptive summaries, not confidence intervals.
Incomplete methods are shown in the full failure roster rather than ranked on their surviving subset.
BA is balanced accuracy. Lower NLL and RMSE are better. CPU and RAM refer to the recorded common worker; see [measurement definitions](MEASUREMENT_SCOPE.md).

## default

| Family | Method | BA | NLL | AUC | RMSE | Fit compute CPU s | Total training CPU s | Fit RSS MiB | Model KiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anneal | NeuroTabular 2.0.0 | 0.628742 | 0.286735 | 0.938832 | — | 4.76562 | 10.407 | 356.934 | 211.745 |
| anneal | NeuroTabular 2.1.0 | 0.628742 | 0.286735 | 0.938832 | — | 4.6875 | 10.204 | 358.723 | 211.776 |
| anneal | XGBoost | 0.956947 | 0.033408 | 0.999934 | — | 2.14062 | 5.625 | 281.922 | 425.146 |
| anneal | LightGBM | 0.958473 | 0.099362 | 0.998145 | — | 2.25 | 5.782 | 285.961 | 1267.65 |
| anneal | CatBoost | 0.958473 | 0.0401875 | 0.999691 | — | 165.484 | 168.922 | 351.492 | 3037.9 |
| bank_marketing | NeuroTabular 2.0.0 | 0.544196 | 0.666704 | 0.719271 | — | 13.6875 | 22.125 | 482.816 | 167.731 |
| bank_marketing | NeuroTabular 2.1.0 | 0.544196 | 0.666704 | 0.719271 | — | 14.75 | 23.938 | 481.336 | 167.763 |
| bank_marketing | XGBoost | 0.545285 | 0.784755 | 0.648531 | — | 3.79688 | 8.875 | 305.582 | 358.062 |
| bank_marketing | LightGBM | 0.528897 | 0.719467 | 0.672174 | — | 3.28125 | 8.688 | 311.133 | 331.75 |
| bank_marketing | CatBoost | 0.535644 | 0.717406 | 0.679673 | — | 72.625 | 78.266 | 337.555 | 1328.83 |
| climate_crashes | NeuroTabular 2.0.0 | 0.776678 | 0.157654 | 0.948365 | — | 4.39062 | 10 | 352.898 | 156.605 |
| climate_crashes | NeuroTabular 2.1.0 | 0.776678 | 0.157654 | 0.948365 | — | 4.35938 | 9.969 | 352.602 | 156.64 |
| climate_crashes | XGBoost | 0.666523 | 0.160275 | 0.939759 | — | 2.10938 | 5.547 | 283.398 | 96.7158 |
| climate_crashes | LightGBM | 0.70525 | 0.145389 | 0.960413 | — | 2.09375 | 5.532 | 282.625 | 206.433 |
| climate_crashes | CatBoost | 0.607143 | 0.137218 | 0.964716 | — | 10.0625 | 13.469 | 253.551 | 1103.51 |
| fiat500 | NeuroTabular 2.0.0 | — | — | — | 800.495 | 5.03125 | 11.907 | 358.371 | 151.308 |
| fiat500 | NeuroTabular 2.1.0 | — | — | — | 786.206 | 4.96875 | 10.61 | 358.137 | 152.319 |
| fiat500 | XGBoost | — | — | — | 788.259 | 2.29688 | 6.282 | 281.02 | 354.361 |
| fiat500 | LightGBM | — | — | — | 785.189 | 2.39062 | 6.344 | 281.875 | 264.047 |
| fiat500 | CatBoost | — | — | — | 757.44 | 20.9531 | 24.297 | 251.738 | 1076.59 |
| gas_nox | NeuroTabular 2.0.0 | — | — | — | 13.2047 | 14.1562 | 21.86 | 434.961 | 151.888 |
| gas_nox | NeuroTabular 2.1.0 | — | — | — | 13.2047 | 13.6719 | 20.36 | 436.371 | 151.919 |
| gas_nox | XGBoost | — | — | — | 13.2022 | 2.26562 | 5.829 | 288.102 | 434.089 |
| gas_nox | LightGBM | — | — | — | 12.5541 | 3.07812 | 6.891 | 287.754 | 276.518 |
| gas_nox | CatBoost | — | — | — | 12.7164 | 11.875 | 15.719 | 257.902 | 1094.91 |
| glass | NeuroTabular 2.0.0 | 0.711538 | 0.779805 | 0.915502 | — | 3.25 | 8.954 | 352.676 | 152.315 |
| glass | NeuroTabular 2.1.0 | 0.711538 | 0.779805 | 0.915502 | — | 3.5 | 9.25 | 351.625 | 152.362 |
| glass | XGBoost | 0.838889 | 0.682309 | 0.955098 | — | 3.125 | 7.329 | 281.98 | 524.997 |
| glass | LightGBM | 0.838889 | 0.683874 | 0.957509 | — | 3.125 | 7.172 | 282.641 | 603.378 |
| glass | CatBoost | 0.744444 | 0.542862 | 0.957489 | — | 11.3906 | 15.735 | 261.5 | 3569.04 |

## hpo4

| Family | Method | BA | NLL | AUC | RMSE | Fit compute CPU s | Total training CPU s | Fit RSS MiB | Model KiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anneal | NeuroTabular 2.0.0 | 0.788055 | 0.145528 | 0.981652 | — | 4.8125 | 51.97 | 355.77 | 211.243 |
| anneal | NeuroTabular 2.1.0 | 0.788055 | 0.145528 | 0.981652 | — | 4.75 | 51.534 | 356.957 | 211.29 |
| anneal | XGBoost | 0.956947 | 0.0322686 | 0.999934 | — | 2.67188 | 29.612 | 294.816 | 2261.15 |
| anneal | LightGBM | 0.958473 | 0.117087 | 0.996711 | — | 2.65625 | 30.065 | 306.414 | 4854.77 |
| anneal | CatBoost | 0.956947 | 0.0581421 | 0.999513 | — | 153.188 | 638.878 | 342.195 | 3037.56 |
| bank_marketing | NeuroTabular 2.0.0 | 0.53827 | 0.683381 | 0.712136 | — | 15.4844 | 107.768 | 480.543 | 167.649 |
| bank_marketing | NeuroTabular 2.1.0 | 0.53827 | 0.683381 | 0.712136 | — | 16.4844 | 106.752 | 480.953 | 167.681 |
| bank_marketing | XGBoost | 0.540707 | 0.719547 | 0.669807 | — | 5.26562 | 54.158 | 314.09 | 1590.77 |
| bank_marketing | LightGBM | 0.528897 | 0.719467 | 0.672174 | — | 3.32812 | 46.189 | 310.719 | 331.75 |
| bank_marketing | CatBoost | 0.524413 | 0.677716 | 0.707314 | — | 46.4688 | 243.235 | 349.965 | 1696.54 |
| climate_crashes | NeuroTabular 2.0.0 | 0.767642 | 0.145814 | 0.959122 | — | 4.46875 | 49.971 | 356.074 | 157.813 |
| climate_crashes | NeuroTabular 2.1.0 | 0.767642 | 0.145814 | 0.959122 | — | 4.40625 | 49.486 | 354.004 | 157.909 |
| climate_crashes | XGBoost | 0.711274 | 0.152959 | 0.952238 | — | 2.17188 | 28.377 | 279.27 | 245.678 |
| climate_crashes | LightGBM | 0.70525 | 0.145389 | 0.960413 | — | 2.09375 | 27.831 | 281.918 | 206.433 |
| climate_crashes | CatBoost | 0.639845 | 0.174614 | 0.947935 | — | 10 | 61.472 | 252.723 | 1099.18 |
| fiat500 | NeuroTabular 2.0.0 | — | — | — | 795.577 | 4.54688 | 54.049 | 354.16 | 151.133 |
| fiat500 | NeuroTabular 2.1.0 | — | — | — | 785.467 | 5.07812 | 57.611 | 354.23 | 151.834 |
| fiat500 | XGBoost | — | — | — | 765.073 | 2.8125 | 33.33 | 289.934 | 1725.21 |
| fiat500 | LightGBM | — | — | — | 785.189 | 2.48438 | 29.096 | 281.84 | 264.047 |
| fiat500 | CatBoost | — | — | — | 746.593 | 24.6562 | 113.908 | 254.043 | 1073.68 |
| gas_nox | NeuroTabular 2.0.0 | — | — | — | 14.3009 | 31.9375 | 126.189 | 499.617 | 804.702 |
| gas_nox | NeuroTabular 2.1.0 | — | — | — | 14.3009 | 17.0156 | 122.064 | 497.043 | 804.749 |
| gas_nox | XGBoost | — | — | — | 13.3015 | 3.375 | 38.815 | 284.461 | 352.08 |
| gas_nox | LightGBM | — | — | — | 12.5959 | 3.67188 | 38.175 | 293.121 | 1607.41 |
| gas_nox | CatBoost | — | — | — | 12.3902 | 12 | 75.502 | 257.727 | 1094.91 |
| glass | NeuroTabular 2.0.0 | 0.550427 | 0.86982 | 0.91281 | — | 4.98438 | 53.455 | 358.637 | 805.938 |
| glass | NeuroTabular 2.1.0 | 0.550427 | 0.86982 | 0.91281 | — | 6.82812 | 64.112 | 357.203 | 805.984 |
| glass | XGBoost | 0.838889 | 0.665751 | 0.955098 | — | 3.07812 | 37.315 | 297.672 | 2883.44 |
| glass | LightGBM | 0.838889 | 0.715209 | 0.957509 | — | 3.53125 | 39.611 | 281.879 | 620.128 |
| glass | CatBoost | 0.766667 | 0.542862 | 0.957489 | — | 2.54688 | 68.596 | 247.824 | 483.48 |

## Uncertainty and complete records

- [Every planned family/method/seed/regime cell](results/tables/PER_SEED.csv), including failures.
- [Paired quality intervals and point cost ratios](results/tables/PAIRED_COMPARISONS.csv).
- [All timing controls, workloads and intervals](results/tables/TIMING_CONTROLS.csv).
- [Full precision JSON](results/results.json) and [original report](results/REPORT.md).
- [Descriptive task summaries](results/summary.json) and [local point Pareto frontiers](results/decision_support.json).

The two families per task cannot establish a universal ranking. Three seeds are not three independent datasets.
Quality intervals are conditional on frozen models and declared TEST units, without multiplicity correction.
Climate has one held-out study, so no independent-study interval is estimated. Macro metrics and AUC are point estimates.

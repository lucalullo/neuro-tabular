# NeuroTabular 2.1.0 benchmark summary

Default regime only; median over three seeds for each method. Only fully covered external methods are eligible. HPO4, NeuroTabular 2.0.0 and incomplete methods remain in the full report. Values come from the frozen per-seed table.

## Predictive performance

| Dataset | Primary metric | Project | Best eligible external baseline | Baseline value | Directional ratio |
|---|---|---:|---|---:|---:|
| Bank Marketing | NLL (lower is better) | 0.666704 | CatBoost | 0.717406 | 0.929326 |
| Climate Crashes | NLL (lower is better) | 0.157654 | CatBoost | 0.137218 | 1.14893 |
| Anneal | NLL (lower is better) | 0.286735 | XGBoost | 0.033408 | 8.58282 |
| Glass | NLL (lower is better) | 0.779805 | CatBoost | 0.542862 | 1.43647 |
| Fiat 500 | RMSE (lower is better) | 786.206 | CatBoost | 757.44 | 1.03798 |
| Gas Turbine NOx | RMSE (lower is better) | 13.2047 | LightGBM | 12.5541 | 1.05182 |

The quality ratio is project / baseline for lower-is-better loss, and baseline / project for higher-is-better scores. A ratio below one favors the project numerically; this is not a confidence interval or an equivalence test. Ties are broken alphabetically for choosing the resource comparator.

## Computational efficiency

| Dataset | Quality-selected comparator | Total training CPU s: project / baseline | Training RSS MiB: project / baseline | Serialized KiB: project / baseline |
|---|---|---:|---:|---:|
| Bank Marketing | CatBoost | 23.938 / 78.266 | 481.336 / 337.555 | 167.763 / 1328.83 |
| Climate Crashes | CatBoost | 9.969 / 13.469 | 352.602 / 253.551 | 156.64 / 1103.51 |
| Anneal | XGBoost | 10.204 / 5.625 | 358.723 / 281.922 | 211.776 / 425.146 |
| Glass | CatBoost | 9.25 / 15.735 | 351.625 / 261.5 | 152.362 / 3569.04 |
| Fiat 500 | CatBoost | 10.61 / 24.297 | 358.137 / 251.738 | 152.319 / 1076.59 |
| Gas Turbine NOx | LightGBM | 20.36 / 6.891 | 436.371 / 287.754 | 151.919 / 276.518 |

The comparator is selected by predictive quality, not by resource cost. Training and RSS definitions are protocol-specific; do not compare the two repositories as if they used a common workload. Stored bytes are not loaded RAM. See the [benchmark overview](../README.md) for coverage, detailed timing and scientific limitations.

[Machine-readable summary](summary.csv). Regenerate from the recorded source CSV with `python benchmarks/plot_overview.py`. No new measurements are generated.

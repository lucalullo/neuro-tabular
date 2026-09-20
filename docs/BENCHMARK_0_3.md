# NeuroTabular 0.3 benchmark evidence

This document summarizes the frozen development evidence used for NeuroTabular
0.3.0. There are 330 fits: 66 neural and 264 tree fits, using paired 60/20/20
train/validation/test splits and three seeds.

## Aggregate results

Equal dataset/seed means, recomputed from frozen raw predictions. Values in
the product final report are shown first; additional digits below come from
its archived machine-readable aggregates, rather than the release request.

| Task | Datasets × seeds × models | Metrics in final report |
| --- | --- | --- |
| Binary | 5 × 3 × 5 | AUC 0.8461; logloss 0.3906; accuracy 0.8117 |
| Multiclass | 9 × 3 × 5 | Accuracy 0.8275; macro F1 0.7571; logloss 0.4371; macro OVR AUC 0.9499 |
| Regression | 8 × 3 × 5 | R² 0.5320; relative RMSE 0.6274 |

| Task | Metric | Full archived aggregate |
| --- | --- | --- |
| binary | auc | 0.8461200732305805 |
| binary | logloss | 0.39058258431783444 |
| binary | accuracy | 0.8117394824649958 |
| multiclass | accuracy | 0.8275319528406876 |
| multiclass | macro_f1 | 0.7571351431249439 |
| multiclass | logloss | 0.43709917510775587 |
| multiclass | auc | 0.9498761147868271 |
| regression | r2 | 0.5320266908638694 |
| regression | relative_rmse | 0.6273886356670959 |

Relative RMSE divides each test RMSE by the RMSE of that split’s training-mean
predictor. Original-unit RMSE and MAE are not averaged across incompatible scales.

HGB, LightGBM, XGBoost and CatBoost use fixed modest 100-iteration recipes,
without tuning. Tree mean AUC spans 0.8747–0.8892 on binary tasks; multiclass
accuracy spans 0.8572–0.8703; regression R² spans 0.4783–0.5811. These panels
do not establish a universal ranking or state-of-the-art performance.

## Known limitations

- credit-g: mean AUC 0.7020; Titanic: 0.7630, with split sensitivity.
- car: majority-class collapse on two seeds and weak rare-class coverage.
- yeast/glass: difficulty on small imbalanced multiclass problems.
- quake: mean R² −0.0140; all tested tree families also have negative R².
- triazines: mean R² 0.0741, weak and split-sensitive.
- cpu_act and auto_price favor tree baselines in these comparisons.
- kin8nm favors the neural model here, but is a public kinematics simulation
  benchmark, not an observational dataset.
- Timings are contextual single measurements including preprocessing and cold
  starts. Some dermatology timings overlapped binary verification; whole-machine
  isolation is not claimed. No GPU throughput claim is supported.
- DataFrame-only, single-output API; no calibrated-probability API; multiclass
  ROC-AUC is external evaluation only, not an early-stopping metric.
- No same-process concurrent-fit guarantee or proven cross-version persistence.
- Unsafe float32 target/weight ranges are rejected.

Physical CUDA hardware validation was not completed for 0.3.0. The repository
contains Linux and Windows CI configuration, while benchmark timings in this
document are CPU-side development measurements.

The full development archive used to produce these aggregates is intentionally
not distributed in the wheel or sdist. Historical research held-outs were not
used for these 0.3 development benchmarks.

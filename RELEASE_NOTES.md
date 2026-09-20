# NeuroTabular 0.3.0

NeuroTabular 0.3.0 expands the project from binary classification to a shared
multi-task neural tabular core supporting binary classification, multiclass
classification, and single-output regression.

## Highlights

- `NeuroTabularClassifier` now detects binary and multiclass targets automatically.
- New `NeuroTabularRegressor` with train-only target standardization and
  predictions returned in original target units.
- Shared preprocessing, backbone, device handling, weighting, training, and
  checkpoint machinery across tasks.
- Exact binary behavior preserved on the frozen 0.2-to-0.3 equivalence panel.
- Pickle/joblib round trips validated in fresh processes for binary,
  multiclass, and regression models.
- Expanded schema, sample-weight, sklearn, packaging, documentation, and
  installed-artifact validation.

The frozen 0.3 development benchmark contains 330 fits across binary,
multiclass, and regression tasks with fixed tree-model context. It is an
engineering benchmark, not evidence of universal superiority or state-of-the-art
performance. Known weak cases include credit-g, Titanic, car, quake, and
triazines; details are in [docs/BENCHMARK_0_3.md](docs/BENCHMARK_0_3.md).

NeuroTabular remains experimental and pre-1.0. Input is pandas DataFrame-only,
targets are single-output, and cross-version persistence compatibility is not
guaranteed. Physical CUDA hardware validation was not completed for this
release, although device/fallback paths are covered by tests.

See the [usage guide](docs/USAGE.md), [API reference](docs/API.md), and
[changelog](CHANGELOG.md) for details.

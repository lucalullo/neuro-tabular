# NeuroTabular

[![PyPI version](https://img.shields.io/pypi/v/neurotabular)](https://pypi.org/project/neurotabular/)

**Neural networks for tabular data, with automatic preprocessing and a scikit-learn-style API.**

## Why NeuroTabular?

NeuroTabular uses PyTorch to train neural models for binary classification, multiclass classification and single-output regression on pandas DataFrames. Numerical and categorical features, missing values, validation and early stopping are handled inside the estimator, with preprocessing learned only from training data.

## Project status

Version **2.1.0** changes the default regression stopping rule to eight validation checks, with at most 30 epochs, and improves extreme-value standardization and very small positive sample-weight handling. Classifier defaults are retained. Optional internal K-fold diagnostics and validation weights are available. No general classification gain is claimed. See the [release notes](https://github.com/lucalullo/neuro-tabular/blob/main/RELEASE_NOTES.md).

## Install

```bash
python -m pip install neurotabular
```

Python 3.11-3.14 is the recommended installation range. For the exact supported combinations and CPU/CUDA scope, see the support scope. See the [installation and compatibility guidance](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/SUPPORT_SCOPE.md).

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from neurotabular import NeuroTabularClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = NeuroTabularClassifier().fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
```

The classifier also supports multiclass targets. For regression, use `NeuroTabularRegressor` with a numerical target. See [usage and examples](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md). This example demonstrates the API, not the benchmark protocol.

## Highlights

- Binary and multiclass classification; single-output regression.
- Automatic mixed-DataFrame preprocessing and missing-value handling.
- scikit-learn-style estimators, cloning, pipelines and cross-validation.
- Validation, early stopping, batching and checkpoint restoration.
- Optional internal K-fold diagnostics (`cv=K`), with K+1 training passes (doubled for `full_data_refit=True`) and one final network.
- Sample weights, optional balanced class weights and explicit validation weights (`cv=None`).
- CPU, optional CUDA selection and pickle/joblib persistence.

## Benchmarks

| Scope | Evidence |
|---|---|
| Competitive evidence | 2.1.0 vs 2.0.0 and six external methods; six families, three seeds, default and four-candidate tuning. |
| Coverage | 181 of 288 final TEST endpoints measured; 107 unavailable. |
| Technical evidence | Only 2 of 36 dedicated inference timing controls valid. |
| Reproduction | Frozen executable protocol; exact environment, source and wheel prerequisites apply. |

The [benchmark overview](https://github.com/lucalullo/neuro-tabular/blob/main/benchmarks/README.md) links predictive quality, computational costs, complete figures, methodology, failures and numerical data. Default and tuned results, historical results and technical timing controls must be interpreted separately.

## When to use it

Use it for neural learning on mixed pandas data with a familiar estimator interface. Bank Marketing has lower default median NLL than the three fully measured tree comparators; Climate has higher default median balanced accuracy than those trees. These are dataset- and metric-specific observations.

## When to consider alternatives

Consider alternatives for multi-output learning, other neural architectures or advanced AutoML. The measured trees perform substantially better on Anneal and Glass. RealMLP and TabM comparisons are unavailable and FT-Transformer coverage is partial.

## Limitations

- Requires a pandas DataFrame; multilabel and multi-output learning are unsupported.
- No universal accuracy, latency, throughput or RAM advantage is established.
- Physical NVIDIA GPU validation and formal end-to-end no-network certification are incomplete.
- At most 128 categorical columns; high cardinality may substantially increase memory.
- Concurrent `fit()` calls in one process are not guaranteed thread-safe.

## Documentation

- [Usage and examples](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md)
- [API reference](https://github.com/lucalullo/neuro-tabular/blob/main/docs/API.md)
- [Support scope](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/SUPPORT_SCOPE.md)
- [API stability and maintenance policies](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/README.md)
- [Internal K-fold diagnostics](https://github.com/lucalullo/neuro-tabular/blob/main/docs/INTERNAL_CV.md)
- [Benchmarks](https://github.com/lucalullo/neuro-tabular/blob/main/benchmarks/README.md)
- [Historical evidence](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md)
- [Performance](https://github.com/lucalullo/neuro-tabular/blob/main/docs/PERFORMANCE_0_3_1.md)
- [Release notes](https://github.com/lucalullo/neuro-tabular/blob/main/RELEASE_NOTES.md)
- [Changelog](https://github.com/lucalullo/neuro-tabular/blob/main/CHANGELOG.md)

## License

NeuroTabular is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/neuro-tabular/blob/main/LICENSE). See [dependency notices](https://github.com/lucalullo/neuro-tabular/blob/main/THIRD_PARTY_NOTICES.md) and the [legal/naming note](https://github.com/lucalullo/neuro-tabular/blob/main/LEGAL.md).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

# NeuroTabular

[![PyPI version](https://img.shields.io/pypi/v/neurotabular)](https://pypi.org/project/neurotabular/)

**Neural networks for tabular data, with automatic preprocessing and a scikit-learn-style API.**

NeuroTabular uses PyTorch to train neural models for binary classification,
multiclass classification, and single-output regression on pandas DataFrames.
It handles numerical and categorical columns, missing values, and reusable
training-only preprocessing inside the estimator.

## Why NeuroTabular?

Train neural networks on pandas DataFrames without building a custom PyTorch
preprocessing pipeline or training loop. NeuroTabular handles mixed features,
missing values, validation, and early stopping behind a scikit-learn-style API.

### Project status

A young library under active development and evaluation. Core classification
and regression functionality is implemented and extensively tested, and the
public API follows a [documented stability policy](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/API_STABILITY_POLICY.md).
Hardware coverage and performance across different workloads still need broader validation.

### When to use it

For neural learning on mixed pandas data when you want a familiar estimator
interface and integration with scikit-learn workflows.

### When to consider alternatives

Consider other tools if you need multiple neural architectures, multi-output
learning, advanced AutoML, or more extensively benchmarked predictive performance.

## Install

```bash
python -m pip install neurotabular
```

Python 3.11–3.14 is the recommended range for standard installation. See the
[support scope](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/SUPPORT_SCOPE.md)
for tested environments and other runtime combinations.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from neurotabular import NeuroTabularClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = NeuroTabularClassifier(random_state=42, max_epochs=10)
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
```

The same classifier handles multiclass targets. For regression, use
`NeuroTabularRegressor`. For numerical and categorical features, missing
values, validation sets, and all three tasks, see the
[complete mixed-data example](https://github.com/lucalullo/neuro-tabular/blob/main/examples/mixed_dataframe_workflow.py).

## Highlights

- Automatic preprocessing of numerical and categorical pandas columns, including missing values.
- Binary and multiclass classification, plus single-output regression.
- scikit-learn-style estimators, cloning, pipelines, and cross-validation.
- Validation, early stopping, batching, and checkpoint restoration.
- `sample_weight` for all tasks and optional `class_weight="balanced"` for classification.
- CPU execution and optional CUDA selection; pickle/joblib model persistence.

## Limitations

- Input must be a pandas DataFrame; multi-output and multilabel tasks are not supported.
- No claim of universal accuracy or speed advantages over established tabular models.
- Physical NVIDIA GPU validation is incomplete; some hardware environments remain outside the verified scope.
- Very high-cardinality categorical features can increase memory use substantially.
- Concurrent `fit()` calls within a process are not guaranteed thread-safe.

See the [benchmark evidence](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md)
for measured results and dataset-specific failure cases.

## Documentation

- [Usage and examples](https://github.com/lucalullo/neuro-tabular/blob/main/docs/USAGE.md)
- [API reference](https://github.com/lucalullo/neuro-tabular/blob/main/docs/API.md)
- [Support scope](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/SUPPORT_SCOPE.md)
- [API stability and maintenance policies](https://github.com/lucalullo/neuro-tabular/blob/main/docs/policies/README.md)
- [Benchmarks](https://github.com/lucalullo/neuro-tabular/blob/main/docs/BENCHMARK_0_3.md)
- [Performance measurements](https://github.com/lucalullo/neuro-tabular/blob/main/docs/PERFORMANCE_0_3_1.md)
- [Release notes](https://github.com/lucalullo/neuro-tabular/blob/main/RELEASE_NOTES.md)

## License

NeuroTabular is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/neuro-tabular/blob/main/LICENSE). Dependency notices
are in [THIRD_PARTY_NOTICES.md](https://github.com/lucalullo/neuro-tabular/blob/main/THIRD_PARTY_NOTICES.md). See [LEGAL.md](https://github.com/lucalullo/neuro-tabular/blob/main/LEGAL.md)
for the project's legal/naming note.

## Author

Created by [Luca Lullo](https://github.com/lucalullo).

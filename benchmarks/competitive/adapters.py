"""Explicit single-fit adapters; importing this module does not load estimators."""

from __future__ import annotations

from importlib import import_module
from typing import Any

METHODS = (
    "neurotabular200",
    "neurotabular210",
    "realmlp",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabm",
    "ft_transformer",
)


def estimator_spec(method: str, task: str, seed: int, params: dict | None = None):
    """Return the exact class/kwargs without importing or fitting the class."""
    if method not in METHODS or task not in {"binary", "multiclass", "regression"}:
        raise ValueError("Unknown method or task")
    suffix = "Regressor" if task == "regression" else "Classifier"
    changes = dict(params or {})
    if set(changes) & {
        "random_state",
        "random_seed",
        "device",
        "n_jobs",
        "n_threads",
        "thread_count",
        "n_cv",
        "n_refit",
        "n_repeats",
        "calibration_method",
    }:
        raise ValueError("Search cannot alter seeds, resources, refits or calibration")
    if method.startswith("neurotabular"):
        return (
            "neurotabular",
            "NeuroTabular" + suffix,
            {"random_state": seed, **changes},
        )
    if method == "xgboost":
        return (
            "xgboost",
            "XGB" + suffix,
            {"n_jobs": 1, "random_state": seed, "enable_categorical": True, **changes},
        )
    if method == "lightgbm":
        return (
            "lightgbm",
            "LGBM" + suffix,
            {"n_jobs": 1, "random_state": seed, "verbosity": -1, **changes},
        )
    if method == "catboost":
        return (
            "catboost",
            "CatBoost" + suffix,
            {
                "random_seed": seed,
                "thread_count": 1,
                "verbose": False,
                "allow_writing_files": False,
                **changes,
            },
        )
    prefix = {"realmlp": "RealMLP_TD_", "tabm": "TabM_D_", "ft_transformer": "FTT_D_"}[
        method
    ]
    return (
        "pytabkit",
        prefix + suffix,
        {
            "device": "cpu",
            "random_state": seed,
            "n_threads": 1,
            "n_cv": 1,
            "n_refit": 0,
            "n_repeats": 1,
            "calibration_method": None,
            **changes,
        },
    )


def category_token(value: Any) -> str:
    import pandas as pd

    if pd.isna(value):
        return "missing:"
    return type(value).__name__ + ":" + str(value)


class FeatureAdapter:
    """Fixed schema, TRAIN-only imputation/encoding, no target information."""

    def __init__(self, method: str, categorical: list[str]):
        self.method = method
        self.categorical = list(categorical)

    def fit(self, X):
        self.columns_ = list(X.columns)
        if not set(self.categorical).issubset(self.columns_):
            raise ValueError("Categorical schema contains unknown columns")
        self.numeric_ = [name for name in self.columns_ if name not in self.categorical]
        if self.method.startswith("neurotabular"):
            return self
        if self.method in {"realmlp", "tabm", "ft_transformer"}:
            self.medians_ = X[self.numeric_].median().fillna(0.0)
        if self.method in {"xgboost", "lightgbm"}:
            self.vocabulary_ = {
                name: sorted(
                    X[name].dropna().astype(object).map(category_token).unique()
                )
                for name in self.categorical
            }
        return self

    def _frame(self, X):
        import numpy as np
        import pandas as pd

        if list(X.columns) != self.columns_:
            raise ValueError("Feature schema/order changed")
        result = X.copy()
        for column in self.categorical:
            if self.method in {"xgboost", "lightgbm"}:
                result[column] = (
                    result[column]
                    .astype(object)
                    .map(
                        lambda value: (
                            np.nan if pd.isna(value) else category_token(value)
                        )
                    )
                )
                # Preserve the TRAIN-only vocabulary: unknown tokens are missing.
                # Explicit masking avoids the newer pandas deprecation warning.
                vocabulary = self.vocabulary_[column]
                tokens = result[column]
                result[column] = pd.Categorical(
                    tokens.where(tokens.isin(vocabulary)), categories=vocabulary
                )
            else:
                result[column] = result[column].astype(object).map(category_token)
        if self.method in {"realmlp", "tabm", "ft_transformer"}:
            result[self.numeric_] = result[self.numeric_].fillna(self.medians_)
        return result

    def transform(self, X):
        if self.method.startswith("neurotabular"):
            if list(X.columns) != self.columns_:
                raise ValueError("Feature schema/order changed")
            return X
        return self._frame(X)


class BenchmarkModel:
    """One estimator with an explicit adapter, no tuning or refit hidden here."""

    def __init__(self, method, task, seed, categorical, params=None):
        self.method, self.task, self.seed = method, task, seed
        self.categorical = list(categorical)
        self.params = dict(params or {})
        self.fit_calls_ = 0

    def fit(self, X, y):
        import numpy as np

        if self.fit_calls_:
            raise ValueError("Each benchmark object admits exactly one fit")
        module, name, kwargs = estimator_spec(
            self.method, self.task, self.seed, self.params
        )
        imported = import_module(module)
        if self.method.startswith("neurotabular"):
            expected = "2.0.0" if self.method == "neurotabular200" else "2.1.0"
            if imported.__version__ != expected:
                raise ValueError("Wrong NeuroTabular version loaded before fit")
            self.loaded_module_ = {
                "version": imported.__version__,
                "file": imported.__file__,
            }
        estimator_class = getattr(imported, name)
        self.estimator_ = estimator_class(**kwargs)
        self.adapter_ = FeatureAdapter(self.method, self.categorical).fit(X)
        features = self.adapter_.transform(X)
        target = np.asarray(y)
        if self.task != "regression" and self.method == "xgboost":
            self.labels_ = np.unique(target)
            lookup = {label: index for index, label in enumerate(self.labels_)}
            target = np.asarray([lookup[label] for label in target])
        self.fit_calls_ += 1
        if self.method == "catboost":
            self.estimator_.fit(features, target, cat_features=self.categorical)
        else:
            self.estimator_.fit(features, target)
        self.public_constructor_ = {"module": module, "class": name, "kwargs": kwargs}
        self.class_order_ = (
            None if self.task == "regression" else list(self.estimator_.classes_)
        )
        if self.method == "xgboost" and self.task != "regression":
            self.class_order_ = self.labels_[
                np.asarray(self.estimator_.classes_, dtype=int)
            ].tolist()
        return self

    def predict(self, X):
        import numpy as np

        result = np.asarray(
            self.estimator_.predict(self.adapter_.transform(X))
        ).reshape(-1)
        if self.task != "regression" and self.method == "xgboost":
            result = self.labels_[result.astype(int)]
        return result

    def predict_proba(self, X):
        import numpy as np

        if self.task == "regression":
            raise ValueError("Regressors do not have predict_proba")
        return np.asarray(self.estimator_.predict_proba(self.adapter_.transform(X)))

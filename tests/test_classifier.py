import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from neurotabular import NeuroTabularClassifier


def test_fit_predict_and_fitted_attributes(mixed_binary_data, fast_model):
    X, y = mixed_binary_data
    returned = fast_model.fit(X, y)
    probabilities = fast_model.predict_proba(X.iloc[:11])
    predictions = fast_model.predict(X.iloc[:11])

    assert returned is fast_model
    assert np.array_equal(fast_model.classes_, [0, 1])
    assert probabilities.shape == (11, 2)
    assert predictions.shape == (11,)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert fast_model.device_ == "cpu"
    assert fast_model.best_epoch_ == 1
    assert fast_model.n_iter_ == 1
    assert fast_model.n_parameters_ > 0
    assert fast_model.fit_time_ >= fast_model.preprocessing_time_ > 0.0
    assert fast_model.training_time_ > 0.0
    assert fast_model.validation_time_ > 0.0
    assert set(fast_model.profile_) == {
        "preprocessing",
        "target_preparation_seconds",
        "device",
        "training",
    }
    assert fast_model.device_info_["requested_device"] == "cpu"
    assert fast_model.device_info_["resolved_device"] == "cpu"


@pytest.mark.parametrize("labels", [("no", "yes"), (-3, 8)])
def test_predict_preserves_original_labels(
    mixed_binary_data, fast_model_kwargs, labels
):
    X, y = mixed_binary_data
    mapped = np.where(y == 0, labels[0], labels[1])
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, mapped)
    assert set(model.predict(X.iloc[:8])).issubset(set(labels))
    assert list(model.classes_) == sorted(labels)


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
def test_supported_feature_compositions(kind, fast_model_kwargs):
    n = 60
    if kind == "numeric":
        X = pd.DataFrame({"x": np.linspace(-2, 2, n), "z": np.arange(n)})
    elif kind == "categorical":
        X = pd.DataFrame(
            {
                "city": np.resize(["a", "b", None], n),
                "flag": np.resize([True, False], n),
            }
        )
    else:
        X = pd.DataFrame(
            {"x": np.linspace(-2, 2, n), "city": np.resize(["a", "b", None], n)}
        )
    y = np.resize([0, 1], n)
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    assert np.isfinite(model.predict_proba(X.iloc[:7])).all()


def test_string_boolean_category_and_integer_categorical_fit(fast_model_kwargs):
    n = 60
    X = pd.DataFrame(
        {
            "string": pd.Series(np.resize(["a", "b"], n), dtype="string"),
            "category": pd.Categorical(np.resize(["x", "y", "z"], n)),
            "boolean": np.resize([True, False], n),
            "store_id": np.resize([10, 20, 30], n),
        }
    )
    y = np.resize([0, 1], n)
    model = NeuroTabularClassifier(
        **fast_model_kwargs, categorical_features=["store_id"]
    ).fit(X, y)
    assert model.categorical_features_ == [
        "string",
        "category",
        "boolean",
        "store_id",
    ]


def test_unseen_category_works_at_prediction(fast_model_kwargs):
    X = pd.DataFrame({"city": np.resize(["Rome", "Milan"], 40)})
    y = np.resize([0, 1], 40)
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    probabilities = model.predict_proba(pd.DataFrame({"city": ["Turin", None]}))
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()


def test_prediction_reorders_columns_and_rejects_schema_changes(
    mixed_binary_data, fast_model
):
    X, y = mixed_binary_data
    fast_model.fit(X, y)
    expected = fast_model.predict_proba(X)
    assert np.allclose(fast_model.predict_proba(X[list(reversed(X.columns))]), expected)
    with pytest.raises(ValueError, match="missing columns"):
        fast_model.predict(X.drop(columns=["age"]))
    with pytest.raises(ValueError, match="unexpected columns"):
        fast_model.predict(X.assign(extra=1))
    duplicate = X.copy()
    duplicate.columns = ["age", "age", "region", "active"]
    with pytest.raises(ValueError, match="unique"):
        fast_model.predict(duplicate)


def test_explicit_eval_set_skips_internal_split_and_is_leakage_safe(
    monkeypatch, fast_model_kwargs
):
    X_train = pd.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0] * 8, "city": ["Rome"] * 32})
    y_train = np.resize([0, 0, 1, 1], 32)
    X_valid = pd.DataFrame({"value": [1000.0, 2000.0] * 4, "city": ["Turin"] * 8})
    y_valid = np.resize([0, 1], 8)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("internal split was called")

    monkeypatch.setattr("neurotabular.classifier.train_test_split", fail_if_called)
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(
        X_train, y_train, eval_set=(X_valid, y_valid)
    )
    assert model._preprocessor_.numeric_medians_[0] == 0.0
    assert "Turin" not in model._preprocessor_.category_vocabs_["city"]
    encoded = model._preprocessor_.transform(X_valid).categorical[:, 0]
    assert np.array_equal(encoded, np.ones(len(X_valid), dtype=np.int64))


def test_full_data_refit_uses_all_rows_after_epoch_selection(
    mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    model = NeuroTabularClassifier(
        **{**fast_model_kwargs, "full_data_refit": True}
    ).fit(X, y)
    assert model.full_data_refit_ is True
    assert model._preprocessor_.fit_sample_count_ == len(X)
    assert model.profile_["full_data_refit"]["training"]["epochs"] == 1


def test_external_validation_does_not_repeat_full_data_training(
    mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    model = NeuroTabularClassifier(
        **{**fast_model_kwargs, "full_data_refit": True}
    ).fit(X.iloc[:72], y[:72], eval_set=(X.iloc[72:], y[72:]))
    assert model.full_data_refit_ is False
    assert "full_data_refit" not in model.profile_


@pytest.mark.parametrize("metric", ["loss", "roc_auc", "accuracy"])
def test_supported_validation_metrics(metric, fast_model_kwargs):
    values = np.linspace(-3.0, 3.0, 60)
    X = pd.DataFrame({"value": values})
    y = (values > 0.0).astype(int)
    train = np.r_[0:24, 36:60]
    valid = np.arange(24, 36)
    model = NeuroTabularClassifier(**fast_model_kwargs, eval_metric=metric).fit(
        X.iloc[train], y[train], eval_set=(X.iloc[valid], y[valid])
    )
    assert model.history_[0]["validation_score"] == pytest.approx(model.best_score_)
    if metric == "roc_auc":
        restored = roc_auc_score(y[valid], model.predict_proba(X.iloc[valid])[:, 1])
        assert restored == pytest.approx(model.best_score_)


def test_predict_before_fit_raises_not_fitted(fast_model):
    with pytest.raises(NotFittedError):
        fast_model.predict(pd.DataFrame({"x": [1.0]}))


def test_non_binary_and_missing_targets_are_rejected(fast_model_kwargs):
    X = pd.DataFrame({"x": np.arange(30)})
    with pytest.raises(ValueError, match="binary targets only"):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, np.resize([0, 1, 2], 30))
    y = np.resize([0.0, 1.0], 30)
    y[0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)


def test_auto_device_and_unavailable_cuda(mixed_binary_data, fast_model_kwargs):
    X, y = mixed_binary_data
    parameters = {**fast_model_kwargs, "device": "auto"}
    model = NeuroTabularClassifier(**parameters).fit(X, y)
    if torch.cuda.is_available():
        assert model.device_ == "cpu" or model.device_.startswith("cuda")
    else:
        assert model.device_ == "cpu"
    if not torch.cuda.is_available():
        parameters["device"] = "cuda"
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            NeuroTabularClassifier(**parameters).fit(X, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_smoke(mixed_binary_data, fast_model_kwargs):
    X, y = mixed_binary_data
    try:
        model = NeuroTabularClassifier(**{**fast_model_kwargs, "device": "cuda"}).fit(
            X, y
        )
    except RuntimeError as exc:
        if "compatibility probe failed" in str(exc):
            pytest.skip(f"CUDA is visible but unusable: {exc}")
        raise
    assert model.device_.startswith("cuda")
    assert np.isfinite(model.predict_proba(X.iloc[:5])).all()

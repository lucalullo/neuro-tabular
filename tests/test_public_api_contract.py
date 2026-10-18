"""Executable public lifecycle contracts; no research performance claims."""

import copy
import inspect
import warnings

import numpy as np
import pytest
from p40_helpers import assert_rng_same, data, kwargs, rng_snapshot
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import (
    check_estimators_unfitted,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_set_params,
)
from sklearn.utils.validation import check_is_fitted

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor, __version__


def cls(task):
    return NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier


@pytest.mark.parametrize(
    "estimator_class", [NeuroTabularClassifier, NeuroTabularRegressor]
)
def test_constructor_repr_clone_and_selective_sklearn_checks(estimator_class):
    model = estimator_class(categorical_features=["group"], **kwargs())
    assert isinstance(model, BaseEstimator)
    assert isinstance(
        model,
        RegressorMixin if estimator_class is NeuroTabularRegressor else ClassifierMixin,
    )
    assert estimator_class.__name__ in repr(model)
    parameters = model.get_params()
    assert set(parameters) == set(inspect.signature(estimator_class).parameters)
    assert len(parameters) == (21 if estimator_class is NeuroTabularRegressor else 22)
    assert clone(model).get_params() == parameters
    assert model.set_params(hidden_dim=10) is model
    assert model.hidden_dim == 10
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(nonexistent_option=1)
    for check in (
        check_no_attributes_set_in_init,
        check_get_params_invariance,
        check_set_params,
        check_estimators_unfitted,
    ):
        check(estimator_class.__name__, clone(model))


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_lifecycle_refit_schema_rng_and_set_params(task):
    X, y = data(task)
    model = cls(task)(**kwargs())
    assert not any(k.endswith("_") for k in vars(model))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.predict(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        before = rng_snapshot()
        assert model.fit(X, y) is model
        assert_rng_same(before)
        assert not caught
    first = model.predict(X)
    assert first.shape == (len(X),)
    assert first.dtype == (
        np.dtype("float64") if task == "regression" else np.asarray(y).dtype
    )
    np.testing.assert_array_equal(first, model.predict(X.iloc[:, ::-1]))
    if task != "regression":
        probabilities = model.predict_proba(X)
        assert probabilities.dtype == np.float64
        assert probabilities.shape == (len(X), len(np.unique(y)))
        np.testing.assert_array_equal(model.classes_, np.unique(y))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)
    else:
        assert not hasattr(model, "predict_proba")
    check_is_fitted(model)
    assert model._fitted_neurotabular_version_ == __version__
    assert model.n_features_in_ == len(X.columns)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    for bad in (X.drop(columns="count"), X.assign(extra=1)):
        with pytest.raises(ValueError, match="schema mismatch"):
            model.predict(bad)
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict(X.to_numpy())
    changed_dtype = X.copy()
    changed_dtype["group"] = changed_dtype["group"].astype("string")
    changed_dtype.loc[0, "group"] = "new-category"
    assert np.isfinite(
        model.predict_proba(changed_dtype)
        if task != "regression"
        else model.predict(changed_dtype)
    ).all()
    # Detached diagnostics can be edited without changing the encoder/model.
    schema = copy.deepcopy(model.input_schema_)
    model.input_schema_[0].update(
        name="edited-diagnostic", kind="categorical", dtype="edited"
    )
    np.testing.assert_array_equal(first, model.predict(X))
    model.input_schema_ = schema
    # Existing BaseEstimator semantics: pending settings, fitted predictor intact.
    old_network = model._model_
    old_device = model.device_
    assert (
        model.set_params(hidden_dim=12, device="cuda:999", min_category_count=5)
        is model
    )
    assert model.get_params()["hidden_dim"] == 12
    assert model._model_ is old_network and model.device_ == old_device
    np.testing.assert_array_equal(first, model.predict(X))
    model.set_params(**kwargs(), min_category_count=2)
    clone_after_fit = clone(model)
    with pytest.raises(NotFittedError):
        check_is_fitted(clone_after_fit)
    assert not hasattr(clone_after_fit, "_fitted_neurotabular_version_")
    # Same data => new model object, same fixed-runtime prediction.
    before = rng_snapshot()
    model.fit(X, y)
    assert_rng_same(before)
    assert model._model_ is not old_network
    np.testing.assert_array_equal(first, model.predict(X))
    # Changed data and schema replace learned state; no old vocabulary survives.
    new_X = X.copy()
    new_X["group"] = "new-training-group"
    before = rng_snapshot()
    model.fit(new_X, y)
    assert_rng_same(before)
    assert set(model._preprocessor_.category_vocabs_["group"]) == {"new-training-group"}
    new_schema = new_X.rename(columns={"value": 10})
    before = rng_snapshot()
    model.fit(new_schema, y)
    assert_rng_same(before)
    assert not hasattr(model, "feature_names_in_")
    assert model.input_schema_[0]["name"] == 10
    with pytest.raises(ValueError, match="schema mismatch"):
        model.predict(X)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_late_failed_refit_preserves_internal_version_and_schema(
    task, failure_type, monkeypatch
):
    X, y = data(task)
    model = cls(task)(**kwargs()).fit(X, y)
    before_prediction = model.predict(X)
    old_network, old_encoder = model._model_, model._preprocessor_
    before_schema = copy.deepcopy(model.input_schema_)
    before_version = model._fitted_neurotabular_version_
    before = rng_snapshot()

    def late_failure(*args, **kw):
        import random

        import torch

        random.random()
        np.random.random()
        torch.rand(1)
        raise failure_type("controlled P40 late failure")

    monkeypatch.setattr("neurotabular._base.train_model", late_failure)
    with pytest.raises(failure_type, match="controlled P40 late failure"):
        model.fit(X.rename(columns={"value": "new"}), y)
    assert_rng_same(before)
    assert model._model_ is old_network and model._preprocessor_ is old_encoder
    assert model.input_schema_ == before_schema
    assert model._fitted_neurotabular_version_ == before_version
    np.testing.assert_array_equal(before_prediction, model.predict(X))


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_randomized_search_dataframe_pipeline(task):
    X, y = data(task)
    transformer = ColumnTransformer(
        [("numeric", StandardScaler(), ["value"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    pipeline = Pipeline([("columns", transformer), ("model", cls(task)(**kwargs()))])
    search = RandomizedSearchCV(
        pipeline,
        {"model__hidden_dim": [8, 10]},
        n_iter=1,
        cv=2,
        random_state=17,
        error_score="raise",
        n_jobs=1,
    ).fit(X, y)
    fitted = search.best_estimator_.named_steps["model"]
    assert search.predict(X).shape == (len(X),)
    assert fitted.n_features_in_ == len(X.columns)
    assert set(fitted.feature_names_in_) == set(X.columns)
    assert fitted._fitted_neurotabular_version_ == __version__

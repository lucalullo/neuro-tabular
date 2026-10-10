"""Public fitted metadata must describe the committed model, never drive it."""

import copy
import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("full_refit", [False, True])
def test_fitted_schema_lifecycle(task, full_refit, fast_model_kwargs):
    X = pd.DataFrame(
        {
            "value": pd.Series(np.arange(48), dtype="Float64"),
            "group": pd.Categorical(
                np.resize(["a", "b", None], 48),
                categories=["a", "b", "unobserved"],
            ),
            "active": pd.Series(np.resize([True, False, None], 48), dtype="boolean"),
        }
    )
    y = (
        np.arange(48) * 0.25
        if task == "regression"
        else np.arange(48) % (3 if task == "multiclass" else 2)
    )
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    model = cls(**{**fast_model_kwargs, "full_data_refit": full_refit})
    assert not hasattr(model, "input_schema_")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    model.fit(X, y)
    check_is_fitted(model)
    assert model.n_features_in_ == 3
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    schema = copy.deepcopy(model.input_schema_)
    assert schema[0] == {"name": "value", "dtype": "Float64", "kind": "numeric"}
    assert schema[1] == {
        "name": "group",
        "dtype": "category",
        "kind": "categorical",
        "n_known_categories": 2,
        "n_rare_categories": 0,
        "encoded_cardinality": 5,
    }
    assert schema[2]["kind"] == "categorical"
    assert schema[2]["dtype"] == "boolean"
    assert model.full_data_refit_ is full_refit
    model.predict(X.iloc[:, ::-1])
    if task != "regression":
        model.predict_proba(X)
    assert model.input_schema_ == schema
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "input_schema_")
    with pytest.raises(NotFittedError):
        check_is_fitted(cloned)
    second = clone(model).fit(X, y)
    assert second.input_schema_ == schema


def test_schema_counts_exclude_validation_zero_weight_and_unobserved_levels(
    fast_model_kwargs,
):
    X = pd.DataFrame({"group": pd.Categorical(["ignored", "rare"] + ["a", "b"] * 23)})
    y = np.arange(48) % 2
    weights = np.ones(48)
    weights[0] = 0
    validation = pd.DataFrame({"group": ["validation_only", None]})
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(
        X, y, sample_weight=weights, eval_set=(validation, [0, 1])
    )
    assert model.input_schema_[0]["n_known_categories"] == 3
    assert model.input_schema_[0]["n_rare_categories"] == 1
    assert model.input_schema_[0]["encoded_cardinality"] == 5
    assert model.n_ignored_samples_ == 1
    # The public snapshot is detached from preprocessing and is never read by predict.
    before = model.predict_proba(X)
    model.input_schema_[0]["n_known_categories"] = -1
    np.testing.assert_array_equal(model.predict_proba(X), before)


@pytest.mark.parametrize("full_refit", [False, True])
def test_schema_counts_follow_final_internal_or_full_data_preprocessor(
    full_refit, fast_model_kwargs
):
    X = pd.DataFrame({"group": pd.Categorical([f"value_{i}" for i in range(48)])})
    model = NeuroTabularClassifier(
        **{**fast_model_kwargs, "full_data_refit": full_refit}
    ).fit(X, np.arange(48) % 2)
    expected = 48 if full_refit else 38  # 10 validation rows at fraction 0.2.
    assert model.input_schema_[0]["n_known_categories"] == expected
    assert model.input_schema_[0]["n_rare_categories"] == expected
    assert model.input_schema_[0]["encoded_cardinality"] == 3


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_late_failed_refit_preserves_metadata_model_and_all_caller_rng(
    task, monkeypatch, mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    if task == "multiclass":
        y = np.arange(len(X)) % 3
    elif task == "regression":
        y = np.arange(len(X)) * 0.1
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    model = cls(**fast_model_kwargs).fit(X, y)
    previous_model = model._model_
    schema = copy.deepcopy(model.input_schema_)
    history = copy.deepcopy(model.history_)
    expected = model.predict(X)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    def fail_after_consuming_rng(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("controlled late fit failure")

    monkeypatch.setattr("neurotabular._base.train_model", fail_after_consuming_rng)
    with pytest.raises(RuntimeError, match="controlled late fit failure"):
        model.fit(X.rename(columns={X.columns[0]: "new_name"}), y)
    assert model._model_ is previous_model
    assert model.input_schema_ == schema
    assert model.history_ == history
    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    np.testing.assert_array_equal(model.predict(X), expected)


def test_schema_retains_nonstring_labels_without_claiming_feature_names(
    mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    X.columns = [10, 20, ("region", 1), ("active", 2)]
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    assert not hasattr(model, "feature_names_in_")
    assert tuple(row["name"] for row in model.input_schema_) == tuple(X.columns)
    np.testing.assert_array_equal(
        model.predict_proba(X), model.predict_proba(X.iloc[:, ::-1])
    )

"""Unrepresentable conversion fails deterministically before training."""

import random
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import neurotabular._base as base_module
from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor
from neurotabular._base import _BaseNeuroTabularEstimator as Base
from neurotabular.preprocessing import TabularPreprocessor


def invalid_values(case, n=2):
    if case == "bigint_list":
        return [10**400] + [1] * (n - 1)
    if case == "bigint_object":
        return np.asarray([10**400] + [1] * (n - 1), dtype=object)
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider range on this platform")
    return np.asarray([np.longdouble("1e400")] + [1] * (n - 1), dtype=np.longdouble)


@pytest.mark.parametrize("case", ["bigint_list", "bigint_object", "wide_float"])
@pytest.mark.parametrize("route", ["weight", "target"])
@pytest.mark.parametrize("policy", ["always", "error"])
def test_unrepresentable_target_and_weight(case, route, policy):
    values = invalid_values(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        with pytest.raises(ValueError):
            if route == "weight":
                Base._validate_sample_weight(values, 2)
            else:
                NeuroTabularRegressor._numeric_target(values, 2)
        assert not caught


@pytest.mark.parametrize("policy", ["always", "error"])
def test_wider_feature_overflow_is_controlled(policy):
    values = invalid_values("wide_float")
    prep = TabularPreprocessor().fit(pd.DataFrame({"x": [0.0, 1.0]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        with pytest.raises(ValueError):
            prep.transform(pd.DataFrame({"x": values}))
        assert not caught


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, object])
@pytest.mark.parametrize("route", ["weight", "target"])
@pytest.mark.parametrize("policy", ["always", "error"])
def test_ordinary_conversions_are_exact(dtype, route, policy):
    values = np.asarray([1, 2], dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        result = (
            Base._validate_sample_weight(values, 2)
            if route == "weight"
            else NeuroTabularRegressor._numeric_target(values, 2)
        )
        np.testing.assert_array_equal(result, np.asarray([1.0, 2.0]))
        assert not caught


@pytest.mark.parametrize("case", ["bigint_list", "bigint_object", "wide_float"])
@pytest.mark.parametrize("route", ["weight", "target"])
@pytest.mark.parametrize("policy", ["always", "error"])
def test_public_fit_rejects_before_training_and_restores_rng(
    case, route, policy, monkeypatch
):
    values = invalid_values(case, 16)

    def no_training(*args, **kwargs):
        raise AssertionError("Invalid input reached the neural trainer")

    monkeypatch.setattr(base_module, "train_model", no_training)
    monkeypatch.setattr(base_module, "train_fixed_epochs", no_training)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 16)})
    model = NeuroTabularClassifier() if route == "weight" else NeuroTabularRegressor()
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        with pytest.raises(ValueError):
            if route == "weight":
                model.fit(X, np.arange(16) % 2, sample_weight=values)
            else:
                model.fit(X, values)
        assert not caught
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])
    assert not hasattr(model, "_model_")


@pytest.mark.parametrize("route", ["weight", "target", "feature"])
@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_conversion_restores_caller_error_policy(route, policy):
    values = invalid_values("wide_float")
    prep = TabularPreprocessor().fit(pd.DataFrame({"x": [0.0, 1.0]}))
    before = np.geterr().copy()
    with np.errstate(over=policy, invalid=policy):
        inside = np.geterr().copy()
        with pytest.raises(ValueError):
            if route == "weight":
                Base._validate_sample_weight(values, 2)
            elif route == "target":
                NeuroTabularRegressor._numeric_target(values, 2)
            else:
                prep.transform(pd.DataFrame({"x": values}))
        assert np.geterr() == inside
    assert np.geterr() == before

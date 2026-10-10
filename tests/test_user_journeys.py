"""Focused realistic pandas/error/persistence support journeys."""

import json
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from p40_helpers import data, kwargs
from sklearn.exceptions import NotFittedError
from test_version_metadata import fresh_load

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor, __version__


@pytest.fixture(scope="module", params=["binary", "multiclass", "regression"])
def journey(request):
    task = request.param
    X, y = data(task)
    X["text"] = pd.Series(np.resize(["red", "blue", None], len(X)), dtype="string")
    X["plain_bool"] = np.arange(len(X)) % 2 == 0
    X["empty_numeric"] = np.nan
    X["empty_category"] = pd.Categorical([None] * len(X), categories=["unused"])
    # Categorical-heavy, including rare IDs; no large model or quality exercise.
    for i in range(6):
        X[f"category_{i}"] = pd.Categorical([f"k{j % (i + 3)}" for j in range(len(X))])
    X.index = (
        np.arange(len(X)) * 3 + 100
        if task == "binary"
        else [f"row_{i}" for i in range(len(X))]
        if task == "multiclass"
        else np.arange(len(X)) // 2
    )
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    model = cls(**kwargs()).fit(X, y)
    return task, cls, model, X, y


def test_frame_indices_roles_and_recovery(journey):
    task, cls, model, X, y = journey
    before = X.copy(deep=True)
    expected = model.predict(X)
    assert model.n_features_in_ == len(X.columns)
    assert model.feature_names_in_.tolist() == list(X.columns)
    assert model.device_ == "cpu" and model._fitted_neurotabular_version_ == __version__
    assert [entry["name"] for entry in model.input_schema_] == list(X.columns)
    assert (
        "text" in model.categorical_features_
        and "plain_bool" in model.categorical_features_
    )
    assert "empty_numeric" in model.numeric_features_
    for index in (
        np.arange(len(X)) * 4,
        [f"s{i}" for i in range(len(X))],
        np.zeros(len(X), dtype=int),
    ):
        query = X.copy()
        query.index = index
        np.testing.assert_array_equal(model.predict(query), expected)
    np.testing.assert_array_equal(model.predict(X.iloc[:, ::-1]), expected)
    # Reordered rows may change float32 GEMM rounding; output order/values stay
    # correct within precision. Exact parity uses identical ordered inputs.
    if task == "regression":
        np.testing.assert_allclose(
            model.predict(X.iloc[::-1, ::-1]), expected[::-1], rtol=1e-6, atol=1e-6
        )
    else:
        np.testing.assert_array_equal(model.predict(X.iloc[::-1, ::-1]), expected[::-1])
        np.testing.assert_allclose(
            model.predict_proba(X.iloc[::-1, ::-1]),
            model.predict_proba(X)[::-1],
            rtol=1e-6,
            atol=1e-6,
        )
    with pytest.raises(NotFittedError):
        cls(**kwargs()).predict(X)
    for query in (X.drop(columns="value"), X.assign(unexpected=1)):
        with pytest.raises(ValueError, match="same column names"):
            model.predict(query)
    query = X.copy()
    query["value"] = "not_numeric"
    with pytest.raises(ValueError, match="value.*refit"):
        model.predict(query)
    query = X.copy()
    query["group"] = query["group"].astype("string")
    query.iloc[0, query.columns.get_loc("group")] = "new_level"
    assert model.predict(query).shape == (len(X),)
    bad_targets = [np.column_stack([y, y]), np.full(len(X), np.nan)]
    for target in bad_targets:
        with pytest.raises((ValueError, TypeError)):
            model.fit(X, target)
        np.testing.assert_array_equal(model.predict(X), expected)
    for weights in (np.full(len(X), -1), np.full(len(X), np.nan), np.ones(len(X) - 1)):
        with pytest.raises(ValueError, match="sample_weight"):
            model.fit(X, y, sample_weight=weights)
        np.testing.assert_array_equal(model.predict(X), expected)
    with pytest.raises(ValueError, match="device"):
        cls(**{**kwargs(), "device": "gpu"}).fit(X, y)
    pd.testing.assert_frame_equal(X, before)
    if task != "regression":
        assert model.classes_.tolist() == sorted(set(y))


@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_mixed_fresh_process_and_size(journey, format, tmp_path):
    task, _, model, X, _ = journey
    path = tmp_path / f"{task}.{format}"
    if format == "pickle":
        path.write_bytes(pickle.dumps(model))
    else:
        joblib.dump(model, path)
    metadata, prediction, probability = fresh_load(path, format, X, tmp_path)
    np.testing.assert_array_equal(prediction, model.predict(X))
    assert metadata["schema"] == model.input_schema_
    assert metadata["version"] == __version__ and metadata["device"] == "cpu"
    if task != "regression":
        np.testing.assert_array_equal(probability, model.predict_proba(X))
    start = time.perf_counter_ns()
    loaded = (
        pickle.loads(path.read_bytes()) if format == "pickle" else joblib.load(path)
    )
    load_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    np.testing.assert_array_equal(loaded.predict(X), prediction)
    predict_ns = time.perf_counter_ns() - start
    # A test run must never overwrite a prior phase's sealed measurements.
    output = tmp_path / "measurements"
    if output.parent.exists():
        output.mkdir(exist_ok=True)
        (output / f"{task}_{format}.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "format": format,
                    "bytes": path.stat().st_size,
                    "load_ns": load_ns,
                    "predict_after_load_including_assert_ns": predict_ns,
                    "rows": len(X),
                    "features": len(X.columns),
                    "fresh_process_parity": True,
                    "scope": "One local measurement, not a memory or latency guarantee",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

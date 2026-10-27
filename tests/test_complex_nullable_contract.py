"""Complex numerical columns must fail before any lossy real conversion."""

import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor
from neurotabular.preprocessing import TabularPreprocessor


def companion(dtype):
    if dtype in {"boolean", "string", "category"}:
        values = [True, False, None] if dtype == "boolean" else ["a", "b", None]
    else:
        values = [1, 2, pd.NA]
    return pd.Series(values, dtype=dtype)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("zero_imaginary", [False, True])
@pytest.mark.parametrize(
    "other", ["Float64", "Int64", "UInt64", "Float32", "boolean", "string", "category"]
)
@pytest.mark.parametrize("strategy", ["standard", "robust"])
def test_complex_nullable_fit_and_transform_rejected(
    dtype, zero_imaginary, other, strategy
):
    imaginary = 0 if zero_imaginary else 2
    frame = pd.DataFrame(
        {
            "bad": np.asarray(
                [1 + imaginary * 1j, np.nan, 3 - imaginary * 1j], dtype=dtype
            ),
            "other": companion(other),
        }
    )
    before = frame.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"real-valued; complex.*\['bad'\]"):
            TabularPreprocessor(numeric_strategy=strategy).fit(frame)
        valid = frame.copy()
        valid["bad"] = [1.0, np.nan, 3.0]
        prep = TabularPreprocessor(numeric_strategy=strategy).fit(valid)
        with pytest.raises(ValueError, match=r"real-valued; complex.*\['bad'\]"):
            prep.transform(frame.loc[:, ["other", "bad"]])
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("dtype", ["Float64", "Float32", "Int64", "UInt64"])
@pytest.mark.parametrize("strategy", ["standard", "robust"])
def test_valid_nullable_mixed_data_is_exact_and_owned(dtype, missing, strategy):
    X = pd.DataFrame(
        {
            "x": pd.Series([1, missing, 3, 4], dtype=dtype),
            "z": [2.0, np.nan, 5.0, 7.0],
            "cat": pd.Series(["a", None, "b", "a"], dtype="string"),
            "flag": pd.Series([True, None, False, True], dtype="boolean"),
        }
    )
    before = X.copy(deep=True)
    reference = X.copy()
    reference["x"] = X["x"].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    nullable = TabularPreprocessor(numeric_strategy=strategy).fit(X)
    real = TabularPreprocessor(numeric_strategy=strategy).fit(reference)
    a, b = nullable.transform(X), real.transform(reference)
    np.testing.assert_array_equal(a.numerical, b.numerical)
    np.testing.assert_array_equal(a.categorical, b.categorical)
    assert a.numerical.dtype == np.float32 and a.categorical.dtype == np.int64
    assert not np.shares_memory(a.numerical, X["z"].to_numpy())
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean"])
def test_object_complex_on_a_fitted_numerical_feature_is_rejected(dtype):
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "other": companion(dtype)})
    prep = TabularPreprocessor().fit(X)
    query = X.copy()
    query["x"] = pd.Series([1 + 2j, 2 + 0j, 3 - 1j], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued; complex"):
            prep.transform(query)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_estimator_rejection_preserves_valid_predictor_rng_and_persistence(
    task, tmp_path
):
    rng = np.random.default_rng(240031)
    n = 60
    values = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "x": values,
            "nullable": pd.Series(np.arange(n) % 5, dtype="Int64"),
            "cat": pd.Categorical(np.resize(["a", "b", None], n)),
        }
    )
    X.loc[::9, "nullable"] = pd.NA
    y = (
        values * 2
        if task == "regression"
        else (np.arange(n) % (2 if task == "binary" else 3))
    )
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    model = cls(
        max_epochs=1, hidden_dim=8, n_blocks=1, device="cpu", random_state=240031
    )
    model.fit(X, y)
    expected = model.predict(X)
    probability = None if task == "regression" else model.predict_proba(X)
    invalid = X.copy()
    invalid["x"] = values.astype(np.complex128) + 2j
    state = np.random.get_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in (
            [model.predict, model.predict_proba]
            if task != "regression"
            else [model.predict]
        ):
            with pytest.raises(ValueError, match="real-valued; complex"):
                method(invalid)
        with pytest.raises(ValueError, match="real-valued; complex"):
            model.fit(invalid, y)
    current = np.random.get_state()
    assert state[0] == current[0] and state[2:] == current[2:]
    np.testing.assert_array_equal(state[1], current[1])
    np.testing.assert_array_equal(model.predict(X), expected)
    restored = pickle.loads(pickle.dumps(model))
    path = tmp_path / (task + ".joblib")
    joblib.dump(model, path)
    for clone in [restored, joblib.load(path)]:
        np.testing.assert_array_equal(clone.predict(X), expected)
        if probability is not None:
            np.testing.assert_array_equal(clone.predict_proba(X), probability)


@pytest.mark.parametrize("bad", ["not-a-number", np.inf, -np.inf])
def test_other_invalid_numerical_inputs_still_fail(bad):
    prep = TabularPreprocessor().fit(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "n": companion("Int64")})
    )
    query = pd.DataFrame({"x": [bad, 2.0, 3.0], "n": companion("Int64")})
    with pytest.raises(ValueError):
        prep.transform(query)

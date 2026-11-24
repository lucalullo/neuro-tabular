"""Numerical range and real-weight errors must be controlled and state-safe."""

import numpy as np
import pandas as pd
import pytest

from neurotabular._base import _BaseNeuroTabularEstimator
from neurotabular.preprocessing import TabularPreprocessor


@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize(
    "values", [[1e308, 1e308], [np.finfo(float).max, -np.finfo(float).max]]
)
def test_robust_training_statistics_overflow_is_controlled(policy, values):
    before = np.geterr().copy()
    with np.errstate(over=policy, invalid=policy):
        expected = np.geterr().copy()
        with pytest.raises(ValueError, match="safe range for training statistics"):
            TabularPreprocessor(numeric_strategy="robust").fit(
                pd.DataFrame({"x": values})
            )
        assert np.geterr() == expected
    assert np.geterr() == before


@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1e308, 1e308, np.nan], [0.0, 0.0, 0.0]),
        (
            [np.finfo(float).max, -np.finfo(float).max, np.nan],
            [np.sqrt(1.5), -np.sqrt(1.5), 0.0],
        ),
        (
            [1e308, np.nextafter(1e308, np.inf), np.nan],
            [-np.sqrt(1.5), np.sqrt(1.5), 0.0],
        ),
    ],
)
def test_standard_training_statistics_handle_finite_extremes(policy, values, expected):
    frame = pd.DataFrame({"x": values})
    before = np.geterr().copy()
    with np.errstate(over=policy, invalid=policy):
        inside = np.geterr().copy()
        prep = TabularPreprocessor().fit(frame)
        transformed = prep.transform(frame).numerical
        np.testing.assert_array_equal(
            transformed[:, 0], np.asarray(expected, np.float32)
        )
        np.testing.assert_array_equal(transformed[:, 1], [0.0, 0.0, 1.0])
        assert np.geterr() == inside
    assert np.geterr() == before


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("imaginary", [0, 2])
@pytest.mark.parametrize("container", ["array", "series", "list"])
def test_complex_sample_weights_are_rejected(dtype, imaginary, container):
    weights = np.array([1 + imaginary * 1j, 2 + 0j], dtype=dtype)
    if container == "series":
        weights = pd.Series(weights)
    elif container == "list":
        weights = weights.tolist()
    with pytest.raises(ValueError, match="sample_weight.*numeric"):
        _BaseNeuroTabularEstimator._validate_sample_weight(weights, 2)


def test_statistics_context_is_reusable_and_restores_caller_policy():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "missing": [np.nan] * 3})
    before = np.geterr().copy()
    for _ in range(3):
        prep = TabularPreprocessor().fit(X)
        assert np.isfinite(prep.transform(X).numerical).all()
        assert np.geterr() == before

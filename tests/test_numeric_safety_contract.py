"""Numerical range and real-weight errors must be controlled and state-safe."""

import numpy as np
import pandas as pd
import pytest

from neurotabular._base import _BaseNeuroTabularEstimator
from neurotabular.preprocessing import TabularPreprocessor


@pytest.mark.parametrize("strategy", ["standard", "robust"])
@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize(
    "values", [[1e308, 1e308], [np.finfo(float).max, -np.finfo(float).max]]
)
def test_training_statistics_overflow_is_controlled(strategy, policy, values):
    before = np.geterr().copy()
    with np.errstate(over=policy, invalid=policy):
        expected = np.geterr().copy()
        with pytest.raises(ValueError, match="safe range for training statistics"):
            TabularPreprocessor(numeric_strategy=strategy).fit(
                pd.DataFrame({"x": values})
            )
        assert np.geterr() == expected
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

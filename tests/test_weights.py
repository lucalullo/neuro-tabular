import numpy as np
import pandas as pd
import pytest

from neurotabular import NeuroTabularClassifier


def _weighted_frame():
    X = pd.DataFrame({"x": np.linspace(-3.0, 3.0, 60)})
    y = np.array([0] * 45 + [1] * 15)
    return X, y


def test_balanced_class_weight_uses_training_target(fast_model_kwargs):
    X, y = _weighted_frame()
    X_valid = pd.DataFrame({"x": [-1.0, 1.0, -2.0, 2.0]})
    y_valid = np.array([0, 1, 0, 1])
    model = NeuroTabularClassifier(
        **{**fast_model_kwargs, "class_weight": "balanced"}
    ).fit(X, y, eval_set=(X_valid, y_valid))
    assert model.class_weight_[0] == pytest.approx(60 / 90)
    assert model.class_weight_[1] == pytest.approx(2.0)


def test_sample_and_class_weights_are_combinable(fast_model_kwargs):
    X, y = _weighted_frame()
    weights = np.linspace(0.25, 2.0, len(X))
    model = NeuroTabularClassifier(
        **{**fast_model_kwargs, "class_weight": "balanced"}
    ).fit(X, y, sample_weight=weights)
    assert np.isfinite(model.best_score_)
    assert np.isfinite(model.predict_proba(X)).all()


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0], "different numbers"),
        (np.full(60, -1.0), "non-negative"),
        (np.full(60, np.inf), "finite"),
        (np.zeros(60), "at least one positive"),
        (np.full((60, 1), 1.0), "one-dimensional"),
        (["x"] * 60, "numeric"),
    ],
)
def test_invalid_sample_weight_is_rejected(weights, message, fast_model_kwargs):
    X, y = _weighted_frame()
    with pytest.raises(ValueError, match=message):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, y, sample_weight=weights)


def test_zero_weight_rows_are_ignored_entirely(fast_model_kwargs):
    X, y = _weighted_frame()
    X = X.assign(category=np.resize(["a", "b"], len(X)))
    X.loc[[0, 59], "x"] = [-1e12, 1e12]
    X.loc[[0, 59], "category"] = ["zero-only-a", "zero-only-b"]
    weights = np.ones(len(X))
    weights[[0, 59]] = 0.0
    keep = weights > 0.0
    weighted_target = y.astype(object)
    weighted_target[0] = None
    weighted_target[59] = "ignored-third-class"
    X_valid = pd.DataFrame({"x": [-1.0, 1.0], "category": ["a", "b"]})
    y_valid = np.array([0, 1])

    weighted = NeuroTabularClassifier(**fast_model_kwargs).fit(
        X, weighted_target, sample_weight=weights, eval_set=(X_valid, y_valid)
    )
    filtered = NeuroTabularClassifier(**fast_model_kwargs).fit(
        X.loc[keep], weighted_target[keep], eval_set=(X_valid, y_valid)
    )

    assert weighted.n_ignored_samples_ == 2
    assert weighted.n_effective_samples_ == int(keep.sum())
    assert "zero-only-a" not in weighted._preprocessor_.category_vocabs_["category"]
    assert np.array_equal(
        weighted._preprocessor_.numeric_medians_,
        filtered._preprocessor_.numeric_medians_,
    )
    assert np.allclose(
        weighted.predict_proba(X_valid),
        filtered.predict_proba(X_valid),
        rtol=0.0,
        atol=1e-7,
    )


def test_zero_weight_rows_cannot_supply_the_second_class(fast_model_kwargs):
    X, y = _weighted_frame()
    weights = np.ones(len(X))
    weights[y == 1] = 0.0
    with pytest.raises(ValueError, match="at least two target classes"):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, y, sample_weight=weights)


def test_weight_dynamic_range_must_be_float32_safe(fast_model_kwargs):
    X, y = _weighted_frame()
    weights = np.ones(len(X))
    weights[0] = np.finfo(np.float64).tiny
    weights[1] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="dynamic range|float32"):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, y, sample_weight=weights)

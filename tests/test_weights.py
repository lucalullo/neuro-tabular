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


def test_zero_weight_training_subset_is_rejected(monkeypatch, fast_model_kwargs):
    X, y = _weighted_frame()
    weights = np.zeros(len(X))
    weights[-1] = 1.0

    def fixed_split(indices, **kwargs):
        return indices[:-12], indices[-12:]

    monkeypatch.setattr("neurotabular.classifier.train_test_split", fixed_split)
    with pytest.raises(ValueError, match="positive combined weight"):
        NeuroTabularClassifier(**fast_model_kwargs).fit(X, y, sample_weight=weights)


def test_roc_auc_requires_positive_weight_for_both_validation_classes(
    monkeypatch, fast_model_kwargs
):
    X, y = _weighted_frame()
    validation = np.array([0, 1, 45, 46])
    training = np.setdiff1d(np.arange(len(X)), validation)
    weights = np.ones(len(X))
    weights[[45, 46]] = 0.0

    def fixed_split(indices, **kwargs):
        return training, validation

    monkeypatch.setattr("neurotabular.classifier.train_test_split", fixed_split)
    parameters = {**fast_model_kwargs, "eval_metric": "roc_auc"}
    with pytest.raises(ValueError, match="positive validation weight"):
        NeuroTabularClassifier(**parameters).fit(X, y, sample_weight=weights)

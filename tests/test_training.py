import math

import numpy as np
import pandas as pd
import pytest
import torch

from neurotabular import NeuroTabularClassifier


def test_min_delta_and_eval_frequency_stop_on_validation_checks():
    X = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 80)})
    y = np.resize([0, 1], 80)
    model = NeuroTabularClassifier(
        hidden_dim=8,
        n_blocks=1,
        max_epochs=20,
        patience=2,
        min_delta=10.0,
        eval_frequency=2,
        device="cpu",
        random_state=3,
    ).fit(X, y)
    assert model.best_epoch_ == 1
    assert model.n_iter_ == 4
    assert [row["epoch"] for row in model.history_] == [1, 2, 4]


def test_auto_and_explicit_batch_size(mixed_binary_data, fast_model_kwargs):
    X, y = mixed_binary_data
    automatic = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    explicit = NeuroTabularClassifier(**{**fast_model_kwargs, "batch_size": 17}).fit(
        X, y
    )
    assert automatic.batch_size_ == len(X) - math.ceil(0.2 * len(X))
    assert explicit.batch_size_ == 17


def test_batched_inference_uses_multiple_calls(monkeypatch, fast_model_kwargs):
    n = 2_500
    X = pd.DataFrame({"x": np.linspace(-2.0, 2.0, n)})
    y = np.resize([0, 1], n)
    model = NeuroTabularClassifier(**{**fast_model_kwargs, "batch_size": 128}).fit(X, y)
    sizes = []
    original = model._model_.forward

    def recording_forward(numerical, categorical):
        sizes.append(len(numerical))
        return original(numerical, categorical)

    monkeypatch.setattr(model._model_, "forward", recording_forward)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (n, 2)
    assert max(sizes) <= model.inference_batch_size_
    assert len(sizes) >= 3


@pytest.mark.parametrize(
    "parameters",
    [
        {"hidden_dim": 0},
        {"n_blocks": 0},
        {"dropout": 1.0},
        {"lr": 0.0},
        {"weight_decay": -1.0},
        {"batch_size": 0},
        {"batch_size": "large"},
        {"max_epochs": 0},
        {"validation_fraction": 0.0},
        {"patience": 0},
        {"min_delta": -1.0},
        {"eval_frequency": 0},
        {"eval_metric": "f1"},
        {"class_weight": "auto"},
        {"class_weight": {0: 1.0, 1: 2.0}},
        {"min_category_count": 0},
        {"device": "tpu"},
        {"random_state": 1.5},
        {"verbose": 2},
    ],
)
def test_invalid_hyperparameters_are_rejected(parameters, fast_model_kwargs):
    X = pd.DataFrame({"x": np.arange(20)})
    y = np.resize([0, 1], 20)
    with pytest.raises((TypeError, ValueError)):
        NeuroTabularClassifier(**{**fast_model_kwargs, **parameters}).fit(X, y)


def test_verbose_zero_is_silent_and_one_reports(
    capsys, mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    assert capsys.readouterr().out == ""
    NeuroTabularClassifier(**{**fast_model_kwargs, "verbose": 1}).fit(X, y)
    output = capsys.readouterr().out
    assert "Epoch 1/1" in output
    assert "Best epoch" in output


def test_cpu_reproducibility(mixed_binary_data, fast_model_kwargs):
    X, y = mixed_binary_data
    first = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    second = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    assert np.allclose(
        first.predict_proba(X), second.predict_proba(X), rtol=0.0, atol=1e-7
    )


def test_cpu_training_never_constructs_autocast(
    monkeypatch, mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data

    def fail_autocast(*args, **kwargs):
        raise AssertionError("CPU/FP32 training must not construct torch.autocast")

    monkeypatch.setattr(torch, "autocast", fail_autocast)
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    assert model.device_ == "cpu"
    assert model.profile_["training"]["amp_enabled"] is False

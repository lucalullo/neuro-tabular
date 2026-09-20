import math
import random

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
    assert model.n_iter_ == 4
    assert [row["epoch"] for row in model.history_] == [1, 2, 4]
    assert model.best_score_ == min(row["validation_score"] for row in model.history_)
    assert model.best_epoch_ in {1, 2, 4}


def test_min_delta_controls_patience_but_not_best_checkpoint(monkeypatch):
    validation_losses = iter([1.0, 0.95, 0.94])

    def controlled_validation(*args, **kwargs):
        loss = next(validation_losses)
        return loss, loss

    monkeypatch.setattr("neurotabular.training._validate", controlled_validation)
    X = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 40)})
    y = np.resize([0, 1], len(X))
    model = NeuroTabularClassifier(
        hidden_dim=8,
        n_blocks=1,
        max_epochs=3,
        patience=2,
        min_delta=0.1,
        device="cpu",
        random_state=3,
    ).fit(X, y)
    assert model.n_iter_ == 3
    assert model.best_epoch_ == 3
    assert model.best_score_ == pytest.approx(0.94)


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
        {"numerical_embedding": "attention"},
        {"use_category_frequency": 1},
        {"feature_gating": 1},
        {"full_data_refit": 1},
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


def test_fit_restores_caller_random_states(mixed_binary_data, fast_model_kwargs):
    X, y = mixed_binary_data
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)

    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_failed_refit_preserves_previous_model(
    monkeypatch, mixed_binary_data, fast_model_kwargs
):
    X, y = mixed_binary_data
    model = NeuroTabularClassifier(**fast_model_kwargs).fit(X, y)
    previous_model = model._model_
    previous_probabilities = model.predict_proba(X)

    def fail_training(*args, **kwargs):
        raise RuntimeError("controlled training failure")

    monkeypatch.setattr("neurotabular._base.train_model", fail_training)
    with pytest.raises(RuntimeError, match="controlled training failure"):
        model.fit(X, y)

    assert model._model_ is previous_model
    assert np.array_equal(model.predict_proba(X), previous_probabilities)


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

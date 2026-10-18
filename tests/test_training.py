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


class _InferenceOutputModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, numerical, categorical):
        return self.logits[numerical[:, 0].long()]


@pytest.mark.parametrize("task_name", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("n_rows", [1, 32, 512, 1024, 1025, 2057])
@pytest.mark.parametrize("batch_size", [1, 17, 1024])
def test_inference_output_materialization_matches_batch_reference(
    task_name, n_rows, batch_size
):
    from neurotabular.preprocessing import ProcessedTable
    from neurotabular.tasks import TaskSpec
    from neurotabular.training import predict_outputs

    task = TaskSpec(task_name, 7 if task_name == "multiclass" else 1)
    values = torch.linspace(-20, 20, n_rows * task.output_dim)
    logits = values.reshape(n_rows, task.output_dim)
    if task_name != "multiclass":
        logits = logits[:, 0]
    model = _InferenceOutputModel(logits)
    numerical = np.arange(n_rows, dtype=np.float32)[:, None]
    categorical = np.empty((n_rows, 0), dtype=np.int64)
    data = ProcessedTable(numerical, categorical)
    original = numerical.copy()
    with torch.inference_mode():
        pieces = [
            task.transform(logits[start : start + batch_size]).cpu().numpy()
            for start in range(0, n_rows, batch_size)
        ]
    expected = np.concatenate(pieces).astype(np.float64, copy=False)
    if task_name != "regression":
        expected = np.clip(expected, 0.0, 1.0)

    result = predict_outputs(
        model, data, device=torch.device("cpu"), batch_size=batch_size, task=task
    )
    assert result.dtype == np.float64
    assert result.shape == expected.shape
    assert result.tobytes() == expected.tobytes()
    assert result.flags.writeable
    assert np.array_equal(numerical, original)
    assert not model.training
    result[...] = -123
    repeated = predict_outputs(
        model, data, device=torch.device("cpu"), batch_size=batch_size, task=task
    )
    assert repeated.tobytes() == expected.tobytes()
    assert not np.shares_memory(result, repeated)


@pytest.mark.parametrize("task_name", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_inference_rejects_nonfinite_transformed_outputs(task_name, bad_value):
    from neurotabular.preprocessing import ProcessedTable
    from neurotabular.tasks import TaskSpec
    from neurotabular.training import predict_outputs

    task = TaskSpec(task_name, 3 if task_name == "multiclass" else 1)
    logits = torch.full((5, task.output_dim), bad_value)
    if task_name != "multiclass":
        logits = logits[:, 0]
    data = ProcessedTable(
        np.arange(5, dtype=np.float32)[:, None], np.empty((5, 0), dtype=np.int64)
    )
    # Infinite logits are valid when sigmoid produces finite probabilities.
    if task_name == "binary" and math.isinf(bad_value):
        result = predict_outputs(
            _InferenceOutputModel(logits),
            data,
            device=torch.device("cpu"),
            batch_size=2,
            task=task,
        )
        assert np.array_equal(result, np.ones(5, dtype=np.float64))
        return
    with pytest.raises(RuntimeError, match="non-finite probabilities"):
        predict_outputs(
            _InferenceOutputModel(logits),
            data,
            device=torch.device("cpu"),
            batch_size=2,
            task=task,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 3])
def test_inference_empty_or_invalid_batch_retains_error(batch_size):
    from neurotabular.preprocessing import ProcessedTable
    from neurotabular.training import predict_outputs

    data = ProcessedTable(
        np.empty((0, 1), dtype=np.float32), np.empty((0, 0), dtype=np.int64)
    )
    with pytest.raises(ValueError):
        predict_outputs(
            _InferenceOutputModel(torch.empty(0)),
            data,
            device=torch.device("cpu"),
            batch_size=batch_size,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inference_noncontiguous_multiclass_and_signed_zero(dtype):
    from neurotabular.preprocessing import ProcessedTable
    from neurotabular.tasks import TaskSpec
    from neurotabular.training import predict_outputs

    source = torch.arange(42, dtype=dtype).reshape(7, 6)[:, ::2]
    task = TaskSpec("multiclass", 3)
    data = ProcessedTable(
        np.arange(7, dtype=np.float32)[:, None], np.empty((7, 0), dtype=np.int64)
    )

    class StridedInferenceModel(_InferenceOutputModel):
        def forward(self, numerical, categorical):
            values = super().forward(numerical, categorical)
            return torch.stack((values, values), dim=-1)[..., 0]

    result = predict_outputs(
        StridedInferenceModel(source),
        data,
        device=torch.device("cpu"),
        batch_size=3,
        task=task,
    )
    with torch.inference_mode():
        expected = np.concatenate(
            [task.transform(source[i : i + 3]).numpy() for i in range(0, 7, 3)]
        ).astype(np.float64, copy=False)
    assert result.tobytes() == expected.tobytes()
    zeros = torch.tensor([-0.0, 0.0, -1.0, 1.0, -0.0, 0.0, 2.0], dtype=dtype)
    result = predict_outputs(
        _InferenceOutputModel(zeros),
        data,
        device=torch.device("cpu"),
        batch_size=3,
        task=TaskSpec("regression"),
    )
    assert result.tobytes() == zeros.numpy().astype(np.float64).tobytes()

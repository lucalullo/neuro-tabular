"""Low-overhead in-memory training for binary tabular networks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from .network import TabularNetwork
from .preprocessing import ProcessedTable

MetricName = Literal["loss", "roc_auc", "accuracy"]


@dataclass
class TrainingResult:
    """Training outputs returned to the public estimator."""

    best_epoch: int
    best_score: float
    best_validation_loss: float
    n_iter: int
    history: list[dict[str, float | int]]
    profile: dict[str, float | bool | str]


@dataclass
class _TensorTable:
    numerical: torch.Tensor
    categorical: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor

    @property
    def n_samples(self) -> int:
        return self.target.shape[0]

    @property
    def device(self) -> torch.device:
        return self.target.device

    def index(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if indices.device != self.device:
            indices = indices.to(self.device)
        return (
            self.numerical.index_select(0, indices),
            self.categorical.index_select(0, indices),
            self.target.index_select(0, indices),
            self.weight.index_select(0, indices),
        )

    def all(self) -> tuple[torch.Tensor, ...]:
        return self.numerical, self.categorical, self.target, self.weight


def resolve_batch_size(
    requested: int | str,
    *,
    n_samples: int,
    n_numeric_inputs: int,
    categorical_cardinalities: list[int],
    hidden_dim: int,
    n_blocks: int,
    device: torch.device,
) -> int:
    """Resolve a deterministic and conservative training batch size."""

    if requested != "auto":
        return min(int(requested), n_samples)
    embedding_width = sum(
        min(16, max(2, math.ceil(2.0 * cardinality**0.25)))
        for cardinality in categorical_cardinalities
    )
    effective_width = n_numeric_inputs + embedding_width + hidden_dim * (n_blocks + 2)
    if n_samples <= 2_048:
        return n_samples
    if device.type == "cpu":
        if n_samples <= 16_384:
            return min(n_samples, 2_048 if effective_width > 512 else 4_096)
        return min(n_samples, 4_096 if effective_width > 512 else 8_192)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except (RuntimeError, TypeError):
        free_bytes = 2 * 2**30
    estimated_row_bytes = max(1, effective_width) * 4 * 8
    memory_limited = max(512, int(0.1 * free_bytes / estimated_row_bytes))
    power_of_two = 1 << max(9, int(math.log2(memory_limited)))
    return min(n_samples, 65_536, power_of_two)


def train_binary_model(
    model: TabularNetwork,
    train_data: ProcessedTable,
    train_target: np.ndarray,
    train_weight: np.ndarray,
    validation_data: ProcessedTable,
    validation_target: np.ndarray,
    validation_weight: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    eval_frequency: int,
    eval_metric: MetricName,
    lr: float,
    weight_decay: float,
    random_state: int,
    verbose: int,
    lr_strategy: str = "cosine",
    use_amp: bool | None = None,
) -> TrainingResult:
    """Train ``model`` with direct tensor indexing and restore best weights."""

    engine_started = perf_counter()
    profile: dict[str, float | bool | str] = {
        "tensor_conversion_seconds": 0.0,
        "device_transfer_seconds": 0.0,
        "batch_creation_seconds": 0.0,
        "forward_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_step_seconds": 0.0,
        "validation_inference_seconds": 0.0,
        "metric_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "scheduler_seconds": 0.0,
        "best_state_restoration_seconds": 0.0,
        "timing_reliable": device.type == "cpu",
        "data_residency": "cpu",
    }
    conversion_started = perf_counter()
    train_tensors = _to_tensors(train_data, train_target, train_weight)
    validation_tensors = _to_tensors(
        validation_data, validation_target, validation_weight
    )
    profile["tensor_conversion_seconds"] = perf_counter() - conversion_started

    transfer_started = perf_counter()
    train_tensors, train_resident = _prepare_device_table(train_tensors, device)
    validation_tensors, validation_resident = _prepare_device_table(
        validation_tensors, device
    )
    profile["data_residency"] = (
        str(device) if train_resident and validation_resident else "cpu-staged"
    )
    profile["device_transfer_seconds"] = perf_counter() - transfer_started

    model.to(device)
    optimizer = _make_optimizer(model, lr, weight_decay, device)
    scheduler = _make_scheduler(optimizer, max_epochs, lr_strategy)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = device.type == "cuda" if use_amp is None else use_amp
    scaler = _make_grad_scaler(use_amp)
    profile["amp_enabled"] = use_amp
    profile["optimizer"] = "AdamW (PyTorch-selected implementation)"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_state)
    profile["engine_setup_seconds"] = perf_counter() - engine_started

    best_score = math.inf if eval_metric == "loss" else -math.inf
    best_validation_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    checks_without_improvement = 0
    history: list[dict[str, float | int]] = []
    training_compute_seconds = 0.0
    validation_seconds = 0.0

    for epoch in range(1, max_epochs + 1):
        epoch_started = perf_counter()
        model.train()
        loss_numerator = torch.zeros((), device=device)
        weight_denominator = torch.zeros((), device=device)
        for numerical, categorical, target, weight in _training_batches(
            train_tensors,
            batch_size,
            generator,
            device,
            profile,
        ):
            optimizer.zero_grad(set_to_none=True)
            forward_started = perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(
                    numerical.to(device, non_blocking=True),
                    categorical.to(device, non_blocking=True),
                )
                target = target.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)
                elementwise_loss = criterion(logits, target)
                batch_weight = weight.sum()
                loss = (elementwise_loss * weight).sum() / batch_weight.clamp_min(1e-12)
            profile["forward_seconds"] = float(profile["forward_seconds"]) + (
                perf_counter() - forward_started
            )
            backward_started = perf_counter()
            scaler.scale(loss).backward()
            profile["backward_seconds"] = float(profile["backward_seconds"]) + (
                perf_counter() - backward_started
            )
            step_started = perf_counter()
            scaler.step(optimizer)
            scaler.update()
            profile["optimizer_step_seconds"] = float(
                profile["optimizer_step_seconds"]
            ) + (perf_counter() - step_started)
            loss_numerator += (elementwise_loss.detach() * weight).sum()
            weight_denominator += batch_weight.detach()
        scheduler_started = perf_counter()
        scheduler.step()
        profile["scheduler_seconds"] = float(profile["scheduler_seconds"]) + (
            perf_counter() - scheduler_started
        )
        epoch_loss = loss_numerator / weight_denominator.clamp_min(1e-12)
        train_loss = float(epoch_loss.detach().cpu())
        if not math.isfinite(train_loss):
            raise RuntimeError("Training produced a non-finite loss.")
        training_compute_seconds += perf_counter() - epoch_started

        should_evaluate = (
            epoch == 1 or epoch % eval_frequency == 0 or epoch == max_epochs
        )
        if not should_evaluate:
            continue
        validation_started = perf_counter()
        validation_loss, score = _validate(
            model,
            validation_tensors,
            batch_size=max(batch_size, 4_096),
            criterion=criterion,
            metric=eval_metric,
            device=device,
            profile=profile,
        )
        validation_seconds += perf_counter() - validation_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_score": score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if verbose:
            print(
                f"Epoch {epoch}/{max_epochs} - train_loss={train_loss:.6f} - "
                f"val_loss={validation_loss:.6f} - val_{eval_metric}={score:.6f}"
            )
        if _is_improvement(score, best_score, eval_metric, min_delta):
            checkpoint_started = perf_counter()
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            profile["checkpoint_seconds"] = float(profile["checkpoint_seconds"]) + (
                perf_counter() - checkpoint_started
            )
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
            if checks_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation score.")
    restoration_started = perf_counter()
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    profile["best_state_restoration_seconds"] = perf_counter() - restoration_started
    profile["training_compute_seconds"] = training_compute_seconds
    profile["validation_seconds"] = validation_seconds
    profile["engine_total_seconds"] = perf_counter() - engine_started
    if verbose:
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation {eval_metric}: {best_score:.6f}")
    return TrainingResult(
        best_epoch=best_epoch,
        best_score=float(best_score),
        best_validation_loss=float(best_validation_loss),
        n_iter=epoch,
        history=history,
        profile=profile,
    )


def predict_probabilities(
    model: TabularNetwork,
    data: ProcessedTable,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run bounded-memory inference and return positive-class probabilities."""

    numerical = torch.from_numpy(data.numerical)
    categorical = torch.from_numpy(data.categorical)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, data.n_samples, batch_size):
            stop = min(start + batch_size, data.n_samples)
            logits = model(
                numerical[start:stop].to(device, non_blocking=True),
                categorical[start:stop].to(device, non_blocking=True),
            )
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(outputs).astype(np.float64, copy=False)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("The fitted model produced non-finite probabilities.")
    return np.clip(probabilities, 0.0, 1.0)


def _to_tensors(
    data: ProcessedTable,
    target: np.ndarray,
    weight: np.ndarray,
) -> _TensorTable:
    return _TensorTable(
        numerical=torch.from_numpy(data.numerical),
        categorical=torch.from_numpy(data.categorical),
        target=torch.from_numpy(np.array(target, dtype=np.float32, copy=True)),
        weight=torch.from_numpy(np.array(weight, dtype=np.float32, copy=True)),
    )


def _prepare_device_table(
    table: _TensorTable, device: torch.device
) -> tuple[_TensorTable, bool]:
    if device.type == "cpu":
        return table, True
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in table.all())
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except (RuntimeError, TypeError):
        free_bytes = 0
    if total_bytes < 0.2 * free_bytes:
        return (
            _TensorTable(*(tensor.to(device) for tensor in table.all())),
            True,
        )
    return _TensorTable(*(tensor.pin_memory() for tensor in table.all())), False


def _training_batches(
    table: _TensorTable,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    profile: dict[str, float | bool | str],
):
    batch_started = perf_counter()
    if batch_size >= table.n_samples:
        profile["batch_creation_seconds"] = float(profile["batch_creation_seconds"]) + (
            perf_counter() - batch_started
        )
        yield table.all()
        return
    order = torch.randperm(table.n_samples, generator=generator)
    for start in range(0, table.n_samples, batch_size):
        indices = order[start : start + batch_size]
        batch = table.index(indices)
        profile["batch_creation_seconds"] = float(profile["batch_creation_seconds"]) + (
            perf_counter() - batch_started
        )
        yield batch
        batch_started = perf_counter()


def _validate(
    model: TabularNetwork,
    table: _TensorTable,
    *,
    batch_size: int,
    criterion: nn.Module,
    metric: MetricName,
    device: torch.device,
    profile: dict[str, float | bool | str],
) -> tuple[float, float]:
    model.eval()
    loss_numerator = torch.zeros((), device=device)
    weight_denominator = torch.zeros((), device=device)
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    inference_started = perf_counter()
    with torch.inference_mode():
        for start in range(0, table.n_samples, batch_size):
            stop = min(start + batch_size, table.n_samples)
            indices = torch.arange(start, stop, device=table.device)
            numerical, categorical, target, weight = table.index(indices)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            logits = model(
                numerical.to(device, non_blocking=True),
                categorical.to(device, non_blocking=True),
            )
            losses = criterion(logits, target)
            loss_numerator += (losses * weight).sum()
            weight_denominator += weight.sum()
            probabilities.append(torch.sigmoid(logits))
            targets.append(target)
            weights.append(weight)
    validation_loss = float(
        (loss_numerator / weight_denominator.clamp_min(1e-12)).cpu()
    )
    probability_array = torch.cat(probabilities).cpu().numpy()
    target_array = torch.cat(targets).cpu().numpy()
    weight_array = torch.cat(weights).cpu().numpy()
    profile["validation_inference_seconds"] = float(
        profile["validation_inference_seconds"]
    ) + (perf_counter() - inference_started)
    metric_started = perf_counter()
    if metric == "loss":
        score = validation_loss
    elif metric == "roc_auc":
        score = float(
            roc_auc_score(target_array, probability_array, sample_weight=weight_array)
        )
    else:
        score = float(
            accuracy_score(
                target_array,
                probability_array >= 0.5,
                sample_weight=weight_array,
            )
        )
    profile["metric_seconds"] = float(profile["metric_seconds"]) + (
        perf_counter() - metric_started
    )
    if not math.isfinite(validation_loss) or not math.isfinite(score):
        raise RuntimeError("Validation produced a non-finite result.")
    return validation_loss, score


def _make_optimizer(
    model: nn.Module, lr: float, weight_decay: float, device: torch.device
) -> torch.optim.Optimizer:
    kwargs: dict[str, object] = {"lr": lr, "weight_decay": weight_decay}
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _make_scheduler(
    optimizer: torch.optim.Optimizer, max_epochs: int, strategy: str
) -> torch.optim.lr_scheduler.LRScheduler:
    if strategy == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if strategy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=0.05 * optimizer.param_groups[0]["lr"]
        )
    if strategy == "warmup_cosine":
        warmup = max(1, min(3, max_epochs // 5))

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, max_epochs - warmup)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ValueError("Unknown learning-rate strategy.")


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _is_improvement(
    score: float,
    best_score: float,
    metric: MetricName,
    min_delta: float,
) -> bool:
    if not math.isfinite(score):
        return False
    if metric == "loss":
        return score < best_score - min_delta
    return score > best_score + min_delta

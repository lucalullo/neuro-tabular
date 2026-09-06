"""Focused ablations for NeuroTabular architecture and training defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from neurotabular.network import TabularNetwork
from neurotabular.preprocessing import TabularPreprocessor
from neurotabular.training import (
    predict_probabilities,
    resolve_batch_size,
    train_binary_model,
)


def make_data(name: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 1_200
    values = rng.normal(size=(n, 8))
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(8)])
    signal = values[:, 0] + 0.8 * values[:, 1] - 0.5 * values[:, 2]
    if name == "mixed":
        X["city"] = rng.choice(["a", "b", "c", "d", None], n)
        X["account"] = rng.choice([f"id_{i}" for i in range(80)], n)
        X.loc[::13, "x0"] = np.nan
        signal = np.nan_to_num(X["x0"].to_numpy(), nan=0.0)
        signal += 0.8 * values[:, 1] + 0.8 * (X["city"] == "a")
    signal += rng.normal(0.0, 0.9, n)
    return X, (signal > np.median(signal)).astype(np.float32)


def run_variant(
    variant: dict[str, object], dataset: str, seed: int
) -> dict[str, object]:
    X, y = make_data(dataset, seed)
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_fit, y_fit, test_size=0.2, stratify=y_fit, random_state=seed
    )
    started = perf_counter()
    preprocessor = TabularPreprocessor(
        min_category_count=int(variant["min_category_count"]),
        numeric_strategy=str(variant["numeric_strategy"]),
    )
    train_data = preprocessor.fit_transform(X_train)
    valid_data = preprocessor.transform(X_valid)
    test_data = preprocessor.transform(X_test)
    torch.manual_seed(seed)
    model = TabularNetwork(
        preprocessor.n_numeric_outputs_,
        preprocessor.categorical_cardinalities_,
        hidden_dim=64,
        n_blocks=2,
        dropout=0.1,
        architecture=str(variant["architecture"]),
        activation=str(variant["activation"]),
        normalization=str(variant["normalization"]),
    )
    device = torch.device("cpu")
    batch_size = resolve_batch_size(
        "auto",
        n_samples=len(X_train),
        n_numeric_inputs=preprocessor.n_numeric_outputs_,
        categorical_cardinalities=preprocessor.categorical_cardinalities_,
        hidden_dim=64,
        n_blocks=2,
        device=device,
    )
    result = train_binary_model(
        model,
        train_data,
        y_train,
        np.ones(len(y_train)),
        valid_data,
        y_valid,
        np.ones(len(y_valid)),
        device=device,
        batch_size=batch_size,
        max_epochs=12,
        patience=3,
        min_delta=float(variant["min_delta"]),
        eval_frequency=int(variant["eval_frequency"]),
        eval_metric="loss",
        lr=3e-3,
        weight_decay=1e-5,
        random_state=seed,
        verbose=0,
        lr_strategy=str(variant["lr_strategy"]),
    )
    fit_seconds = perf_counter() - started
    probability = predict_probabilities(
        model, test_data, device=device, batch_size=max(1_024, batch_size)
    )
    return {
        "variant": variant["name"],
        "dataset": dataset,
        "seed": seed,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "fit_seconds": fit_seconds,
        "epochs": result.n_iter,
        "parameters": model.parameter_count,
    }


def engine_microbenchmark(seed: int) -> dict[str, float]:
    """Compare DataLoader traversal with direct tensor indexing."""

    generator = torch.Generator().manual_seed(seed)
    numerical = torch.randn(5_000, 20, generator=generator)
    categorical = torch.randint(0, 20, (5_000, 4), generator=generator)
    target = torch.randint(0, 2, (5_000,), generator=generator).float()
    weight = torch.ones(5_000)
    dataset = TensorDataset(numerical, categorical, target, weight)
    started = perf_counter()
    for _ in DataLoader(dataset, batch_size=256, shuffle=True, generator=generator):
        pass
    loader_seconds = perf_counter() - started
    started = perf_counter()
    order = torch.randperm(len(dataset), generator=generator)
    for offset in range(0, len(dataset), 256):
        index = order[offset : offset + 256]
        _ = (
            numerical.index_select(0, index),
            categorical.index_select(0, index),
            target.index_select(0, index),
            weight.index_select(0, index),
        )
    direct_seconds = perf_counter() - started
    return {
        "dataloader_seconds": loader_seconds,
        "direct_index_seconds": direct_seconds,
        "speedup": loader_seconds / direct_seconds,
    }


def variants() -> list[dict[str, object]]:
    candidate = {
        "architecture": "residual",
        "activation": "silu",
        "normalization": "layer_norm",
        "numeric_strategy": "robust",
        "min_category_count": 2,
        "eval_frequency": 2,
        "min_delta": 1e-4,
        "lr_strategy": "cosine",
    }
    changes = [
        ("candidate", {}),
        ("plain_mlp", {"architecture": "plain"}),
        ("gelu", {"activation": "gelu"}),
        ("no_normalization", {"normalization": "none"}),
        ("standard_preprocessing", {"numeric_strategy": "standard"}),
        ("rare_handling_off", {"min_category_count": 1}),
        ("validation_every_epoch", {"eval_frequency": 1}),
        ("validation_every_3", {"eval_frequency": 3}),
        ("microscopic_delta", {"min_delta": 1e-12}),
        ("constant_lr", {"lr_strategy": "constant"}),
        ("warmup_cosine_lr", {"lr_strategy": "warmup_cosine"}),
    ]
    return [{"name": name, **candidate, **change} for name, change in changes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 31])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for dataset in ["numeric", "mixed"]:
        for seed in args.seeds:
            for variant in variants():
                result = run_variant(variant, dataset, seed)
                results.append(result)
                print(json.dumps(result, sort_keys=True))
    payload = {"results": results, "engine": engine_microbenchmark(args.seeds[0])}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"engine": payload["engine"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Measure cold end-to-end torch.compile cost on one substantial workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from neurotabular.network import TabularNetwork
from neurotabular.preprocessing import TabularPreprocessor
from neurotabular.training import resolve_batch_size, train_binary_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["off", "reduce-overhead"], required=True)
    parser.add_argument(
        "--optimizer", choices=["auto", "foreach", "fused"], default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values, target = make_classification(
        n_samples=8_000,
        n_features=24,
        n_informative=16,
        n_redundant=4,
        random_state=47,
    )
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, target, test_size=0.2, stratify=target, random_state=47
    )
    started = perf_counter()
    preprocessor = TabularPreprocessor().fit(X_train)
    train_data = preprocessor.transform(X_train)
    valid_data = preprocessor.transform(X_valid)
    torch.manual_seed(47)
    model = TabularNetwork(
        preprocessor.n_numeric_outputs_,
        preprocessor.categorical_cardinalities_,
        hidden_dim=64,
        n_blocks=2,
        dropout=0.1,
        n_continuous_features=preprocessor.n_continuous_features_,
        numerical_knots=torch.from_numpy(preprocessor.numeric_knots_),
        numerical_embedding="scalar",
        dataset_size=len(X_train),
    )
    batch_size = resolve_batch_size(
        "auto",
        n_samples=len(X_train),
        n_numeric_inputs=model.input_width,
        categorical_cardinalities=preprocessor.categorical_cardinalities_,
        hidden_dim=64,
        n_blocks=2,
        device=torch.device("cpu"),
    )
    payload: dict[str, object] = {
        "mode": args.mode,
        "optimizer": args.optimizer,
        "rows": len(X),
        "batch_size": batch_size,
        "parameters": model.parameter_count,
        "torch": torch.__version__,
    }
    try:
        result = train_binary_model(
            model,
            train_data,
            y_train.astype(np.float32),
            np.ones(len(y_train)),
            valid_data,
            y_valid.astype(np.float32),
            np.ones(len(y_valid)),
            device=torch.device("cpu"),
            batch_size=batch_size,
            max_epochs=5,
            patience=5,
            min_delta=1e-4,
            eval_frequency=1,
            eval_metric="loss",
            lr=3e-3,
            weight_decay=1e-5,
            random_state=47,
            verbose=0,
            use_amp=False,
            compile_mode=None if args.mode == "off" else "reduce-overhead",
            optimizer_strategy=args.optimizer,
        )
        payload.update(
            {
                "status": "ok",
                "end_to_end_seconds": perf_counter() - started,
                "engine_profile": result.profile,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "unsupported",
                "end_to_end_seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

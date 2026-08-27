"""Reproducible binary-classification benchmark matrix."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from neurotabular import NeuroTabularClassifier


class PeakRSS:
    """Sample process resident memory while a model fits."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.peak = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            sleep(0.005)

    def __enter__(self) -> PeakRSS:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)


def _numeric(seed: int, n: int, features: int) -> tuple[pd.DataFrame, np.ndarray]:
    values, target = make_classification(
        n_samples=n,
        n_features=features,
        n_informative=max(3, features // 2),
        n_redundant=max(1, features // 6),
        class_sep=1.0,
        random_state=seed,
    )
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(features)]), target


def make_dataset(name: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Create one deterministic benchmark dataset."""

    if name == "small_numeric":
        return _numeric(seed, 600, 12)
    if name == "medium_synthetic":
        return _numeric(seed, 5_000, 24)
    if name == "imbalanced_binary":
        values, target = make_classification(
            n_samples=1_200,
            n_features=14,
            n_informative=8,
            weights=[0.9, 0.1],
            class_sep=1.0,
            random_state=seed,
        )
        return pd.DataFrame(values, columns=[f"x{i}" for i in range(14)]), target

    rng = np.random.default_rng(seed)
    n = 800 if name == "small_mixed" else 1_200
    numeric = rng.normal(size=(n, 6))
    frame = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(6)])
    if name == "categorical_heavy":
        for index in range(6):
            frame[f"cat{index}"] = rng.choice(
                [f"c{index}_{value}" for value in range(8)], n
            )
        signal = numeric[:, 0] + (frame["cat0"] == "c0_1") * 1.2
    elif name == "moderate_high_cardinality":
        customer = rng.integers(0, 300, n)
        frame["customer"] = pd.Series(customer).map(lambda value: f"id_{value}")
        frame["segment"] = rng.choice([f"s{i}" for i in range(20)], n)
        signal = numeric[:, 0] + (customer % 7 == 0) * 0.9
    else:
        frame["city"] = rng.choice(["Rome", "Milan", "Turin", "Naples"], n)
        frame["member"] = rng.choice([True, False], n)
        signal = numeric[:, 0] + 0.7 * (frame["city"] == "Milan")
        if name == "mixed_with_nans":
            frame.loc[::11, "x0"] = np.nan
            frame.loc[::13, "city"] = None
            signal = np.nan_to_num(frame["x0"].to_numpy(), nan=0.0)
            signal += 0.7 * (frame["city"] == "Milan")
    signal = np.asarray(signal, dtype=float) + rng.normal(0.0, 0.8, n)
    return frame, (signal > np.median(signal)).astype(np.int64)


def make_encoder(X: pd.DataFrame) -> ColumnTransformer:
    """Build leakage-safe ordinal encoding for heterogeneous baselines."""

    categorical = [
        column
        for column in X.columns
        if X[column].dtype == object
        or isinstance(X[column].dtype, (pd.StringDtype, pd.CategoricalDtype))
        or pd.api.types.is_bool_dtype(X[column].dtype)
    ]
    return ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
                categorical,
            )
        ],
        remainder="passthrough",
    )


def make_histogram_baseline(X: pd.DataFrame, seed: int) -> Pipeline:
    """Build a leakage-safe sklearn histogram baseline."""

    return Pipeline(
        [
            ("preprocessing", make_encoder(X)),
            (
                "classifier",
                HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            ),
        ]
    )


def optional_external_baselines(X: pd.DataFrame, seed: int) -> list[tuple[str, object]]:
    """Return installed benchmark-only baselines, never runtime dependencies."""

    models: list[tuple[str, object]] = []
    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "LightGBM",
                Pipeline(
                    [
                        ("preprocessing", make_encoder(X)),
                        (
                            "classifier",
                            LGBMClassifier(
                                n_estimators=100,
                                learning_rate=0.05,
                                num_leaves=31,
                                random_state=seed,
                                n_jobs=1,
                                verbosity=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    except (ImportError, OSError):
        pass
    try:
        from catboost import CatBoostClassifier

        models.append(
            (
                "CatBoost",
                Pipeline(
                    [
                        ("preprocessing", make_encoder(X)),
                        (
                            "classifier",
                            CatBoostClassifier(
                                iterations=100,
                                depth=6,
                                learning_rate=0.05,
                                random_seed=seed,
                                thread_count=1,
                                verbose=False,
                                allow_writing_files=False,
                            ),
                        ),
                    ]
                ),
            )
        )
    except (ImportError, OSError):
        pass
    return models


def measure_model(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, object]:
    """Fit and measure one model."""

    start_rss = psutil.Process().memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with PeakRSS() as memory:
        started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
    started = perf_counter()
    probability = model.predict_proba(X_test)[:, 1]
    predict_seconds = perf_counter() - started
    result = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "epochs": getattr(model, "n_iter_", None),
        "preprocessing_seconds": getattr(model, "preprocessing_time_", None),
        "parameters": getattr(model, "n_parameters_", None),
        "peak_rss_delta_mib": max(0.0, (memory.peak - start_rss) / 2**20),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else None
        ),
    }
    return result


def load_legacy(args: argparse.Namespace):
    """Load an optional legacy estimator without embedding it in this project."""

    if not args.legacy_source:
        return None
    sys.path.insert(0, str(Path(args.legacy_source).resolve()))
    module = importlib.import_module(args.legacy_module)
    return getattr(module, args.legacy_class)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy-source")
    parser.add_argument("--legacy-module", default="legacy_package")
    parser.add_argument("--legacy-class", default="LegacyClassifier")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark matrix and emit JSON."""

    args = parse_args()
    datasets = [
        "small_numeric",
        "small_mixed",
        "mixed_with_nans",
        "categorical_heavy",
        "moderate_high_cardinality",
        "imbalanced_binary",
        "medium_synthetic",
    ]
    legacy_class = load_legacy(args)
    results: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in args.seeds:
            X, y = make_dataset(dataset, seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            models: list[tuple[str, object]] = [
                (
                    "NeuroTabular",
                    NeuroTabularClassifier(random_state=seed, device="cpu"),
                ),
                ("HistGradientBoosting", make_histogram_baseline(X_train, seed)),
            ]
            models.extend(optional_external_baselines(X_train, seed))
            if legacy_class is not None and seed == args.seeds[0]:
                models.append(
                    (
                        "LegacyPrototype",
                        legacy_class(
                            hidden_dim=64,
                            n_layers=2,
                            epochs=30,
                            patience=4,
                            batch_size=256,
                            random_state=seed,
                            device="cpu",
                        ),
                    )
                )
            for model_name, model in models:
                measured = measure_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                measured.update({"dataset": dataset, "seed": seed, "rows": len(X)})
                results.append(measured)
                print(json.dumps(measured, sort_keys=True))
    payload = {
        "hardware": {
            "platform": sys.platform,
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

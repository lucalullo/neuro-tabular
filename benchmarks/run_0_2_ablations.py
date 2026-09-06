"""Cumulative NeuroTabular 0.2 architecture ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from run_benchmarks import PeakRSS, make_dataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from neurotabular import NeuroTabularClassifier

VARIANTS = {
    "scalar": {
        "numerical_embedding": "scalar",
        "use_category_frequency": False,
        "feature_gating": False,
    },
    "affine": {
        "numerical_embedding": "affine",
        "use_category_frequency": False,
        "feature_gating": False,
    },
    "scalar_frequency": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
    },
    "scalar_frequency_refit": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
        "full_data_refit": True,
    },
    "scalar_gating": {
        "numerical_embedding": "scalar",
        "use_category_frequency": False,
        "feature_gating": True,
    },
    "scalar_frequency_gating": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": True,
    },
    "affine_frequency": {
        "numerical_embedding": "affine",
        "use_category_frequency": True,
        "feature_gating": False,
    },
    "categorical_dropout": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
        "_experimental_categorical_dropout": 0.05,
    },
    "embedding_dropout": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
        "_experimental_embedding_dropout": 0.05,
    },
    "frequency_capped_64": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
        "_experimental_max_categories": 64,
    },
    "frequency_capped_hashed_16": {
        "numerical_embedding": "scalar",
        "use_category_frequency": True,
        "feature_gating": False,
        "_experimental_max_categories": 64,
        "_experimental_hash_buckets": 16,
    },
    "periodic": {
        "numerical_embedding": "periodic",
        "use_category_frequency": False,
        "feature_gating": False,
    },
    "piecewise": {
        "numerical_embedding": "piecewise",
        "use_category_frequency": False,
        "feature_gating": False,
    },
    "piecewise_frequency": {
        "numerical_embedding": "piecewise",
        "use_category_frequency": True,
        "feature_gating": False,
    },
    "piecewise_frequency_gating": {
        "numerical_embedding": "piecewise",
        "use_category_frequency": True,
        "feature_gating": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 31])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "small_numeric",
            "mixed_with_nans",
            "categorical_heavy",
            "moderate_high_cardinality",
            "imbalanced_binary",
            "medium_synthetic",
        ],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_variants = args.variants or list(VARIANTS)
    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            X, y = make_dataset(dataset, seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            for variant in selected_variants:
                settings = VARIANTS[variant]
                public_settings = {
                    key: value
                    for key, value in settings.items()
                    if not key.startswith("_experimental_")
                }
                model = NeuroTabularClassifier(
                    **public_settings,
                    max_epochs=args.epochs,
                    random_state=seed,
                    device="cpu",
                )
                for key, value in settings.items():
                    if key.startswith("_experimental_"):
                        setattr(model, key, value)
                start_rss = psutil.Process().memory_info().rss
                with PeakRSS() as memory:
                    started = perf_counter()
                    model.fit(X_train, y_train)
                    fit_seconds = perf_counter() - started
                started = perf_counter()
                probability = model.predict_proba(X_test)[:, 1]
                predict_seconds = perf_counter() - started
                row = {
                    "variant": variant,
                    "dataset": dataset,
                    "seed": seed,
                    "roc_auc": float(roc_auc_score(y_test, probability)),
                    "log_loss": float(log_loss(y_test, probability)),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "epochs": model.n_iter_,
                    "best_epoch": model.best_epoch_,
                    "parameters": model.n_parameters_,
                    "processed_mib": (
                        model._preprocessor_.transform(X_train).nbytes / 2**20
                    ),
                    "peak_rss_delta_mib": max(0.0, (memory.peak - start_rss) / 2**20),
                }
                results.append(row)
                print(json.dumps(row, sort_keys=True))
    payload = {
        "variants": {name: VARIANTS[name] for name in selected_variants},
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {}
    for variant in selected_variants:
        rows = [row for row in results if row["variant"] == variant]
        summary[variant] = {
            "mean_auc": float(np.mean([row["roc_auc"] for row in rows])),
            "mean_log_loss": float(np.mean([row["log_loss"] for row in rows])),
            "median_fit_seconds": float(
                np.median([row["fit_seconds"] for row in rows])
            ),
            "median_predict_seconds": float(
                np.median([row["predict_seconds"] for row in rows])
            ),
        }
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()

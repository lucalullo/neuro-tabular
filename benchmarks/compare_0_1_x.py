"""Compare the immutable 0.1.0 baseline with the local 0.1.1 candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from run_benchmarks import make_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from neurotabular import NeuroTabularClassifier


def _load_baseline(source: Path):
    package = source / "src" / "neurotabular"
    spec = importlib.util.spec_from_file_location(
        "neurotabular_0_1_0",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the baseline package from {package}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NeuroTabularClassifier


def _measure(model_class, X_train, y_train, X_test, y_test, seed, prediction_repeats):
    model = model_class(random_state=seed, device="cpu")
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    probabilities = model.predict_proba(X_test)[:, 1]
    prediction_times = []
    for _ in range(prediction_repeats):
        started = perf_counter()
        probabilities = model.predict_proba(X_test)[:, 1]
        prediction_times.append(perf_counter() - started)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "fit_seconds": fit_seconds,
        "predict_seconds": statistics.median(prediction_times),
        "epochs": model.n_iter_,
    }


def _warm(model_class) -> None:
    X, y = make_dataset("small_numeric", 1)
    model_class(
        hidden_dim=8,
        n_blocks=1,
        max_epochs=1,
        patience=1,
        device="cpu",
        random_state=1,
    ).fit(X.iloc[:80], y[:80])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "NeuroTabular-0.1.0",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prediction-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_class = _load_baseline(args.baseline_source.resolve())
    _warm(baseline_class)
    _warm(NeuroTabularClassifier)

    datasets = [
        "small_numeric",
        "small_mixed",
        "mixed_with_nans",
        "categorical_heavy",
        "moderate_high_cardinality",
        "imbalanced_binary",
        "medium_synthetic",
    ]
    results = []
    for dataset_index, dataset in enumerate(datasets):
        for seed_index, seed in enumerate(args.seeds):
            X, y = make_dataset(dataset, seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            for repeat in range(args.repeats):
                versions = [
                    ("0.1.0", baseline_class),
                    ("0.1.1", NeuroTabularClassifier),
                ]
                if (dataset_index + seed_index + repeat) % 2:
                    versions.reverse()
                for version, model_class in versions:
                    result = _measure(
                        model_class,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        seed,
                        args.prediction_repeats,
                    )
                    result.update(
                        {
                            "version": version,
                            "dataset": dataset,
                            "seed": seed,
                            "repeat": repeat + 1,
                        }
                    )
                    results.append(result)
                    print(json.dumps(result, sort_keys=True))

    paired_medians = []
    for version in ["0.1.0", "0.1.1"]:
        for dataset in datasets:
            for seed in args.seeds:
                rows = [
                    row
                    for row in results
                    if row["version"] == version
                    and row["dataset"] == dataset
                    and row["seed"] == seed
                ]
                paired_medians.append(
                    {
                        "version": version,
                        "dataset": dataset,
                        "seed": seed,
                        "roc_auc": rows[0]["roc_auc"],
                        "epochs": rows[0]["epochs"],
                        "fit_seconds": statistics.median(
                            row["fit_seconds"] for row in rows
                        ),
                        "predict_seconds": statistics.median(
                            row["predict_seconds"] for row in rows
                        ),
                    }
                )
    summary = {}
    for version in ["0.1.0", "0.1.1"]:
        rows = [row for row in paired_medians if row["version"] == version]
        summary[version] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ["roc_auc", "fit_seconds", "predict_seconds", "epochs"]
        }
    payload = {
        "method": "warm, interleaved, same process",
        "summary": summary,
        "paired_medians": paired_medians,
        "results": results,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

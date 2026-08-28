"""Profile NeuroTabular fit and inference phases in a fresh process."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import psutil
import torch


class PeakRSS:
    """Sample resident memory without adding a runtime dependency."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.start = self._process.memory_info().rss
        self.peak = self.start
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            sleep(0.002)

    def __enter__(self) -> PeakRSS:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)


def make_profile_data(rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a deterministic mixed workload with missing and rare values."""

    rng = np.random.default_rng(seed)
    numerical = rng.normal(size=(rows, 12))
    frame = pd.DataFrame(numerical, columns=[f"x{i}" for i in range(12)])
    frame["city"] = rng.choice([f"city_{i}" for i in range(24)], rows)
    frame["account"] = rng.choice([f"account_{i}" for i in range(400)], rows)
    frame["active"] = rng.choice([True, False], rows)
    frame.loc[::11, "x0"] = np.nan
    frame.loc[::17, "x3"] = np.nan
    frame.loc[::19, "city"] = None
    signal = (
        np.nan_to_num(frame["x0"].to_numpy(), nan=0.0)
        + 0.8 * numerical[:, 1]
        - 0.5 * numerical[:, 2]
        + 0.9 * (frame["city"] == "city_3").to_numpy()
        + 0.4 * frame["active"].to_numpy()
        + rng.normal(0.0, 0.9, rows)
    )
    return frame, (signal > np.median(signal)).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from neurotabular import NeuroTabularClassifier, __version__

    X, y = make_profile_data(args.rows, args.seed)
    model = NeuroTabularClassifier(
        hidden_dim=32,
        n_blocks=1,
        batch_size="auto",
        max_epochs=args.epochs,
        patience=args.epochs,
        eval_frequency=1,
        eval_metric="roc_auc",
        device="cpu",
        random_state=args.seed,
    )
    with PeakRSS() as memory:
        model.fit(X, y)
        model.predict_proba(X)
    prediction_transform = dict(model._preprocessor_.last_transform_profile_)
    prediction_total = float(model.last_prediction_time_)
    payload = {
        "label": args.label,
        "version": __version__,
        "workload": {
            "rows": args.rows,
            "columns": X.shape[1],
            "epochs_requested": args.epochs,
            "batch_size": model.batch_size_,
            "parameters": model.n_parameters_,
        },
        "hardware": {
            "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "fit_seconds": model.fit_time_,
        "fit_profile": model.profile_,
        "prediction": {
            "total_seconds": prediction_total,
            "preprocessing": prediction_transform,
            "network_seconds_approx": max(
                0.0, prediction_total - prediction_transform["total_seconds"]
            ),
        },
        "peak_rss_delta_mib": max(0.0, (memory.peak - memory.start) / 2**20),
        "peak_vram_mib": None,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

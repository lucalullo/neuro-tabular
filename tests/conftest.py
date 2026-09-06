from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurotabular import NeuroTabularClassifier


@pytest.fixture
def mixed_binary_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(123)
    n_samples = 96
    age = rng.normal(40, 11, n_samples)
    balance = rng.normal(0, 1, n_samples)
    region = rng.choice(["north", "centre", "south"], n_samples).astype(object)
    active = rng.choice([True, False], n_samples)
    age[::13] = np.nan
    region[::17] = None
    X = pd.DataFrame(
        {"age": age, "balance": balance, "region": region, "active": active}
    )
    signal = (
        np.nan_to_num(age, nan=40.0) / 11
        + 0.8 * (region == "north")
        + 0.4 * active
        + balance
    )
    y = (signal > np.median(signal)).astype(np.int64)
    return X, y


@pytest.fixture
def fast_model_kwargs() -> dict[str, object]:
    return {
        "hidden_dim": 8,
        "n_blocks": 1,
        "batch_size": "auto",
        "max_epochs": 1,
        "patience": 1,
        "validation_fraction": 0.2,
        "eval_frequency": 1,
        "device": "cpu",
        "random_state": 9,
    }


@pytest.fixture
def fast_model(fast_model_kwargs) -> NeuroTabularClassifier:
    return NeuroTabularClassifier(**fast_model_kwargs)

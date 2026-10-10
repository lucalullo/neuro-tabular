"""Synthetic P40 software fixtures, independent of research data and recipes."""

import copy
import random

import numpy as np
import pandas as pd
import torch


def data(task):
    n = 60
    X = pd.DataFrame(
        {
            "value": np.linspace(-2, 2, n),
            "count": pd.Series(np.arange(n) % 7, dtype="Int64"),
            "group": pd.Categorical(
                np.resize(["north", "south", None], n),
                categories=["north", "south", "declared"],
            ),
            "flag": pd.Series(np.resize([True, False, None], n), dtype="boolean"),
        }
    )
    X.loc[::11, "value"] = np.nan
    X.loc[::13, "count"] = pd.NA
    y = (
        np.linspace(10, 30, n)
        if task == "regression"
        else np.asarray(["a", "b", "c"])[np.arange(n) % (2 if task == "binary" else 3)]
    )
    return X, y


def kwargs():
    return dict(
        hidden_dim=8,
        n_blocks=1,
        max_epochs=1,
        patience=1,
        device="cpu",
        random_state=17,
    )


def rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def assert_rng_same(before):
    after = rng_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def public_state(model):
    return copy.deepcopy(
        {
            k: v
            for k, v in vars(model).items()
            if k.endswith("_") and not k.startswith("_")
        }
    )

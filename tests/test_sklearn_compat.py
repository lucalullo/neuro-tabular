import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from neurotabular import NeuroTabularClassifier


def test_estimator_is_cloneable():
    estimator = NeuroTabularClassifier(
        hidden_dim=16,
        max_epochs=2,
        categorical_features=["group"],
    )
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()


def test_get_params_and_set_params():
    estimator = NeuroTabularClassifier()
    assert estimator.get_params()["hidden_dim"] == 64
    assert estimator.set_params(hidden_dim=20) is estimator
    assert estimator.hidden_dim == 20


def test_public_defaults_are_release_defaults():
    assert NeuroTabularClassifier().get_params() == {
        "batch_size": "auto",
        "categorical_features": None,
        "class_weight": None,
        "device": "auto",
        "dropout": 0.1,
        "eval_frequency": 1,
        "eval_metric": "loss",
        "feature_gating": False,
        "full_data_refit": False,
        "hidden_dim": 64,
        "lr": 0.003,
        "max_epochs": 30,
        "min_category_count": 2,
        "min_delta": 0.0001,
        "n_blocks": 2,
        "numerical_embedding": "scalar",
        "patience": 4,
        "random_state": 42,
        "validation_fraction": 0.2,
        "verbose": 0,
        "weight_decay": 0.00001,
        "use_category_frequency": True,
    }


def test_cross_val_score_with_roc_auc():
    values, y = make_classification(
        n_samples=72,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=4,
    )
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    estimator = NeuroTabularClassifier(
        hidden_dim=8,
        n_blocks=1,
        max_epochs=1,
        patience=1,
        eval_frequency=1,
        batch_size="auto",
        validation_fraction=0.2,
        device="cpu",
        random_state=4,
    )
    scores = cross_val_score(estimator, X, y, cv=3, scoring="roc_auc")
    assert scores.shape == (3,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

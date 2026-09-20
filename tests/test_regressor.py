import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone, is_regressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from neurotabular import NeuroTabularRegressor
from neurotabular.tasks import TaskSpec
from neurotabular.training import _is_improvement


def model(**kwargs):
    return NeuroTabularRegressor(
        max_epochs=3, hidden_dim=12, n_blocks=1, device="cpu", **kwargs
    )


def data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X, y = X.iloc[:100].copy(), y.iloc[:100].to_numpy(copy=True)
    X["category"] = np.resize(["a", "b", None], len(X))
    X.iloc[::11, 0] = np.nan
    return X, y


@pytest.mark.parametrize(
    "scale,offset", [(1.0, -250.0), (1e180, 0), (1e-180, 0), (1e-9, 1.0), (1.0, 0)]
)
def test_target_scales_inverse_and_schema(scale, offset):
    X, y = data()
    target = y.astype(int) if scale == 1 and offset == 0 else y * scale + offset
    m = model().fit(X, target)
    p = m.predict(X)
    assert p.shape == (len(X),) and np.isfinite(p).all()
    assert m._model_.output.out_features == 1
    assert isinstance(m._task_.criterion(), torch.nn.MSELoss)
    expected = (
        m._predict_values(X) * m._target_spread_ + m._target_center_
    ) * m._target_magnitude_
    assert np.array_equal(p, expected)
    assert np.array_equal(p, m.predict(X[X.columns[::-1]]))
    with pytest.raises(ValueError, match="schema mismatch"):
        m.predict(X.drop(columns="category"))
    with pytest.raises(ValueError, match="schema mismatch"):
        m.predict(X.assign(extra=1))
    assert not hasattr(m, "predict_proba") and not hasattr(m, "classes_")
    assert "class_weight" not in m.get_params()


def test_weighted_train_only_target_statistics_and_zero_rows():
    X, y = data()
    w = np.linspace(0.1, 2, len(y))
    m = model().fit(X, y, sample_weight=w, eval_set=(X.iloc[:10], y[:10] + 1e4))
    mean = np.average(y, weights=w)
    std = np.sqrt(np.average((y - mean) ** 2, weights=w))
    assert m.target_mean_ == pytest.approx(mean)
    assert m.target_scale_ == pytest.approx(std)
    internal = model().fit(X, y, sample_weight=w)
    train, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    assert internal.target_mean_ == pytest.approx(
        np.average(y[train], weights=w[train])
    )
    extra = pd.concat([X, X.iloc[:1].assign(category="never_seen")], ignore_index=True)
    extra.iloc[-1, 0] = 1e200
    zero = model().fit(extra, np.append(y, np.nan), sample_weight=np.append(w, 0))
    assert np.array_equal(internal.predict(X), zero.predict(X))
    assert zero.target_mean_ == internal.target_mean_ and zero.n_ignored_samples_ == 1
    scaled = model().fit(X, y, sample_weight=w * 10)
    assert np.allclose(scaled.predict(X), internal.predict(X), atol=1e-5)


@pytest.mark.parametrize("value", [0.0, -7.25, 1e100])
def test_constant_targets(value):
    X, y = data()
    m = model(full_data_refit=True).fit(
        X, np.full_like(y, value), sample_weight=np.linspace(1, 2, len(y))
    )
    assert m.target_is_constant_ and m.target_scale_ == 1
    assert np.array_equal(m.predict(X), np.full_like(y, value))
    assert m.full_data_refit_


@pytest.mark.parametrize("metric", ["loss", "rmse", "mae", "r2"])
def test_metric_direction_and_restored_checkpoint(metric):
    X, y = data()
    m = model(eval_metric=metric).fit(
        X.iloc[:75], y[:75], eval_set=(X.iloc[75:], y[75:])
    )
    scores = [r["validation_score"] for r in m.history_]
    assert m.best_score_ == (max(scores) if metric == "r2" else min(scores))
    z = m._encode_target(y[75:])
    raw = m._predict_values(X.iloc[75:])
    expected = {
        "loss": mean_squared_error(z, raw),
        "rmse": np.sqrt(mean_squared_error(z, raw)),
        "mae": mean_absolute_error(z, raw),
        "r2": r2_score(z, raw),
    }[metric]
    assert m.best_score_ == pytest.approx(expected, abs=1e-6)
    assert _is_improvement(0.2, 0.3, metric, 0) == (metric != "r2")


def test_shared_regression_metrics_weighting():
    spec = TaskSpec("regression")
    target, pred, w = (
        np.array([0.0, 1.0, 3.0]),
        np.array([1.0, 2.0, 1.0]),
        np.array([1.0, 2.0, 4.0]),
    )
    assert spec.score(target, pred, w, "rmse") == pytest.approx(np.sqrt(19 / 7))
    assert spec.score(target, pred, w, "mae") == pytest.approx(11 / 7)
    assert spec.score(target, pred, w, "r2") == pytest.approx(
        r2_score(target, pred, sample_weight=w)
    )


def test_regression_sklearn_and_rng_and_refit():
    X, y = data()
    m = model(full_data_refit=True)
    assert is_regressor(m) and clone(m).get_params() == m.get_params()
    py, npstate, ts = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
    )
    m.fit(X, y)
    assert random.getstate() == py and np.array_equal(
        np.random.get_state()[1], npstate[1]
    )
    assert torch.equal(torch.random.get_rng_state(), ts)
    assert m.target_mean_ == pytest.approx(y.mean())
    scores = cross_val_score(make_pipeline(model()), X, y, cv=3)
    assert np.isfinite(scores).all()
    expected = m.predict(X)
    with pytest.raises(ValueError):
        m.fit(X, np.full_like(y, np.inf))
    assert np.array_equal(m.predict(X), expected)
    assert np.array_equal(model(full_data_refit=True).fit(X, y).predict(X), expected)


@pytest.mark.parametrize("kind", ["complex", "nan", "inf", "string", "2d", "short"])
def test_invalid_regression_targets(kind):
    X, y = data()
    if kind == "complex":
        y = y.astype(complex)
    elif kind in {"nan", "inf"}:
        y[0] = np.nan if kind == "nan" else np.inf
    elif kind == "string":
        y = y.astype(str)
    elif kind == "2d":
        y = y[:, None]
    else:
        y = y[:-1]
    with pytest.raises(ValueError):
        model().fit(X, y)


def test_regression_metric_and_validation_errors():
    X, y = data()
    with pytest.raises(ValueError, match="eval_metric"):
        model(eval_metric="accuracy").fit(X, y)
    with pytest.raises(ValueError, match="two validation rows"):
        model(eval_metric="r2").fit(X, y, eval_set=(X.iloc[:1], y[:1]))
    with pytest.raises(ValueError, match="two validation rows"):
        model(eval_metric="r2").fit(X.iloc[:4], y[:4])
    with pytest.raises(ValueError, match="finite"):
        model().fit(X, y, eval_set=(X.iloc[:3], [1, 2, np.inf]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA unavailable")
def test_regression_physical_cuda():
    X, y = data()
    m = NeuroTabularRegressor(max_epochs=2, device="cuda").fit(X, y)
    assert np.isfinite(m.predict(X)).all()

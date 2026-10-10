"""Regression and randomized checks for the pre-1.0 target contract."""

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from p40_helpers import assert_rng_same, kwargs, public_state, rng_snapshot
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import type_of_target

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor

N = 48
X = pd.DataFrame({"value": np.linspace(-1, 1, N)})
VALID = [
    [0, 1],
    [False, True],
    ["left", "right"],
    [-2.0, 7.0],
    [0, 1, 2],
    ["a", "b", "c"],
    [-3.0, 0.0, 7.0],
    np.asarray([0, 1], dtype=object),
    np.asarray([False, True], dtype=object),
    np.asarray([-2.0, 7.0], dtype=object),
    np.asarray(["left", "right"], dtype=object),
]
INVALID = [
    np.resize([0.1, 0.5], N),
    np.linspace(0.1, 1.1, N),
    np.resize([0.0, np.nan], N),
    np.resize([0.0, np.inf], N),
    np.resize([0.0, -np.inf], N),
    np.resize([0.0, 1.0, np.inf], N),
    np.resize([0.0, np.inf], N).astype(object),
    np.resize(np.asarray(["a", np.inf], dtype=object), N),
    np.zeros((N, 1)),
    np.resize([[0, 1], [1, 0]], (N, 2)),
    np.resize([[0.1, 0.5], [0.7, 0.9]], (N, 2)),
    np.resize([[0, 2], [1, 3]], (N, 2)),
]


@pytest.fixture(scope="module")
def binary_model():
    return NeuroTabularClassifier(**kwargs()).fit(X, np.resize([0, 1], N))


@pytest.fixture(scope="module")
def regression_model():
    return NeuroTabularRegressor(**kwargs()).fit(X, np.linspace(1, 3, N))


@pytest.mark.parametrize("labels", VALID)
@pytest.mark.parametrize("external", [False, True])
def test_valid_targets_fit_eval_score_and_original_dtype(labels, external):
    y = np.resize(labels, N)
    inferred = pd.Series(y).infer_objects().to_numpy() if y.dtype.kind == "O" else y
    assert type_of_target(inferred) in {"binary", "multiclass"}
    m = NeuroTabularClassifier(**kwargs()).fit(
        X, y, **({"eval_set": (X, y)} if external else {})
    )
    p = m.predict(X)
    assert p.dtype == inferred.dtype
    np.testing.assert_array_equal(m.classes_, np.unique(inferred))
    assert m.score(X, y) == accuracy_score(inferred, p)
    weight = np.linspace(0.1, 2, N)
    assert m.score(X, y, weight) == accuracy_score(inferred, p, sample_weight=weight)
    assert np.isfinite(m.predict_proba(X)).all()


@pytest.mark.parametrize("y", INVALID)
@pytest.mark.parametrize("route", ["split", "external_primary", "eval", "score"])
@pytest.mark.parametrize("warning_filter", ["always", "error"])
def test_invalid_targets_rejected_consistently_before_training_or_prediction(
    y, route, warning_filter, binary_model, monkeypatch
):
    def downstream(*args, **kwargs):
        pytest.fail("Invalid target reached training or prediction")

    monkeypatch.setattr("neurotabular._base.train_model", downstream)
    m = NeuroTabularClassifier(**kwargs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(warning_filter)
        with pytest.raises(ValueError):
            if route == "score":
                monkeypatch.setattr(binary_model, "predict", downstream)
                binary_model.score(X, y)
            elif route == "eval":
                m.fit(X, np.resize([0, 1], N), eval_set=(X, y))
            else:
                m.fit(
                    X,
                    y,
                    **(
                        {"eval_set": (X, np.resize([0, 1], N))}
                        if route == "external_primary"
                        else {}
                    ),
                )
        assert not caught


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("route", ["split", "external_primary", "eval", "score"])
@pytest.mark.parametrize("warning_filter", ["always", "error"])
def test_regression_finite_target_contract(
    bad, route, warning_filter, regression_model
):
    y = np.linspace(1, 3, N)
    y[0] = bad
    m = NeuroTabularRegressor(**kwargs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(warning_filter)
        with pytest.raises(ValueError, match="missing values|finite values"):
            if route == "score":
                regression_model.score(X, y)
            elif route == "eval":
                m.fit(X, np.linspace(1, 3, N), eval_set=(X, y))
            else:
                m.fit(
                    X,
                    y,
                    **(
                        {"eval_set": (X, np.linspace(1, 3, N))}
                        if route == "external_primary"
                        else {}
                    ),
                )
        assert not caught


@pytest.mark.parametrize("y", INVALID)
@pytest.mark.parametrize("external", [False, True])
def test_invalid_refit_preserves_fitted_state_and_rng(y, external):
    valid = np.resize([0, 1], N)
    m = NeuroTabularClassifier(**kwargs()).fit(X, valid)
    p, proba = m.predict(X), m.predict_proba(X)
    public = pickle.dumps(public_state(m))
    network, encoder = m._model_, m._preprocessor_
    schema, version = copy.deepcopy(m.input_schema_), m._fitted_neurotabular_version_
    before = rng_snapshot()
    with pytest.raises(ValueError):
        m.fit(
            X,
            valid if external else y,
            **({"eval_set": (X, y)} if external else {}),
        )
    assert_rng_same(before)
    assert pickle.dumps(public_state(m)) == public
    assert m._model_ is network and m._preprocessor_ is encoder
    assert m.input_schema_ == schema and m._fitted_neurotabular_version_ == version
    np.testing.assert_array_equal(m.predict(X), p)
    np.testing.assert_array_equal(m.predict_proba(X), proba)


@pytest.mark.parametrize("bad", [0.25, np.nan, np.inf, -np.inf])
def test_zero_weight_invalid_classification_rows_remain_ignored(bad):
    y = np.resize([0.0, 1.0], N)
    expected = NeuroTabularClassifier(**kwargs()).fit(X, y)
    extra_X = pd.concat([X, X.iloc[:1]], ignore_index=True)
    m = NeuroTabularClassifier(**kwargs()).fit(
        extra_X, np.append(y, bad), sample_weight=np.append(np.ones(N), 0)
    )
    np.testing.assert_array_equal(m.classes_, expected.classes_)
    np.testing.assert_array_equal(m.predict_proba(X), expected.predict_proba(X))
    assert m.n_ignored_samples_ == 1


def test_ignored_object_label_does_not_change_effective_target_type():
    y = np.resize([0, 1], N)
    expected = NeuroTabularClassifier(**kwargs()).fit(X, y, eval_set=(X, y))
    extra_X = pd.concat([X, X.iloc[:1]], ignore_index=True)
    target = np.append(y.astype(object), "ignored-only-label")
    m = NeuroTabularClassifier(**kwargs()).fit(
        extra_X,
        target,
        sample_weight=np.append(np.ones(N), 0),
        eval_set=(X, y),
    )
    np.testing.assert_array_equal(m.classes_, expected.classes_)
    assert m.predict(X).dtype == expected.predict(X).dtype
    np.testing.assert_array_equal(m.predict_proba(X), expected.predict_proba(X))
    assert m.score(X, y) == expected.score(X, y)


@pytest.mark.parametrize("cls", [NeuroTabularClassifier, NeuroTabularRegressor])
def test_unfitted_score_keeps_not_fitted_precedence(cls):
    with pytest.raises(NotFittedError):
        cls(**kwargs()).score(None, [np.inf])


@pytest.mark.parametrize("labels", VALID)
def test_default_cv_score_and_pipeline(labels):
    y = np.resize(labels, N)
    # sklearn's CV splitter types its input before delegating to the estimator.
    if y.dtype.kind == "O":
        y = pd.Series(y).infer_objects().to_numpy()
    scores = cross_val_score(
        make_pipeline(NeuroTabularClassifier(**kwargs())),
        X,
        y,
        cv=2,
        error_score="raise",
    )
    assert np.isfinite(scores).all()


def test_score_single_class_and_new_truth_labels_are_valid(binary_model):
    y = np.zeros(N, dtype=int)
    assert binary_model.score(X, y) == accuracy_score(y, binary_model.predict(X))
    assert binary_model.score(X, np.full(N, 7)) == 0


def test_valid_regression_score_is_unchanged(regression_model):
    y = np.linspace(1, 3, N)
    p = regression_model.predict(X)
    assert regression_model.score(X, y) == r2_score(y, p)
    w = np.linspace(0.1, 2, N)
    assert regression_model.score(X, y, w) == r2_score(y, p, sample_weight=w)


@pytest.mark.parametrize("seed", range(512))
def test_randomized_target_type_oracle(seed):
    rng = np.random.default_rng(seed)
    kind = seed % 12
    y = rng.integers(-3, 4, size=24)
    if kind == 1:
        y = np.asarray([f"class_{v}" for v in y])
    elif kind == 2:
        y = y.astype(float)
    elif kind == 3:
        y = y % 2 == 0
    elif kind == 4:
        y = y.astype(float) + 0.25
    elif kind == 5:
        y = y.astype(float)
        y[rng.integers(24)] = np.nan
    elif kind == 6:
        y = y.astype(float)
        y[rng.integers(24)] = rng.choice([-np.inf, np.inf])
    elif kind == 7:
        y = y.reshape(12, 2)
    elif kind == 8:
        y = (y % 2).reshape(12, 2)
    elif kind == 9:
        y = y.astype(object)
    elif kind == 10:
        y = pd.Series(y, dtype="Int64")
    elif kind == 11:
        y = y.astype(float) * 1e300
    values = np.asarray(y)
    if kind == 9:
        # Independent scalar-array inference for the numeric-object sample.
        values = np.asarray(values.tolist())
    missing = pd.isna(values).any()
    finite = not (values.dtype.kind in "biufc" and not np.isfinite(values).all())
    expected = False
    if values.ndim == 1 and not missing and finite:
        with np.errstate(invalid="ignore", over="ignore"):
            expected = type_of_target(values) in {"binary", "multiclass"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        if expected:
            out = NeuroTabularClassifier._classification_target(y, len(values))
            np.testing.assert_array_equal(out, values)
        else:
            with pytest.raises(ValueError):
                NeuroTabularClassifier._classification_target(y, len(values))
        assert not caught


@pytest.mark.parametrize("seed", range(96))
def test_randomized_public_fit_and_score_routes(seed):
    rng = np.random.default_rng(seed)
    count = 2 + seed % 3
    labels = rng.choice(np.arange(-100, 101), size=count, replace=False)
    if seed % 4 == 1:
        labels = labels.astype(str)
    elif seed % 4 == 2:
        labels = labels.astype(float)
    elif seed % 4 == 3:
        labels = labels.astype(float) + 0.25
    y = np.resize(labels, N)
    rng.shuffle(y)
    expected = type_of_target(y) in {"binary", "multiclass"}
    m = NeuroTabularClassifier(**kwargs())
    if expected:
        m.fit(X, y, **({"eval_set": (X, y)} if seed % 2 else {}))
        assert m.score(X, y) == accuracy_score(y, m.predict(X))
    else:
        with pytest.raises(ValueError, match="discrete class labels"):
            m.fit(X, y, **({"eval_set": (X, y)} if seed % 2 else {}))

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from neurotabular import NeuroTabularClassifier


def model(**kwargs):
    return NeuroTabularClassifier(
        max_epochs=3, hidden_dim=12, n_blocks=1, device="cpu", **kwargs
    )


@pytest.mark.parametrize("classes", [3, 5])
@pytest.mark.parametrize("labels", ["string", "integer"])
def test_multiclass_shapes_encoding_weights_and_missing(classes, labels):
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=13,
    )
    X = pd.DataFrame(X)
    X.columns = [str(c) for c in X.columns]
    X["category"] = np.resize(["a", "b", None], len(X))
    X.loc[::11, "0"] = np.nan
    names = (
        np.array([f"label_{i}" for i in range(classes)])
        if labels == "string"
        else np.arange(classes) * 7 - 20
    )
    target = names[y]
    weights = np.linspace(0.2, 2, len(X))
    m = model(class_weight="balanced").fit(X, target, sample_weight=weights)
    p = m.predict_proba(X)
    assert p.shape == (len(X), classes) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    assert np.isfinite(log_loss(target, p))  # warnings are errors in the suite
    assert np.array_equal(m.classes_, np.sort(names))
    assert np.array_equal(m.predict(X), m.classes_[p.argmax(1)])
    assert set(m.predict(X)) <= set(names)
    assert m._model_.output.out_features == classes
    assert isinstance(m._task_.criterion(), torch.nn.CrossEntropyLoss)
    assert set(m.class_weight_) == set(names)
    assert np.array_equal(m.predict_proba(X[X.columns[::-1]]), p)
    with pytest.raises(ValueError, match="schema mismatch"):
        m.predict(X.drop(columns="category"))
    with pytest.raises(ValueError, match="schema mismatch"):
        m.predict(X.assign(extra=1))


def test_multiclass_zero_weight_rows_are_absent_before_discovery():
    X, y = load_iris(return_X_y=True, as_frame=True)
    base = model().fit(X, y)
    extra = pd.concat(
        [X, pd.DataFrame([[1e12] * 4], columns=X.columns)], ignore_index=True
    )
    trained = model().fit(
        extra, np.append(y, 999), sample_weight=np.append(np.ones(len(X)), 0)
    )
    assert trained.n_classes_ == 3 and trained.n_ignored_samples_ == 1
    assert np.array_equal(base.predict_proba(X), trained.predict_proba(X))


@pytest.mark.parametrize("metric", ["loss", "accuracy"])
def test_multiclass_eval_set_clone_pipeline_cv_and_persistence(metric, tmp_path):
    X, y = load_iris(return_X_y=True, as_frame=True)
    m = model(eval_metric=metric).fit(X, y, eval_set=(X.iloc[::3], y.iloc[::3]))
    assert is_classifier(m) and clone(m).get_params() == m.get_params()
    scores = cross_val_score(
        make_pipeline(model(eval_metric=metric)), X, y, cv=3, scoring="accuracy"
    )
    assert np.isfinite(scores).all()
    path = tmp_path / "multiclass.joblib"
    joblib.dump(m, path)
    assert np.array_equal(joblib.load(path).predict_proba(X), m.predict_proba(X))
    with pytest.raises(ValueError, match="not present in training"):
        m.fit(X, y, eval_set=(X.iloc[:3], [999] * 3))
    assert np.isfinite(m.predict_proba(X)).all()  # failed refit is transactional


def test_multiclass_auc_is_explicitly_unsupported():
    X, y = load_iris(return_X_y=True, as_frame=True)
    with pytest.raises(ValueError, match="binary-only"):
        model(eval_metric="roc_auc").fit(X, y)


def test_multiclass_full_refit_and_determinism():
    X, y = load_iris(return_X_y=True, as_frame=True)
    a, b = model(full_data_refit=True).fit(X, y), model(full_data_refit=True).fit(X, y)
    assert a.full_data_refit_ and np.array_equal(a.predict_proba(X), b.predict_proba(X))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA unavailable")
def test_multiclass_physical_cuda():
    X, y = load_iris(return_X_y=True, as_frame=True)
    m = NeuroTabularClassifier(max_epochs=2, device="cuda").fit(X, y)
    assert np.allclose(m.predict_proba(X).sum(1), 1, atol=1e-6)

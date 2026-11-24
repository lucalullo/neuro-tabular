import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor
from neurotabular._cross_validation import make_plan


def test_internal_cv_is_disabled_and_optional():
    for cls in (NeuroTabularClassifier, NeuroTabularRegressor):
        model = cls()
        assert model.cv is None
        assert clone(cls(cv=3)).cv == 3
        assert not hasattr(model, "cv_results_")
        assert model.max_epochs == 30
    assert NeuroTabularRegressor().patience == 8


def test_internal_cv_nested_group_roles_without_training():
    X = pd.DataFrame({"value": np.arange(96, dtype=float)})
    y, groups = np.arange(96) % 3, np.arange(96) // 6
    plan = make_plan(NeuroTabularClassifier(cv=3), X, y, groups=groups)
    seen = []
    for fold in plan.folds:
        fit, stop = fold.train[fold.fit], fold.train[fold.stop]
        assert not set(groups[fit]) & set(groups[stop])
        assert not set(groups[fold.train]) & set(groups[fold.validation])
        seen.extend(fold.validation)
    assert sorted(seen) == list(range(96))


def test_internal_cv_invalid_support_has_no_training_dispatch():
    X = pd.DataFrame({"value": np.arange(12, dtype=float)})
    with pytest.raises(ValueError):
        make_plan(NeuroTabularClassifier(cv=3), X, [0] * 10 + [1] * 2)

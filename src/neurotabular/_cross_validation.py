"""Optional TRAIN-only internal CV. Imported only when cv is enabled.

Outer validation rows never select a checkpoint. Each fold has its own inner
FIT/STOP, followed by a fresh final fit. Fold models are not retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    validation: np.ndarray
    fit: np.ndarray
    stop: np.ndarray


@dataclass(frozen=True)
class Plan:
    X: pd.DataFrame
    y: np.ndarray
    weight: np.ndarray | None
    groups: np.ndarray | None
    folds: tuple[Fold, ...]
    final_fit: np.ndarray
    final_stop: np.ndarray
    original_rows: int
    signature: str


def validate_cv(value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 2
    ):
        raise ValueError("cv must be None (disabled) or an integer >= 2.")


def _indices(values):
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _group_codes(groups, n_rows, positive):
    if groups is None:
        return None
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("groups must be one-dimensional and match X positionally.")
    values = values[positive]
    try:
        if pd.isna(values).any():
            raise ValueError("Positive-weight rows must have nonmissing groups.")
        codes, _ = pd.factorize(values, sort=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "groups must contain nonmissing, hashable scalar IDs."
        ) from exc
    return codes


def _inner_indices(model, y, groups, classification):
    indices = np.arange(len(y))
    if groups is None:
        return train_test_split(
            indices,
            test_size=model.validation_fraction,
            random_state=model.random_state,
            shuffle=True,
            stratify=y if classification else None,
        )
    n_groups = np.unique(groups).size
    if n_groups < 2:
        raise ValueError("At least two independent groups are required for FIT/STOP.")
    if classification:
        # Group sizes and class constraints prevent an exact requested fraction.
        # This fixed TRAIN-only rule never searches alternative lucky partitions.
        splits = min(n_groups, max(2, int(round(1 / model.validation_fraction))))
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=model.random_state
        )
        return next(splitter.split(indices, y, groups))
    return next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=model.validation_fraction,
            random_state=model.random_state,
        ).split(indices, y, groups)
    )


def _check_split(model, y, weight, groups, train, valid, classification, classes):
    if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
        raise ValueError("Invalid or overlapping CV roles.")
    if groups is not None and np.intersect1d(groups[train], groups[valid]).size:
        raise ValueError("A group crosses CV roles.")
    if classification:
        if not np.array_equal(np.unique(y[train]), classes):
            raise ValueError(
                "Every CV FIT must contain all positive-weight TRAIN classes."
            )
        if model.eval_metric == "roc_auc" and np.unique(y[valid]).size != 2:
            raise ValueError(
                "roc_auc requires both classes in every CV STOP/validation."
            )
    elif model.eval_metric == "r2" and len(valid) < 2:
        raise ValueError("r2 requires at least two rows in every CV STOP/validation.")
    if weight is not None:
        for part in (train, valid):
            model._normalize_weight(weight[part], name="CV sample_weight")


def make_plan(model, X, y, sample_weight=None, groups=None):
    """Validate every role before the first neural training dispatch."""
    validate_cv(model.cv)
    model._validate_hyperparameters()
    X = model._validate_X(X)
    n_rows = len(X)
    raw = model._target_array(y, n_rows, allow_missing=True)
    weights = model._validate_sample_weight(sample_weight, n_rows)
    positive = np.ones(n_rows, dtype=bool) if weights is None else weights > 0
    codes = _group_codes(groups, n_rows, positive)
    X = X.iloc[np.flatnonzero(positive)]
    target = model._validate_target(raw[positive], len(X))
    weights = None if weights is None else weights[positive].copy()
    classification = model._task_.name != "regression"
    classes = np.unique(target) if classification else None
    if model.cv is None:
        raise ValueError("CV plan requires an enabled cv value.")
    if len(X) < model.cv:
        raise ValueError("cv exceeds the number of positive-weight rows.")
    if codes is not None and np.unique(codes).size < model.cv:
        raise ValueError("cv exceeds the number of independent groups.")
    if classification and np.unique(target, return_counts=True)[1].min() < model.cv:
        raise ValueError("Each class needs at least cv positive-weight rows.")
    if classification:
        splitter = (
            StratifiedKFold(
                n_splits=model.cv, shuffle=True, random_state=model.random_state
            )
            if codes is None
            else StratifiedGroupKFold(
                n_splits=model.cv, shuffle=True, random_state=model.random_state
            )
        )
    else:
        splitter = (
            KFold(n_splits=model.cv, shuffle=True, random_state=model.random_state)
            if codes is None
            else GroupKFold(n_splits=model.cv)
        )
    all_rows = np.arange(len(X))
    folds = []
    seen = np.zeros(len(X), dtype=int)
    for train, validation in splitter.split(all_rows, target, codes):
        _check_split(
            model, target, weights, codes, train, validation, classification, classes
        )
        inner_fit, inner_stop = _inner_indices(
            model,
            target[train],
            None if codes is None else codes[train],
            classification,
        )
        inner_weight = None if weights is None else weights[train]
        inner_groups = None if codes is None else codes[train]
        _check_split(
            model,
            target[train],
            inner_weight,
            inner_groups,
            inner_fit,
            inner_stop,
            classification,
            classes,
        )
        seen[validation] += 1
        folds.append(Fold(*map(_indices, (train, validation, inner_fit, inner_stop))))
    if not np.all(seen == 1):
        raise ValueError(
            "Every positive-weight row must have exactly one outer CV role."
        )
    final_fit, final_stop = _inner_indices(model, target, codes, classification)
    _check_split(
        model, target, weights, codes, final_fit, final_stop, classification, classes
    )
    signature = sha256()
    signature.update(np.flatnonzero(positive).astype("<i8").tobytes())
    for fold in folds:
        for part in (fold.train, fold.validation, fold.fit, fold.stop):
            signature.update(len(part).to_bytes(8, "little"))
            signature.update(part.astype("<i8").tobytes())
    for part in (final_fit, final_stop):
        signature.update(np.asarray(part, dtype="<i8").tobytes())
    if codes is not None:
        signature.update(codes.astype("<i8").tobytes())
    return Plan(
        X,
        target,
        weights,
        codes,
        tuple(folds),
        _indices(final_fit),
        _indices(final_stop),
        n_rows,
        signature.hexdigest(),
    )


def _regression_score(target, prediction, weight, metric):
    magnitude = max(
        float(np.abs(target).max()),
        float(np.abs(prediction).max()),
        np.finfo(float).tiny,
    )
    y, p = target / magnitude, prediction / magnitude
    squared = float(np.average((y - p) ** 2, weights=weight))
    if metric in {"loss", "rmse"}:
        score = np.sqrt(squared) * magnitude
        name = "rmse_original_units"
    elif metric == "mae":
        score = np.average(np.abs(y - p), weights=weight) * magnitude
        name = "mae_original_units"
    else:
        denominator = float(
            np.average((y - np.average(y, weights=weight)) ** 2, weights=weight)
        )
        score = (
            (1.0 if squared == 0 else 0.0)
            if denominator == 0
            else 1 - squared / denominator
        )
        name = "r2"
    if not np.isfinite(score):
        raise ValueError("CV metric cannot be represented as finite float64.")
    return name, float(score)


def score_fold(model, X, y, sample_weight=None):
    """One OOF API call, after checkpoint selection is complete."""
    weight = (
        None
        if sample_weight is None
        else model._normalize_weight(sample_weight, name="CV score weight")
    )
    if model._task_.name == "regression":
        return _regression_score(y, model.predict(X), weight, model.eval_metric)
    probability = model.predict_proba(X)
    if model.eval_metric == "loss":
        return "nll", float(
            log_loss(y, probability, labels=model.classes_, sample_weight=weight)
        )
    if model.eval_metric == "roc_auc":
        return "roc_auc", float(
            roc_auc_score(
                y == model.classes_[1], probability[:, 1], sample_weight=weight
            )
        )
    encoded = (
        (probability[:, 1] >= 0.5).astype(int)
        if len(model.classes_) == 2
        else probability.argmax(1)
    )
    return "accuracy", float(
        accuracy_score(y, model.classes_[encoded], sample_weight=weight)
    )


def fit_internal_cv(model, X, y, sample_weight=None, groups=None):
    started = perf_counter()
    plan = make_plan(model, X, y, sample_weight, groups)
    planned = perf_counter()
    results = []
    sums = {
        name: 0.0
        for name in ("training_time_", "validation_time_", "preprocessing_time_")
    }
    for number, fold in enumerate(plan.folds):
        child = clone(model).set_params(cv=None)
        child._cv_inner_split = (fold.fit, fold.stop)
        weights = None if plan.weight is None else plan.weight[fold.train]
        child.fit(plan.X.iloc[fold.train], plan.y[fold.train], sample_weight=weights)
        scoring_started = perf_counter()
        metric, score = score_fold(
            child,
            plan.X.iloc[fold.validation],
            plan.y[fold.validation],
            None if plan.weight is None else plan.weight[fold.validation],
        )
        if not np.isfinite(score):
            raise ValueError("CV score is not finite.")
        results.append(
            {
                "fold": number,
                "fit_rows": len(fold.fit),
                "stop_rows": len(fold.stop),
                "validation_rows": len(fold.validation),
                "outer_train_rows": len(fold.train),
                "actual_stop_fraction": len(fold.stop) / len(fold.train),
                "best_epoch": int(child.best_epoch_),
                "n_iter": int(child.n_iter_),
                "score": score,
                "fit_seconds": float(child.fit_time_),
                "scoring_seconds": perf_counter() - scoring_started,
                "full_data_refit": bool(child.full_data_refit_),
            }
        )
        for name in sums:
            sums[name] += float(getattr(child, name))
        # No state dict, preprocessor, fold model or OOF prediction array is kept.
        del child
    final = clone(model).set_params(cv=None)
    if plan.groups is not None:
        final._cv_inner_split = (plan.final_fit, plan.final_stop)
    # Ungrouped final fit deliberately uses the exact native holdout path.
    final.fit(plan.X, plan.y, sample_weight=plan.weight)
    final_seconds = float(final.fit_time_)
    for name in sums:
        sums[name] += float(getattr(final, name))
    cv_value = model.cv
    model.__dict__.clear()
    model.__dict__.update(final.__dict__)
    model.cv = cv_value
    model.n_ignored_samples_ = plan.original_rows - len(plan.X)
    model.n_effective_samples_ = len(plan.X)
    scores = np.array([row["score"] for row in results], dtype=float)
    model.cv_results_ = {
        "n_splits": int(cv_value),
        "metric": metric,
        "direction": "maximize"
        if metric in {"accuracy", "roc_auc", "r2"}
        else "minimize",
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "folds": tuple(results),
        "split_sha256": plan.signature,
        "scope": "TRAIN_only_OOF_diagnostic_not_independent_confirmation",
        "groups": "declared"
        if plan.groups is not None
        else "row_level_no_group_guarantee",
        "training_passes": (int(cv_value) + 1) * (2 if model.full_data_refit else 1),
        "final_models": 1,
        "plan_index_bytes": sum(
            part.nbytes
            for fold in plan.folds
            for part in (fold.train, fold.validation, fold.fit, fold.stop)
        )
        + plan.final_fit.nbytes
        + plan.final_stop.nbytes,
        "selects_recipe_or_checkpoint": False,
    }
    model.profile_["internal_cv"] = {
        "planning_seconds": planned - started,
        "folds": tuple(results),
        "final_fit_seconds": final_seconds,
        "training_passes": model.cv_results_["training_passes"],
    }
    for name, value in sums.items():
        setattr(model, name, value)
    model.fit_time_ = perf_counter() - started

"""Shared estimator fit mechanics; task semantics live in the public estimators."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from copy import copy
from numbers import Integral, Real
from time import perf_counter
from typing import Hashable

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .device import resolve_device
from .network import TabularNetwork
from .preprocessing import TabularPreprocessor
from .tasks import BINARY_TASK
from .training import (
    predict_outputs,
    resolve_batch_size,
    train_fixed_epochs,
    train_model,
)


class _BaseNeuroTabularEstimator(BaseEstimator):
    """Shared validation, preprocessing, transactional fit and device mechanics."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 3e-3,
        weight_decay: float = 1e-5,
        batch_size: int | str = "auto",
        max_epochs: int = 30,
        validation_fraction: float = 0.2,
        patience: int = 4,
        min_delta: float = 1e-4,
        eval_frequency: int = 1,
        eval_metric: str = "loss",
        class_weight: str | None = None,
        categorical_features: Sequence[Hashable] | None = None,
        min_category_count: int = 2,
        numerical_embedding: str = "scalar",
        use_category_frequency: bool = True,
        feature_gating: bool = False,
        full_data_refit: bool = False,
        device: str = "auto",
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.eval_frequency = eval_frequency
        self.eval_metric = eval_metric
        self.class_weight = class_weight
        self.categorical_features = categorical_features
        self.min_category_count = min_category_count
        self.numerical_embedding = numerical_embedding
        self.use_category_frequency = use_category_frequency
        self.feature_gating = feature_gating
        self.full_data_refit = full_data_refit
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: object,
        sample_weight: object | None = None,
        eval_set: tuple[pd.DataFrame, object] | None = None,
    ) -> _BaseNeuroTabularEstimator:
        """Fit the estimator transactionally and restore caller RNG state.

        When ``eval_set`` is omitted, fitting creates a stratified internal
        validation split. Preprocessing is learned from training rows only. Rows
        with zero ``sample_weight`` are ignored completely. A failed re-fit leaves
        an already fitted estimator usable.
        """

        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        cuda_random_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        candidate = copy(self)
        candidate._clear_fitted_state()
        try:
            candidate._fit_impl(X, y, sample_weight=sample_weight, eval_set=eval_set)
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)
            torch.random.set_rng_state(torch_random_state)
            if cuda_random_states is not None:
                torch.cuda.set_rng_state_all(cuda_random_states)
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
        return self

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: object,
        sample_weight: object | None = None,
        eval_set: tuple[pd.DataFrame, object] | None = None,
    ) -> None:
        """Build a complete fitted candidate before :meth:`fit` commits it."""

        fit_started = perf_counter()
        self._validate_hyperparameters()
        X = self._validate_X(X)
        original_sample_count = len(X)
        raw_target = self._target_array(y, original_sample_count, allow_missing=True)
        all_sample_weight = self._validate_sample_weight(
            sample_weight, original_sample_count
        )
        if all_sample_weight is not None:
            positive_weight = all_sample_weight > 0.0
            X = X.iloc[np.flatnonzero(positive_weight)]
            raw_target = raw_target[positive_weight]
            all_sample_weight = all_sample_weight[positive_weight]
        y_array = self._validate_target(raw_target, len(X))
        self.n_effective_samples_ = len(X)
        self.n_ignored_samples_ = original_sample_count - len(X)
        device = self._resolve_device()
        self.device_ = str(device)
        self._set_random_state(device)

        if eval_set is None:
            indices = np.arange(len(X))
            try:
                train_indices, validation_indices = train_test_split(
                    indices,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=None if self._task_.name == "regression" else y_array,
                )
            except ValueError as exc:
                if self._task_.name == "regression":
                    raise ValueError(
                        "Unable to create a validation split; provide more rows or adjust validation_fraction."
                    ) from exc
                raise ValueError(
                    "Unable to create a stratified validation split. Provide more "
                    "samples per class or adjust validation_fraction."
                ) from exc
            X_train = X.iloc[train_indices]
            X_validation = X.iloc[validation_indices]
            y_train = y_array[train_indices]
            y_validation = y_array[validation_indices]
            train_sample_weight = (
                None if all_sample_weight is None else all_sample_weight[train_indices]
            )
            validation_weight = (
                np.ones(len(validation_indices), dtype=np.float64)
                if all_sample_weight is None
                else all_sample_weight[validation_indices]
            )
        else:
            X_validation, raw_validation_target = self._validate_eval_set(eval_set)
            y_validation = self._validate_validation_target(
                raw_validation_target, len(X_validation)
            )
            X_train = X
            y_train = y_array
            train_sample_weight = all_sample_weight
            validation_weight = np.ones(len(X_validation), dtype=np.float64)
        if (
            self._task_.name == "regression"
            and self.eval_metric == "r2"
            and len(y_validation) < 2
        ):
            raise ValueError("eval_metric='r2' requires at least two validation rows.")
        if self.eval_metric == "roc_auc" and np.unique(y_validation).size < 2:
            raise ValueError(
                "eval_metric='roc_auc' requires both target classes in validation."
            )
        if not np.any(validation_weight > 0.0):
            raise ValueError("Validation must contain at least one positive weight.")

        preprocessing_started = perf_counter()
        self._preprocessor_ = TabularPreprocessor(
            categorical_features=self.categorical_features,
            min_category_count=self.min_category_count,
            use_category_frequency=self.use_category_frequency,
            max_categories=getattr(self, "_experimental_max_categories", None),
            hash_buckets=int(getattr(self, "_experimental_hash_buckets", 0)),
        )
        self._preprocessor_.fit(X_train)
        fit_preprocessing_profile = dict(self._preprocessor_.fit_profile_)
        train_data = self._preprocessor_.transform(X_train)
        train_transform_profile = dict(self._preprocessor_.last_transform_profile_)
        validation_data = self._preprocessor_.transform(X_validation)
        validation_transform_profile = dict(self._preprocessor_.last_transform_profile_)
        self.preprocessing_time_ = perf_counter() - preprocessing_started

        self.n_features_in_ = X.shape[1]
        if all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.numeric_features_ = list(self._preprocessor_.numeric_features_)
        self.categorical_features_ = list(self._preprocessor_.categorical_features_)

        target_started = perf_counter()
        self._prepare_target(y_train, train_sample_weight)
        y_train_encoded = self._encode_target(y_train)
        y_validation_encoded = self._encode_target(y_validation)
        if self.eval_metric == "roc_auc" and any(
            not np.any(validation_weight[y_validation_encoded == class_id] > 0.0)
            for class_id in (0.0, 1.0)
        ):
            raise ValueError(
                "eval_metric='roc_auc' requires positive validation weight for "
                "both target classes."
            )
        validation_weight = self._normalize_weight(
            validation_weight, name="validation sample weight"
        )
        train_weight = self._combined_training_weight(
            y_train_encoded, train_sample_weight
        )
        target_preparation_time = perf_counter() - target_started
        self._model_ = self._new_model(self._preprocessor_)
        self._set_prior_bias(self._model_, y_train_encoded, train_weight)
        self.n_parameters_ = self._model_.parameter_count
        self.embedding_dimensions_ = list(self._model_.embedding_dimensions)
        self.batch_size_ = resolve_batch_size(
            self.batch_size,
            n_samples=len(X_train),
            n_numeric_inputs=self._model_.input_width,
            categorical_cardinalities=(self._preprocessor_.categorical_cardinalities_),
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            device=device,
        )
        self.inference_batch_size_ = max(1_024, self.batch_size_)

        training_result = train_model(
            self._model_,
            train_data,
            y_train_encoded,
            train_weight,
            validation_data,
            y_validation_encoded,
            validation_weight,
            device=device,
            batch_size=self.batch_size_,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_delta=float(self.min_delta),
            eval_frequency=self.eval_frequency,
            eval_metric=self.eval_metric,
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            random_state=self.random_state,
            verbose=self.verbose,
            use_amp=bool(self.device_info_["amp_enabled"]),
            task=self._task_,
        )
        self.best_epoch_ = training_result.best_epoch
        self.best_score_ = training_result.best_score
        self.best_validation_loss_ = training_result.best_validation_loss
        self.n_iter_ = training_result.n_iter
        self.history_ = training_result.history
        self.training_time_ = float(training_result.profile["training_compute_seconds"])
        self.validation_time_ = float(training_result.profile["validation_seconds"])
        refit_profile = None
        self.full_data_refit_ = False
        if self.full_data_refit and eval_set is None:
            refit_started = perf_counter()
            refit_preprocessor = TabularPreprocessor(
                categorical_features=self.categorical_features,
                min_category_count=self.min_category_count,
                use_category_frequency=self.use_category_frequency,
                max_categories=getattr(self, "_experimental_max_categories", None),
                hash_buckets=int(getattr(self, "_experimental_hash_buckets", 0)),
            ).fit(X)
            refit_fit_profile = dict(refit_preprocessor.fit_profile_)
            refit_data = refit_preprocessor.transform(X)
            refit_transform_profile = dict(refit_preprocessor.last_transform_profile_)
            self._prepare_target(y_array, all_sample_weight)
            full_target = self._encode_target(y_array)
            full_weight = self._combined_training_weight(full_target, all_sample_weight)
            self._set_random_state(device)
            refit_model = self._new_model(refit_preprocessor)
            self._set_prior_bias(refit_model, full_target, full_weight)
            refit_batch_size = resolve_batch_size(
                self.batch_size,
                n_samples=len(X),
                n_numeric_inputs=refit_model.input_width,
                categorical_cardinalities=(
                    refit_preprocessor.categorical_cardinalities_
                ),
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                device=device,
            )
            refit_training_profile = train_fixed_epochs(
                refit_model,
                refit_data,
                full_target,
                full_weight,
                device=device,
                batch_size=refit_batch_size,
                epochs=self.best_epoch_,
                lr=float(self.lr),
                weight_decay=float(self.weight_decay),
                random_state=self.random_state,
                use_amp=bool(self.device_info_["amp_enabled"]),
                task=self._task_,
            )
            self._preprocessor_ = refit_preprocessor
            self._model_ = refit_model
            self.batch_size_ = refit_batch_size
            self.inference_batch_size_ = max(1_024, refit_batch_size)
            self.n_parameters_ = refit_model.parameter_count
            self.embedding_dimensions_ = list(refit_model.embedding_dimensions)
            self.full_data_refit_ = True
            self.training_time_ += float(refit_training_profile["engine_total_seconds"])
            refit_profile = {
                "preprocessing_fit": refit_fit_profile,
                "transform": refit_transform_profile,
                "training": refit_training_profile,
                "total_seconds": perf_counter() - refit_started,
            }
        self.profile_ = {
            "preprocessing": {
                "fit": fit_preprocessing_profile,
                "train_transform": train_transform_profile,
                "validation_transform": validation_transform_profile,
                "total_seconds": self.preprocessing_time_,
            },
            "training": training_result.profile,
            "target_preparation_seconds": target_preparation_time,
            "device": dict(self.device_info_),
        }
        if refit_profile is not None:
            self.profile_["full_data_refit"] = refit_profile
        self.fit_time_ = perf_counter() - fit_started

    def _predict_values(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_model_", "_preprocessor_"])
        prediction_started = perf_counter()
        X = self._validate_X(X, fitting=False)
        data = self._preprocessor_.transform(X)
        probabilities = predict_outputs(
            self._model_,
            data,
            device=torch.device(self.device_),
            batch_size=self.inference_batch_size_,
            task=getattr(self, "_task_", BINARY_TASK),
        )
        self.last_prediction_time_ = perf_counter() - prediction_started
        return probabilities

    def _new_model(self, preprocessor: TabularPreprocessor) -> TabularNetwork:
        return TabularNetwork(
            n_numeric_features=preprocessor.n_numeric_outputs_,
            categorical_cardinalities=preprocessor.categorical_cardinalities_,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            n_continuous_features=preprocessor.n_continuous_features_,
            numerical_knots=torch.from_numpy(preprocessor.numeric_knots_),
            numerical_embedding=self.numerical_embedding,
            dataset_size=preprocessor.fit_sample_count_,
            feature_gating=self.feature_gating,
            categorical_dropout=float(
                getattr(self, "_experimental_categorical_dropout", 0.0)
            ),
            embedding_dropout=float(
                getattr(self, "_experimental_embedding_dropout", 0.0)
            ),
            output_dim=self._task_.output_dim,
        )

    @staticmethod
    def _target_array(
        y: object,
        n_samples: int,
        *,
        name: str = "y",
        allow_missing: bool = False,
    ) -> np.ndarray:
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if len(y_array) != n_samples:
            raise ValueError(f"X and {name} contain different numbers of samples.")
        if not allow_missing and pd.isna(y_array).any():
            raise ValueError(f"{name} must not contain missing values.")
        return y_array

    @staticmethod
    def _normalize_weight(weight: np.ndarray, *, name: str) -> np.ndarray:
        """Scale weights to mean one without overflowing their float32 tensor."""

        values = np.asarray(weight, dtype=np.float64)
        positive = values > 0.0
        if not np.any(positive):
            raise ValueError(f"{name} must contain at least one positive value.")
        maximum = float(values[positive].max())
        scaled = values / maximum
        if np.any(positive & (scaled == 0.0)):
            raise ValueError(f"{name} has a dynamic range too large for training.")
        normalized = scaled / float(np.mean(scaled, dtype=np.float64))
        float32 = np.finfo(np.float32)
        positive_normalized = normalized[positive]
        if (
            not np.isfinite(normalized).all()
            or float(normalized.max()) > float32.max
            or float(positive_normalized.min()) < float32.tiny
        ):
            raise ValueError(f"{name} cannot be represented safely in float32.")
        return normalized

    @staticmethod
    def _validate_sample_weight(
        sample_weight: object | None, n_samples: int
    ) -> np.ndarray | None:
        if sample_weight is None:
            return None
        try:
            weights = np.asarray(sample_weight, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight must contain numeric values.") from exc
        if weights.ndim != 1:
            raise ValueError("sample_weight must be one-dimensional.")
        if len(weights) != n_samples:
            raise ValueError(
                "X and sample_weight contain different numbers of samples."
            )
        if not np.isfinite(weights).all():
            raise ValueError("sample_weight must contain only finite values.")
        if np.any(weights < 0.0):
            raise ValueError("sample_weight values must be non-negative.")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight must contain at least one positive value.")
        return weights

    def _validate_eval_set(self, eval_set: object) -> tuple[pd.DataFrame, object]:
        if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
            raise TypeError("eval_set must be a single (X_valid, y_valid) pair.")
        X_validation = self._validate_X(eval_set[0], fitting=False)
        return X_validation, eval_set[1]

    @staticmethod
    def _validate_X(X: pd.DataFrame, *, fitting: bool = True) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")
        if len(X) == 0:
            raise ValueError("X must contain at least one row.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature column.")
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")
        return X

    def _validate_hyperparameters(self) -> None:
        self._positive_integer(self.hidden_dim, "hidden_dim")
        self._positive_integer(self.n_blocks, "n_blocks")
        self._unit_interval(self.dropout, "dropout", upper_inclusive=False)
        self._positive_real(self.lr, "lr")
        self._non_negative_real(self.weight_decay, "weight_decay")
        if self.batch_size != "auto":
            self._positive_integer(self.batch_size, "batch_size")
        self._positive_integer(self.max_epochs, "max_epochs")
        self._unit_interval(
            self.validation_fraction,
            "validation_fraction",
            lower_inclusive=False,
            upper_inclusive=False,
        )
        self._positive_integer(self.patience, "patience")
        self._non_negative_real(self.min_delta, "min_delta")
        self._positive_integer(self.eval_frequency, "eval_frequency")
        self._positive_integer(self.min_category_count, "min_category_count")
        if self.numerical_embedding not in {
            "scalar",
            "affine",
            "periodic",
            "piecewise",
        }:
            raise ValueError(
                "numerical_embedding must be 'scalar', 'affine', 'periodic', "
                "or 'piecewise'."
            )
        if not isinstance(self.use_category_frequency, bool):
            raise TypeError("use_category_frequency must be a boolean.")
        if not isinstance(self.feature_gating, bool):
            raise TypeError("feature_gating must be a boolean.")
        if not isinstance(self.full_data_refit, bool):
            raise TypeError("full_data_refit must be a boolean.")
        if self.eval_metric not in self._supported_metrics:
            raise ValueError(
                f"eval_metric must be one of {sorted(self._supported_metrics)!r}."
            )
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")
        if not isinstance(self.random_state, Integral) or isinstance(
            self.random_state, bool
        ):
            raise TypeError("random_state must be an integer.")
        if self.verbose not in {0, 1}:
            raise ValueError("verbose must be 0 or 1.")
        if not isinstance(self.device, str):
            raise TypeError("device must be 'auto', 'cpu', or a CUDA device string.")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string.")

    def _resolve_device(self) -> torch.device:
        device, info = resolve_device(self.device)
        self.device_info_ = info
        return device

    def _clear_fitted_state(self) -> None:
        for name in tuple(self.__dict__):
            if name.endswith("_"):
                del self.__dict__[name]

    def _set_random_state(self, device: torch.device) -> None:
        seed = int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _positive_integer(value: object, name: str) -> None:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _positive_real(value: object, name: str) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def _non_negative_real(value: object, name: str) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number.")

    @staticmethod
    def _unit_interval(
        value: object,
        name: str,
        *,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
    ) -> None:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number in the unit interval.")
        lower_ok = value >= 0.0 if lower_inclusive else value > 0.0
        upper_ok = value <= 1.0 if upper_inclusive else value < 1.0
        if not math.isfinite(float(value)) or not lower_ok or not upper_ok:
            raise ValueError(f"{name} must be in the required unit interval.")

    def _prepare_target(self, y: np.ndarray, weight: np.ndarray | None) -> None:
        """Optional train-only target statistics hook."""

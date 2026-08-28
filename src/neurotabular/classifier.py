"""Public scikit-learn-compatible binary classifier."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from numbers import Integral, Real
from time import perf_counter
from typing import Hashable

import numpy as np
import pandas as pd
import torch
from pandas.errors import InvalidIndexError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .device import resolve_device
from .network import TabularNetwork
from .preprocessing import TabularPreprocessor
from .training import (
    predict_probabilities,
    refit_binary_model,
    resolve_batch_size,
    train_binary_model,
)


class NeuroTabularClassifier(ClassifierMixin, BaseEstimator):
    """A compact neural binary classifier for pandas DataFrames.

    NeuroTabular handles numerical missing values, categorical detection,
    categorical embeddings, validation, batching, and early stopping without
    requiring a separate preprocessing pipeline.

    Parameters
    ----------
    hidden_dim : int, default=64
        Width of the projected representation and residual blocks.
    n_blocks : int, default=2
        Number of residual feed-forward blocks.
    dropout : float, default=0.1
        Dropout probability inside residual blocks.
    lr : float, default=0.003
        Initial AdamW learning rate for the cosine schedule.
    weight_decay : float, default=1e-5
        AdamW weight decay.
    batch_size : {"auto"} or int, default="auto"
        Deterministic automatic batching or an explicit positive size.
    max_epochs : int, default=30
        Maximum number of training epochs.
    validation_fraction : float, default=0.2
        Fraction used by the internal stratified validation split.
    patience : int, default=4
        Consecutive validation checks without a significant improvement.
    min_delta : float, default=1e-4
        Minimum absolute validation improvement that resets patience.
    eval_frequency : int, default=1
        Validate every N epochs, plus the first and final epochs.
    eval_metric : {"loss", "roc_auc", "accuracy"}, default="loss"
        Metric used for early stopping and best-weight selection.
    class_weight : {None, "balanced"}, default=None
        Optional inverse-frequency training class weights.
    categorical_features : sequence of hashable or None, default=None
        Columns forced to categorical in addition to automatic detection.
    min_category_count : int, default=2
        Training frequency below which a category uses the rare bucket.
    numerical_embedding : {"scalar", "affine", "periodic", "piecewise"}, \
            default="scalar"
        Leakage-safe representation used for numerical features.
    use_category_frequency : bool, default=True
        Add a training-only log-frequency side feature for each categorical column.
    feature_gating : bool, default=False
        Apply a lightweight gate to the initial cross-feature projection.
    full_data_refit : bool, default=False
        After internal early stopping, optionally retrain a fresh model on all
        rows for ``best_epoch_`` epochs. External validation already trains on
        all supplied training rows, so refit is skipped in that case.
    device : str, default="auto"
        ``"auto"``, ``"cpu"``, ``"cuda"``, or a CUDA device string. Automatic
        CUDA requires a successful synchronized compatibility probe; explicit
        CUDA requests raise a diagnostic error instead of falling back.
    random_state : int, default=42
        Seed for Python, NumPy, PyTorch, splitting, and batch shuffling.
    verbose : {0, 1}, default=0
        Whether to print validation progress and the selected epoch.
    """

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
    ) -> NeuroTabularClassifier:
        """Fit the binary neural classifier and restore its best weights.

        When ``eval_set`` is omitted, fitting creates a stratified internal
        validation split. Preprocessing state is always learned from training
        rows only.
        """

        fit_started = perf_counter()
        self._validate_hyperparameters()
        X = self._validate_X(X)
        y_array = self._validate_target(y, len(X))
        all_sample_weight = self._validate_sample_weight(sample_weight, len(X))
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
                    stratify=y_array,
                )
            except ValueError as exc:
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

        training_result = train_binary_model(
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
            refit_training_profile = refit_binary_model(
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
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities with columns ordered by ``classes_``."""

        probabilities = self._positive_class_probability(X)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return original class labels using a 0.5 probability threshold."""

        probabilities = self._positive_class_probability(X)
        return self.classes_[(probabilities >= 0.5).astype(np.int64)]

    def _positive_class_probability(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_model_", "_preprocessor_", "classes_"])
        prediction_started = perf_counter()
        X = self._validate_X(X, fitting=False)
        data = self._preprocessor_.transform(X)
        probabilities = predict_probabilities(
            self._model_,
            data,
            device=torch.device(self.device_),
            batch_size=self.inference_batch_size_,
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
        )

    @staticmethod
    def _set_prior_bias(
        model: TabularNetwork, target: np.ndarray, weight: np.ndarray
    ) -> None:
        positive_weight = float(weight[target == 1.0].sum())
        negative_weight = float(weight[target == 0.0].sum())
        if positive_weight > 0.0 and negative_weight > 0.0:
            model.set_output_bias(math.log(positive_weight / negative_weight))

    def _validate_target(self, y: object, n_samples: int) -> np.ndarray:
        y_array = self._target_array(y, n_samples)
        try:
            classes = np.unique(y_array)
        except TypeError as exc:
            raise ValueError("y classes must be mutually comparable.") from exc
        if len(classes) != 2:
            raise ValueError(
                "NeuroTabularClassifier currently supports binary targets only; "
                f"received {len(classes)} classes."
            )
        self.classes_ = classes
        return y_array

    def _validate_validation_target(self, y: object, n_samples: int) -> np.ndarray:
        y_array = self._target_array(y, n_samples, name="eval_set y")
        try:
            positions = pd.Index(self.classes_).get_indexer(y_array)
        except (InvalidIndexError, TypeError):
            positions = np.full(len(y_array), -1, dtype=np.int64)
        if np.any(positions < 0):
            unknown = pd.unique(y_array[positions < 0]).tolist()
            raise ValueError(
                f"eval_set y contains classes not present in training: {unknown!r}."
            )
        return y_array

    @staticmethod
    def _target_array(y: object, n_samples: int, *, name: str = "y") -> np.ndarray:
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if len(y_array) != n_samples:
            raise ValueError(f"X and {name} contain different numbers of samples.")
        if pd.isna(y_array).any():
            raise ValueError(f"{name} must not contain missing values.")
        return y_array

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        return (y == self.classes_[1]).astype(np.float32)

    def _combined_training_weight(
        self, y_encoded: np.ndarray, sample_weight: np.ndarray | None
    ) -> np.ndarray:
        if self.class_weight is None:
            self.class_weight_ = None
            combined = np.ones(len(y_encoded), dtype=np.float64)
        else:
            counts = np.bincount(y_encoded.astype(np.int64), minlength=2)
            if np.any(counts == 0):
                raise ValueError(
                    "class_weight='balanced' requires both training classes."
                )
            values = len(y_encoded) / (2.0 * counts.astype(np.float64))
            self.class_weight_ = {
                self.classes_[0]: float(values[0]),
                self.classes_[1]: float(values[1]),
            }
            combined = values[y_encoded.astype(np.int64)]
        if sample_weight is not None:
            combined = combined * sample_weight
        if not np.any(combined > 0.0):
            raise ValueError(
                "The training subset must contain a positive combined weight."
            )
        return combined

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
        if self.eval_metric not in {"loss", "roc_auc", "accuracy"}:
            raise ValueError("eval_metric must be 'loss', 'roc_auc', or 'accuracy'.")
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

"""Single-output regression on the shared tabular training engine."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from ._base import _BaseNeuroTabularEstimator
from .network import TabularNetwork
from .tasks import TaskSpec


class NeuroTabularRegressor(RegressorMixin, _BaseNeuroTabularEstimator):
    """Neural regression with train-only weighted target standardization.

    Shares the classifier's architecture and optimization parameters, except
    ``class_weight``. ``eval_metric`` is loss (standardized MSE), rmse, mae or r2;
    the first three are minimized and r2 is maximized. History metrics use the
    standardized target scale. ``predict`` returns original target units and
    sklearn ``score`` returns R². Internal validation is a shuffled random split.
    A constant training target produces that exact constant at prediction time.
    """

    _supported_metrics = {"loss", "rmse", "mae", "r2"}

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
        super().__init__(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            validation_fraction=validation_fraction,
            patience=patience,
            min_delta=min_delta,
            eval_frequency=eval_frequency,
            eval_metric=eval_metric,
            categorical_features=categorical_features,
            min_category_count=min_category_count,
            numerical_embedding=numerical_embedding,
            use_category_frequency=use_category_frequency,
            feature_gating=feature_gating,
            full_data_refit=full_data_refit,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

    @staticmethod
    def _numeric_target(y: object, n_samples: int, name: str = "y") -> np.ndarray:
        values = _BaseNeuroTabularEstimator._target_array(y, n_samples, name=name)
        numeric_object = values.dtype.kind == "O" and all(
            isinstance(v, Real) and not isinstance(v, bool) for v in values
        )
        if np.iscomplexobj(values) or (
            values.dtype.kind not in "iuf" and not numeric_object
        ):
            raise ValueError(
                f"{name} must contain real numeric values (no complex targets)."
            )
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def _validate_target(self, y: object, n_samples: int) -> np.ndarray:
        self._task_ = TaskSpec("regression", 1)
        return self._numeric_target(y, n_samples)

    def _validate_validation_target(self, y: object, n_samples: int) -> np.ndarray:
        values = self._numeric_target(y, n_samples, "eval_set y")
        if self.eval_metric == "r2" and n_samples < 2:
            raise ValueError("eval_metric='r2' requires at least two validation rows.")
        return values

    def _prepare_target(self, y: np.ndarray, weight: np.ndarray | None) -> None:
        # Computing moments on [-1, 1] avoids squaring huge raw targets and
        # handles very small scales without an arbitrary variance threshold.
        self._target_magnitude_ = max(float(np.abs(y).max()), np.finfo(float).tiny)
        scaled = y / self._target_magnitude_
        w = (
            None
            if weight is None
            else self._normalize_weight(weight, name="target sample weight")
        )
        self.target_is_constant_ = bool(np.all(y == y[0]))
        self._target_center_ = float(np.average(scaled, weights=w))
        self._target_spread_ = float(
            np.sqrt(np.average((scaled - self._target_center_) ** 2, weights=w))
        )
        self.target_mean_ = self._target_center_ * self._target_magnitude_
        self.target_scale_ = self._target_spread_ * self._target_magnitude_
        if self.target_is_constant_:
            self.target_mean_ = float(y[0])
            self.target_scale_ = 1.0
        elif self._target_spread_ == 0 or self.target_scale_ == 0:
            raise ValueError("Target variation cannot be represented safely.")

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        if self.target_is_constant_:
            values = y - self.target_mean_
        else:
            values = (
                y / self._target_magnitude_ - self._target_center_
            ) / self._target_spread_
        if not np.isfinite(values).all() or np.any(
            np.abs(values) > np.sqrt(np.finfo(np.float32).max) / 4
        ):
            raise ValueError(
                "Standardized target cannot be represented safely for float32 MSE."
            )
        return values.astype(np.float32)

    def _combined_training_weight(
        self, y_encoded: np.ndarray, sample_weight: np.ndarray | None
    ) -> np.ndarray:
        values = np.ones(len(y_encoded)) if sample_weight is None else sample_weight
        return self._normalize_weight(values, name="training sample weight")

    @staticmethod
    def _set_prior_bias(
        model: TabularNetwork, target: np.ndarray, weight: np.ndarray
    ) -> None:
        model.set_output_bias(0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return a finite one-dimensional array in original target units."""
        values = self._predict_values(X)
        if self.target_is_constant_:
            return np.full(len(values), self.target_mean_, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = (
                values * self._target_spread_ + self._target_center_
            ) * self._target_magnitude_
        if not np.isfinite(prediction).all():
            raise RuntimeError(
                "Inverse target transformation produced non-finite predictions."
            )
        return prediction

"""Public scikit-learn-compatible binary and multiclass classifier."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from pandas.errors import InvalidIndexError
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import type_of_target

from ._base import _BaseNeuroTabularEstimator
from .network import TabularNetwork
from .tasks import BINARY_TASK, TaskSpec


class NeuroTabularClassifier(ClassifierMixin, _BaseNeuroTabularEstimator):
    """A compact neural binary and multiclass classifier for pandas DataFrames.

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

    _supported_metrics = {"loss", "roc_auc", "accuracy"}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities with columns ordered by ``classes_``."""

        probabilities = self._predict_values(X)
        if len(self.classes_) > 2:
            # Restore the simplex at the returned float64 precision. Float32
            # softmax sums can otherwise trigger sklearn's log-loss warning.
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return original labels: binary threshold 0.5, multiclass argmax."""

        probabilities = self._predict_values(X)
        if len(self.classes_) > 2:
            return self.classes_[probabilities.argmax(axis=1)]
        return self.classes_[(probabilities >= 0.5).astype(np.int64)]

    @staticmethod
    def _set_prior_bias(
        model: TabularNetwork, target: np.ndarray, weight: np.ndarray
    ) -> None:
        positive_weight = float(weight[target == 1.0].sum())
        if model.output.out_features > 1:
            counts = np.bincount(
                target.astype(np.int64),
                weights=weight,
                minlength=model.output.out_features,
            )
            if np.any(counts <= 0):
                raise ValueError(
                    "Every training class needs positive effective weight."
                )
            with torch.no_grad():
                model.output.bias.copy_(
                    torch.as_tensor(
                        np.log(counts / counts.sum()), dtype=model.output.bias.dtype
                    )
                )
            return
        negative_weight = float(weight[target == 0.0].sum())
        if positive_weight > 0.0 and negative_weight > 0.0:
            model.set_output_bias(math.log(positive_weight / negative_weight))

    def _validate_target(self, y: object, n_samples: int) -> np.ndarray:
        y_array = self._target_array(y, n_samples)
        try:
            classes = np.unique(y_array)
        except TypeError as exc:
            raise ValueError("y classes must be mutually comparable.") from exc
        if len(classes) < 2:
            raise ValueError(
                "NeuroTabularClassifier requires at least two target classes; "
                f"received {len(classes)} classes."
            )
        if len(classes) > 2:
            if type_of_target(y_array) != "multiclass":
                raise ValueError("Classifier y must contain discrete class labels.")
            if self.eval_metric == "roc_auc":
                raise ValueError(
                    "eval_metric='roc_auc' is binary-only; use 'loss' or 'accuracy' for multiclass."
                )
        self._task_ = (
            BINARY_TASK if len(classes) == 2 else TaskSpec("multiclass", len(classes))
        )
        self.classes_ = classes
        self.n_classes_ = len(classes)
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

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        if len(self.classes_) > 2:
            return np.searchsorted(self.classes_, y).astype(np.int64)
        return (y == self.classes_[1]).astype(np.float32)

    def _combined_training_weight(
        self, y_encoded: np.ndarray, sample_weight: np.ndarray | None
    ) -> np.ndarray:
        if self.class_weight is None:
            self.class_weight_ = None
            combined = np.ones(len(y_encoded), dtype=np.float64)
        else:
            counts = np.bincount(
                y_encoded.astype(np.int64), minlength=len(self.classes_)
            )
            if np.any(counts == 0):
                raise ValueError(
                    "class_weight='balanced' requires all training classes."
                )
            values = len(y_encoded) / (
                float(len(self.classes_)) * counts.astype(np.float64)
            )
            self.class_weight_ = {
                label: float(value)
                for label, value in zip(self.classes_, values, strict=True)
            }
            combined = values[y_encoded.astype(np.int64)]
        if sample_weight is not None:
            combined = combined * sample_weight
        if not np.any(combined > 0.0):
            raise ValueError(
                "The training subset must contain a positive combined weight."
            )
        return self._normalize_weight(combined, name="combined training weight")

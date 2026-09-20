"""Small internal task boundary shared by training and prediction."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn


@dataclass(frozen=True)
class TaskSpec:
    """Loss, prediction transform and metric semantics; not a public API."""

    name: str = "binary"
    output_dim: int = 1

    def criterion(self) -> nn.Module:
        if self.name == "regression":
            return nn.MSELoss(reduction="none")
        if self.name == "multiclass":
            return nn.CrossEntropyLoss(reduction="none")
        return nn.BCEWithLogitsLoss(reduction="none")

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        if self.name == "regression":
            return logits
        if self.name == "multiclass":
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def score(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        weight: np.ndarray,
        metric: str,
    ) -> float:
        if self.name == "regression":
            if metric == "rmse":
                return float(
                    np.sqrt(
                        mean_squared_error(target, prediction, sample_weight=weight)
                    )
                )
            if metric == "mae":
                return float(
                    mean_absolute_error(target, prediction, sample_weight=weight)
                )
            return float(r2_score(target, prediction, sample_weight=weight))
        if metric == "roc_auc":
            return float(roc_auc_score(target, prediction, sample_weight=weight))
        labels = (
            prediction.argmax(1) if self.name == "multiclass" else prediction >= 0.5
        )
        return float(accuracy_score(target, labels, sample_weight=weight))


BINARY_TASK = TaskSpec()

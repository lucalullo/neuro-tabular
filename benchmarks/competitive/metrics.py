"""Class-aligned scoring of fixed predictions; never fits a model or a threshold."""

from __future__ import annotations


def score(task, y, prediction, classes=None, native_labels=None):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y).reshape(-1)
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) != len(y):
        raise ValueError("Invalid or misaligned predictions")
    if task == "regression":
        values = values.reshape(-1)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, values))),
            "mae": float(mean_absolute_error(y, values)),
            "r2": float(r2_score(y, values)),
        }
    if task not in {"binary", "multiclass"}:
        raise ValueError("Unknown task")
    classes = np.asarray(classes)
    if (
        values.ndim != 2
        or values.shape[1] != len(classes)
        or len(set(classes.tolist())) != len(classes)
    ):
        raise ValueError("Probability columns must match the recorded class order")
    if (
        (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid probabilities")
    mapping = {value: index for index, value in enumerate(classes.tolist())}
    if not set(y.tolist()).issubset(mapping):
        raise ValueError("Evaluation contains a class absent from TRAIN")
    positions = np.asarray([mapping[value] for value in y])
    labels = (
        classes[values.argmax(1)]
        if native_labels is None
        else np.asarray(native_labels).reshape(-1)
    )
    if len(labels) != len(y) or not set(labels.tolist()).issubset(mapping):
        raise ValueError("Invalid native decision labels")
    # Clipping is local to the scorer; serialized probabilities are preserved.
    nll = -np.log(np.clip(values[np.arange(len(y)), positions], 1e-15, 1)).mean()
    auc, reason = None, None
    if task == "binary":
        if len(classes) != 2:
            raise ValueError("Binary classifier must have exactly two classes")
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(positions, values[:, 1]))
        else:
            reason = "Only one class present in evaluation"
    elif len(np.unique(y)) == len(classes):
        auc = float(
            roc_auc_score(
                positions,
                values,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(classes)),
            )
        )
    else:
        reason = "AUC OVR macro undefined: evaluation lacks a recorded class"
    result = {
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "macro_f1": float(
            f1_score(y, labels, labels=classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, labels, labels=classes, average="macro", zero_division=0)
        ),
        "nll": float(nll),
        "auc": auc,
        "auc_unavailable_reason": reason,
    }
    if len(classes) > 5:
        result["top5"] = float(
            (np.argsort(-values, axis=1, kind="stable")[:, :5] == positions[:, None])
            .any(1)
            .mean()
        )
    return result

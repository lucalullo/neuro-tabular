"""Minimal synthetic CUDA compatibility check for Kaggle or another GPU host."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from neurotabular import NeuroTabularClassifier


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
        print("Compiled architectures:", torch.cuda.get_arch_list())

    rng = np.random.default_rng(42)
    values = rng.normal(size=(2_000, 12))
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
    X["segment"] = rng.choice(["a", "b", "c", None], len(X))
    signal = values[:, 0] + 0.8 * values[:, 1] + 0.5 * (X["segment"] == "a")
    y = (signal + rng.normal(0.0, 0.8, len(X)) > 0.0).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = NeuroTabularClassifier(device="auto", random_state=42)
    model.fit(X_train, y_train)
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("Requested device:", model.device_info_["requested_device"])
    print("Resolved device:", model.device_)
    print("Device diagnostics:", model.device_info_)
    print("Batch size:", model.batch_size_)
    print("Epochs:", model.n_iter_)
    print("Fit seconds:", model.fit_time_)
    print("ROC-AUC:", score)


if __name__ == "__main__":
    main()

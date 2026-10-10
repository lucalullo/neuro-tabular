"""Three tasks with mixed data, unseen categories and optional trusted persistence.

Run: python examples/mixed_dataframe_workflow.py --task regression
The small synthetic fixture demonstrates the API, not predictive quality.
"""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from neurotabular import (
    NeuroTabularClassifier,
    NeuroTabularRegressor,
    __version__,
)


def model_metadata(model):
    """An optional user-owned sidecar; not an estimator/persistence API."""
    n_classes = getattr(model, "n_classes_", None)
    return {
        "neurotabular_version_at_export": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "fitted_task": (
            "regression"
            if isinstance(model, NeuroTabularRegressor)
            else "binary"
            if n_classes == 2
            else "multiclass"
        ),
        "fitted_device": model.device_,
        "feature_count": int(model.n_features_in_),
        "class_count": None if n_classes is None else int(n_classes),
        "best_epoch": int(model.best_epoch_),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("binary", "multiclass", "regression"), default="binary"
    )
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda or cuda:N")
    parser.add_argument(
        "--output-dir", type=Path, help="Optional local model and metadata directory"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    n = 120
    amount = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "amount": amount,
            "visits": pd.Series(np.arange(n) % 11, dtype="Int64"),
            "city": pd.Categorical(np.resize(["Rome", "Milan", None], n)),
            "active": pd.Series(np.resize([True, False, None], n), dtype="boolean"),
        }
    )
    X.loc[::13, "amount"] = np.nan
    X.loc[::17, "visits"] = pd.NA
    if args.task == "regression":
        y = 20 + 3 * amount + rng.normal(scale=0.1, size=n)
        cls = NeuroTabularRegressor
        stratify = None
    else:
        y = (
            np.where(amount > 0, "yes", "no")
            if args.task == "binary"
            else np.asarray(["low", "middle", "high"])[np.digitize(amount, [-0.4, 0.4])]
        )
        cls = NeuroTabularClassifier
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=stratify,
    )
    # A short software example, not a recommended tuning recipe.
    model = cls(max_epochs=3, hidden_dim=12, device=args.device, random_state=7)
    model.fit(X_train, y_train)
    prediction_frame = X_test.copy()
    prediction_frame["city"] = prediction_frame["city"].astype("string")
    prediction_frame.loc[prediction_frame.index[0], "city"] = "unseen-city"
    prediction_frame = prediction_frame.loc[:, list(reversed(X.columns))]
    prediction = model.predict(prediction_frame)
    print("Task:", args.task, "device:", model.device_)
    print("Prediction shape:", prediction.shape)
    print("Test score on original test inputs:", model.score(X_test, y_test))
    if args.task != "regression":
        print("Probability columns:", model.classes_)
        print("Probability shape:", model.predict_proba(prediction_frame).shape)
    # input_schema_ belongs to prepared P37 work; public 0.3.1 lacks it.
    if hasattr(model, "input_schema_"):
        print("Schema:", pd.DataFrame(model.input_schema_).to_string(index=False))
    metadata = model_metadata(model)
    print(json.dumps(metadata, indent=2))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.task}.joblib"
        joblib.dump(model, path)
        (args.output_dir / f"{args.task}.metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        # Load only trusted artifacts. This round-trip assumes the same runtime.
        restored = joblib.load(path)
        np.testing.assert_array_equal(prediction, restored.predict(prediction_frame))
        print("Same-runtime persistence: PASS")


if __name__ == "__main__":
    main()

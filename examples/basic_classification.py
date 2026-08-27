"""Minimal binary classification with mixed synthetic data."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurotabular import NeuroTabularClassifier

rng = np.random.default_rng(7)
n_samples = 300
age = rng.normal(42, 12, n_samples)
income = rng.lognormal(10.4, 0.45, n_samples)
city = rng.choice(["Rome", "Milan", "Naples"], n_samples).astype(object)
is_member = rng.choice([True, False], n_samples)

age[::19] = np.nan
city[::23] = None
X = pd.DataFrame({"age": age, "income": income, "city": city, "is_member": is_member})
signal = (
    np.nan_to_num(age, nan=42.0) / 12
    + 0.8 * (city == "Milan")
    + 0.5 * is_member
    + rng.normal(0, 0.5, n_samples)
)
y = (signal > np.median(signal)).astype(int)

split = 240
model = NeuroTabularClassifier(eval_metric="roc_auc", random_state=7)
model.fit(
    X.iloc[:split],
    y[:split],
    eval_set=(X.iloc[split:], y[split:]),
)

probability = model.predict_proba(X.iloc[split:])[:, 1]
print("ROC-AUC:", roc_auc_score(y[split:], probability))
print("Best epoch:", model.best_epoch_)
print("Fit seconds:", model.fit_time_)
print("Predictions:", model.predict(X.iloc[split : split + 5]))

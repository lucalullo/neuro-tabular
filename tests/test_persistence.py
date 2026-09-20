"""Serialization must survive a new interpreter, not just an in-process load."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.datasets import load_iris

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_fresh_process_persistence(task, format, tmp_path):
    X, y = load_iris(return_X_y=True, as_frame=True)
    if task == "binary":
        y = (y == 0).astype(int)
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    if task == "regression":
        y = X.iloc[:, 0].to_numpy() * 10 - 20
    model = cls(max_epochs=3, hidden_dim=12, device="cpu").fit(X, y)
    path = tmp_path / "model.bin"
    if format == "pickle":
        path.write_bytes(pickle.dumps(model))
    else:
        joblib.dump(model, path)
    X.to_pickle(tmp_path / "X.pkl")
    script = """import sys,pickle,joblib,numpy as np,pandas as pd,torch
from pathlib import Path
root=Path(sys.argv[1]); torch.set_num_threads(int(sys.argv[3]))
m=pickle.loads((root/'model.bin').read_bytes()) if sys.argv[2]=='pickle' else joblib.load(root/'model.bin')
X=pd.read_pickle(root/'X.pkl')
np.savez(root/'predictions.npz',labels=m.predict(X),probabilities=m.predict_proba(X) if hasattr(m,'predict_proba') else m.predict(X))
"""
    import torch

    source = Path(__file__).resolve().parents[1] / "src"
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            format,
            str(torch.get_num_threads()),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(tmp_path / "predictions.npz") as raw:
        assert np.array_equal(raw["labels"], model.predict(X))
        if task != "regression":
            assert np.array_equal(raw["probabilities"], model.predict_proba(X))

"""P39 delta: serialized nullable/rare/unseen schema and all fitted metadata.

Three synthetic software fits, each reused for both fresh-process formats.
Internal encoder inspection here is regression coverage, not a public API.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from examples.mixed_dataframe_workflow import model_metadata
from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor


def snapshot(model):
    fitted = {
        name: value
        for name, value in vars(model).items()
        if name.endswith("_") and not name.startswith("_")
    }
    # Timers may change during predict; compare the state saved at dump time.
    encoder = model._preprocessor_
    state = {
        name: getattr(encoder, name)
        for name in (
            "category_vocabs_",
            "rare_categories_",
            "category_id_frequencies_",
            "numeric_medians_",
            "numeric_centers_",
            "numeric_scales_",
        )
    }
    return {"fitted": fitted, "encoder": state, "metadata": model_metadata(model)}


def assert_same(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.fixture(scope="module", params=["binary", "multiclass", "regression"])
def fitted_case(request):
    n = 72
    X = pd.DataFrame(
        {
            "value": np.linspace(-1, 1, n),
            "count": pd.Series(np.arange(n) % 5, dtype="Int64"),
            "category": pd.Categorical(
                ["zero-weight-only", "rare"] + ["common"] * (n - 2),
                categories=["zero-weight-only", "rare", "common", "declared-only"],
            ),
            "text": pd.Series(np.resize(["a", "b", None], n), dtype="string"),
            "flag": pd.Series(np.resize([True, False, None], n), dtype="boolean"),
        }
    )
    X.loc[::7, "value"] = np.nan
    X.loc[::9, "count"] = pd.NA
    task = request.param
    y = (
        np.linspace(10, 30, n)
        if task == "regression"
        else np.asarray(["a", "b", "c"])[np.arange(n) % (2 if task == "binary" else 3)]
    )
    validation = X.iloc[6:24].copy()
    validation["category"] = "validation-only"
    weights = np.ones(n)
    weights[0] = 0
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    torch.set_num_threads(1)
    model = cls(
        hidden_dim=8, n_blocks=1, max_epochs=1, device="cpu", random_state=9
    ).fit(
        X,
        y,
        sample_weight=weights,
        eval_set=(validation, y[6:24]),
    )
    encoder = model._preprocessor_
    assert encoder.rare_categories_["category"] == {"rare"}
    known = (
        set(encoder.category_vocabs_["category"]) | encoder.rare_categories_["category"]
    )
    assert known == {"rare", "common"}
    probe = X.iloc[1:7].copy()
    probe["category"] = pd.Series(
        ["rare", "common", "brand-new", "declared-only", None, "validation-only"],
        index=probe.index,
        dtype="string",
    )
    probe = probe.loc[:, list(reversed(X.columns))]
    probe.loc[probe.index[0], "count"] = pd.NA
    before = probe.copy(deep=True)
    prediction = model.predict(probe)
    probability = model.predict_proba(probe) if task != "regression" else None
    pd.testing.assert_frame_equal(before, probe)
    with pytest.raises(ValueError) as error:
        model.predict(probe.drop(columns="count"))
    return model, probe, prediction, probability, str(error.value)


@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_fresh_process_edge_schema(fitted_case, format, tmp_path):
    model, probe, prediction, probability, schema_error = fitted_case
    expected = pickle.loads(pickle.dumps(snapshot(model)))
    if format == "pickle":
        (tmp_path / "model.bin").write_bytes(pickle.dumps(model))
    else:
        joblib.dump(model, tmp_path / "model.bin")
    probe.to_pickle(tmp_path / "probe.pkl")
    script = """
import pickle,sys,joblib,numpy as np,pandas as pd,torch
from pathlib import Path
from test_persistence_edge_schema import snapshot
torch.set_num_threads(1)
root=Path(sys.argv[1])
model=pickle.loads((root/'model.bin').read_bytes()) if sys.argv[2]=='pickle' else joblib.load(root/'model.bin')
(root/'snapshot.pkl').write_bytes(pickle.dumps(snapshot(model)))
probe=pd.read_pickle(root/'probe.pkl')
pred=model.predict(probe)
prob=model.predict_proba(probe) if hasattr(model,'predict_proba') else np.array([])
np.savez(root/'outputs.npz',prediction=pred,probability=prob)
try:
    model.predict(probe.drop(columns='count'))
except ValueError as exc:
    (root/'error.txt').write_text(str(exc),encoding='utf-8')
else:
    raise AssertionError('Missing schema exception after load')
"""
    root = Path(__file__).resolve().parents[1]
    # Fresh interpreter outside the checkout, with only explicit source roots.
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path), format],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(root / "src"), str(root), str(root / "tests")]
            ),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert_same(expected, pickle.loads((tmp_path / "snapshot.pkl").read_bytes()))
    with np.load(tmp_path / "outputs.npz") as outputs:
        np.testing.assert_array_equal(outputs["prediction"], prediction)
        if probability is not None:
            np.testing.assert_array_equal(outputs["probability"], probability)
    assert (tmp_path / "error.txt").read_text("utf-8") == schema_error

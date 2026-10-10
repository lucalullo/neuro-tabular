"""Fresh-process provenance for new models and trusted public-line fixtures."""

import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from p40_helpers import data, kwargs

from neurotabular import NeuroTabularClassifier, NeuroTabularRegressor, __version__


@pytest.fixture(scope="module")
def public_fixture_runtime():
    """Old binary artifacts/state parity have the P40 pinned Linux CPU scope.

    Other supported environments still run fresh same-runtime persistence and
    training tests; they must not imply cross-dependency artifact guarantees.
    """
    provenance = Path(__file__).parent / "fixtures/public_0_3_1/PROVENANCE.json"
    if not provenance.is_file():
        pytest.skip(
            "Historical fixtures are repository-only; excluded from public distributions"
        )
    reference = json.loads(provenance.read_text("utf-8"))
    differences = []
    if platform.system() != "Linux" or platform.python_version() != reference["python"]:
        differences.append("Python/OS")
    for name, expected in reference["dependencies"].items():
        if importlib.metadata.version(name) != expected:
            differences.append(name)
    if differences:
        pytest.skip(
            "Public fixture requires its pinned P40 Linux CPU runtime: "
            + ", ".join(differences)
        )


@pytest.fixture(scope="module", params=["binary", "multiclass", "regression"])
def fitted_version_case(request):
    task = request.param
    X, y = data(task)
    cls = NeuroTabularRegressor if task == "regression" else NeuroTabularClassifier
    return task, cls(**kwargs()).fit(X, y), X


def fresh_load(path, format, X, tmp_path):
    X.to_pickle(tmp_path / "X.pkl")
    script = """
import sys,pickle,joblib,numpy as np,pandas as pd,torch
from pathlib import Path
torch.set_num_threads(1)
root=Path(sys.argv[3])
model=pickle.loads(Path(sys.argv[1]).read_bytes()) if sys.argv[2]=='pickle' else joblib.load(sys.argv[1])
metadata=dict(version=getattr(model,'_fitted_neurotabular_version_',None),schema=getattr(model,'input_schema_',None),features=model.n_features_in_,device=model.device_,classes=getattr(model,'classes_',None))
(root/'metadata.pkl').write_bytes(pickle.dumps(metadata))
X=pd.read_pickle(root/'X.pkl')
np.savez(root/'outputs.npz',prediction=model.predict(X),probability=model.predict_proba(X) if hasattr(model,'predict_proba') else np.array([]))
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path), format, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = pickle.loads((tmp_path / "metadata.pkl").read_bytes())
    with np.load(tmp_path / "outputs.npz") as outputs:
        return metadata, outputs["prediction"].copy(), outputs["probability"].copy()


@pytest.mark.parametrize("format", ["pickle", "joblib"])
def test_new_fitted_version_fresh_process(fitted_version_case, format, tmp_path):
    task, model, X = fitted_version_case
    path = tmp_path / "model.bin"
    if format == "pickle":
        path.write_bytes(pickle.dumps(model))
    else:
        joblib.dump(model, path)
    metadata, prediction, probability = fresh_load(path, format, X, tmp_path)
    assert metadata["version"] == __version__ == model._fitted_neurotabular_version_
    assert metadata["schema"] == model.input_schema_
    assert metadata["features"] == model.n_features_in_ and metadata["device"] == "cpu"
    np.testing.assert_array_equal(prediction, model.predict(X))
    if task != "regression":
        np.testing.assert_array_equal(metadata["classes"], model.classes_)
        np.testing.assert_array_equal(probability, model.predict_proba(X))


def test_internal_version_is_not_prediction_input(fitted_version_case):
    _, model, X = fitted_version_case
    before = model.predict(X)
    model._fitted_neurotabular_version_ = "diagnostic-only"
    try:
        np.testing.assert_array_equal(before, model.predict(X))
    finally:
        model._fitted_neurotabular_version_ = __version__


@pytest.mark.parametrize("format", ["pickle", "joblib"])
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_public_031_fixture_loads_without_inventing_metadata(
    task, format, tmp_path, public_fixture_runtime
):
    X, _ = data(task)
    root = Path(__file__).parent / "fixtures/public_0_3_1"
    metadata, prediction, probability = fresh_load(
        root / f"{task}.{format}", format, X, tmp_path
    )
    assert metadata["version"] is None and metadata["schema"] is None
    with np.load(root / f"{task}.outputs.npz") as expected:
        np.testing.assert_array_equal(prediction, expected["prediction"])
        np.testing.assert_array_equal(probability, expected["probability"])


def test_new_fit_preserves_public_line_trained_state(
    fitted_version_case, public_fixture_runtime
):
    task, model, X = fitted_version_case
    root = Path(__file__).parent / "fixtures/public_0_3_1"
    old = joblib.load(root / f"{task}.joblib")
    new_state, old_state = model._model_.state_dict(), old._model_.state_dict()
    assert new_state.keys() == old_state.keys()
    for key in new_state:
        assert torch.equal(new_state[key], old_state[key]), key
    np.testing.assert_array_equal(model.predict(X), old.predict(X))

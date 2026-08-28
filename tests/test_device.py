import numpy as np
import pandas as pd
import pytest
import torch

from neurotabular import NeuroTabularClassifier
from neurotabular.device import resolve_device


def _mock_cuda(monkeypatch, *, count=1, capability=(6, 0), probe_error=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla P100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(
        torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "compute_90"]
    )
    monkeypatch.setattr(
        "neurotabular.device._run_cuda_probe", lambda device: probe_error
    )


def test_auto_incompatible_cuda_warns_and_falls_back(monkeypatch):
    _mock_cuda(
        monkeypatch,
        probe_error="AcceleratorError: no kernel image is available for execution",
    )
    with pytest.warns(RuntimeWarning, match="falling back to CPU") as warning:
        device, info = resolve_device("auto")

    message = str(warning[0].message)
    assert device == torch.device("cpu")
    assert info["fallback_used"] is True
    assert info["resolved_device"] == "cpu"
    assert "Tesla P100" in message
    assert "sm_60" in message
    assert "sm_70" in message
    assert "no kernel image" in message


def test_auto_incompatible_cuda_fit_continues_on_cpu(monkeypatch):
    _mock_cuda(
        monkeypatch,
        capability=(8, 0),
        probe_error="no kernel image is available for execution",
    )
    X = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 40)})
    y = np.resize([0, 1], len(X))

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        model = NeuroTabularClassifier(
            hidden_dim=8,
            n_blocks=1,
            max_epochs=1,
            patience=1,
            device="auto",
        ).fit(X, y)

    assert model.device_ == "cpu"
    assert model.device_info_["fallback_used"] is True
    assert model.device_info_["amp_enabled"] is False
    assert model.profile_["training"]["amp_enabled"] is False
    assert np.isfinite(model.predict_proba(X.iloc[:3])).all()


def test_explicit_incompatible_cuda_has_clear_diagnostic(monkeypatch):
    _mock_cuda(
        monkeypatch,
        probe_error="AcceleratorError: no kernel image is available for execution",
    )
    with pytest.raises(RuntimeError, match="requested device 'cuda'") as error:
        resolve_device("cuda")

    message = str(error.value)
    assert "Tesla P100" in message
    assert "sm_60" in message
    assert "PyTorch=" in message
    assert "compiled CUDA architectures=sm_70, sm_75, compute_90" in message


def test_explicit_missing_cuda_index_is_clear(monkeypatch):
    _mock_cuda(monkeypatch, count=1)
    with pytest.raises(RuntimeError, match="index 1 is unavailable"):
        resolve_device("cuda:1")


def test_cuda_unavailable_auto_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, info = resolve_device("auto")
    assert device == torch.device("cpu")
    assert info["fallback_used"] is False
    assert info["cuda_available"] is False


def test_explicit_cpu_does_not_query_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("device='cpu' must not query CUDA"),
    )
    device, info = resolve_device("cpu")
    assert device == torch.device("cpu")
    assert info["requested_device"] == "cpu"


def test_cuda_unavailable_explicit_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_compatible_cuda_auto_selects_cuda(monkeypatch):
    _mock_cuda(monkeypatch, probe_error=None)
    device, info = resolve_device("auto")
    assert device == torch.device("cuda:0")
    assert info["probe_succeeded"] is True
    assert info["fallback_used"] is False


def test_cpu_seed_does_not_touch_visible_but_unselected_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: pytest.fail("CUDA seeding must not run for a CPU path"),
    )
    NeuroTabularClassifier(random_state=11)._set_random_state(torch.device("cpu"))


def test_cuda_seed_runs_only_after_cuda_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", calls.append)
    NeuroTabularClassifier(random_state=11)._set_random_state(torch.device("cuda:0"))
    assert calls == [11]

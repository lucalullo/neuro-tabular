"""Privacy checks for the optional support example, not a new public API."""

import importlib.util
import json
import platform
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "support_environment",
    Path(__file__).resolve().parents[1] / "examples/support_environment.py",
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_allowlist_and_no_environment_or_host_collection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not inspect host identity")

    monkeypatch.setenv("SECRET_TOKEN", "PRIVATE_SENTINEL_TOKEN")
    monkeypatch.setattr(platform, "node", forbidden)
    monkeypatch.setattr(platform, "uname", forbidden)
    result = support.environment_report(requested_device="cpu", resolved_device="cpu")
    assert set(result) == {
        "neurotabular",
        "python",
        "torch",
        "numpy",
        "pandas",
        "scikit_learn",
        "os",
        "requested_device",
        "resolved_device",
        "cuda_available",
        "torch_cuda",
    }
    assert result["neurotabular"] == support.__version__
    assert result["requested_device"] == result["resolved_device"] == "cpu"
    encoded = json.dumps(result)
    assert "PRIVATE_SENTINEL" not in encoded
    assert "/home/" not in encoded and "Users" not in encoded and "\\" not in encoded


@pytest.mark.parametrize(
    "private",
    ["/" + "home" + "/private/token", "C:" + "\\Users\\private", "api-key=SECRET"],
)
def test_untrusted_free_text_is_not_echoed(monkeypatch, private):
    monkeypatch.setattr(support, "__version__", private)
    monkeypatch.setattr(support.sys, "platform", private)
    result = support.environment_report(
        requested_device=private, resolved_device=private
    )
    assert result["neurotabular"] == "UNKNOWN" and result["os"] == "Other"
    assert result["requested_device"] == result["resolved_device"] == "INVALID"
    assert private not in json.dumps(result)


def test_exact_allowlisted_output(monkeypatch):
    monkeypatch.setattr(support, "__version__", "0.3.1")
    monkeypatch.setattr(support.sys, "platform", "linux")
    monkeypatch.setattr(support.sys, "version_info", (3, 12, 3))
    for module, version in (
        (support.torch, "2.14.0+cpu"),
        (support.np, "2.5.3"),
        (support.pd, "3.0.1"),
        (support.sklearn, "1.8.0"),
    ):
        monkeypatch.setattr(module, "__version__", version)
    monkeypatch.setattr(support.torch.version, "cuda", None)
    monkeypatch.setattr(support.torch.cuda, "is_available", lambda: False)
    assert support.environment_report(
        requested_device="cpu", resolved_device="cpu"
    ) == {
        "neurotabular": "0.3.1",
        "python": "3.12.3",
        "torch": "2.14.0+cpu",
        "numpy": "2.5.3",
        "pandas": "3.0.1",
        "scikit_learn": "1.8.0",
        "os": "Linux",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "cuda_available": False,
        "torch_cuda": None,
    }

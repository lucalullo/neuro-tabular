"""Safe runtime device selection and CUDA diagnostics."""

from __future__ import annotations

import warnings
from typing import Any

import torch


def resolve_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    """Resolve a user device after a real, synchronized CUDA kernel probe."""

    if requested == "cpu":
        return torch.device("cpu"), _cpu_info(requested)

    if requested != "auto":
        try:
            parsed = torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device {requested!r}.") from exc
        if parsed.type != "cuda":
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string.")
    else:
        parsed = torch.device("cuda")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return _cuda_failure(
            requested,
            "the CUDA availability check failed",
            exc,
            cuda_available=False,
        )
    if not cuda_available:
        if requested == "auto":
            return torch.device("cpu"), _cpu_info(requested)
        raise RuntimeError(
            _cuda_error_message(
                requested,
                "CUDA is not available to this PyTorch installation",
                {"cuda_available": False},
            )
        )

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        return _cuda_failure(
            requested,
            "the CUDA device count check failed",
            exc,
            cuda_available=True,
        )

    if parsed.index is None:
        try:
            device_index = int(torch.cuda.current_device())
        except Exception:
            device_index = 0
    else:
        device_index = parsed.index
    if device_index < 0 or device_index >= device_count:
        details = {
            "cuda_available": True,
            "cuda_device_count": device_count,
            "device_index": device_index,
        }
        reason = f"CUDA device index {device_index} is unavailable"
        if requested == "auto":
            return _warn_and_fallback(requested, reason, details)
        raise RuntimeError(_cuda_error_message(requested, reason, details))

    device = torch.device(f"cuda:{device_index}")
    details = _cuda_details(requested, device, device_count)
    probe_error = _run_cuda_probe(device)
    if probe_error is not None:
        details["probe_error"] = probe_error
        return _cuda_failure(
            requested,
            "the CUDA compatibility probe failed",
            None,
            **details,
        )

    details.update(
        {
            "resolved_device": str(device),
            "fallback_used": False,
            "probe_succeeded": True,
        }
    )
    return device, details


def _run_cuda_probe(device: torch.device) -> str | None:
    """Launch and synchronize one tiny kernel, returning a diagnostic on failure."""

    probe: torch.Tensor | None = None
    try:
        probe = torch.ones(1, dtype=torch.float32, device=device)
        probe.add_(1.0)
        torch.cuda.synchronize(device)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        del probe
    return None


def _cuda_details(
    requested: str, device: torch.device, device_count: int
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "requested_device": requested,
        "resolved_device": None,
        "fallback_used": False,
        "cuda_available": True,
        "cuda_device_count": device_count,
        "device_index": device.index,
        "gpu_name": _cuda_metadata(torch.cuda.get_device_name, device),
        "compute_capability": _cuda_metadata(torch.cuda.get_device_capability, device),
        "supported_architectures": _cuda_metadata(torch.cuda.get_arch_list),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "probe_succeeded": False,
    }
    capability = details["compute_capability"]
    details["amp_enabled"] = bool(
        isinstance(capability, tuple) and capability >= (7, 0)
    )
    return details


def _cuda_metadata(function, *args):
    try:
        return function(*args)
    except Exception:
        return None


def _cpu_info(requested: str) -> dict[str, Any]:
    cuda_was_checked = requested == "auto"
    return {
        "requested_device": requested,
        "resolved_device": "cpu",
        "fallback_used": False,
        "cuda_available": False if cuda_was_checked else None,
        "cuda_device_count": 0 if cuda_was_checked else None,
        "device_index": None,
        "gpu_name": None,
        "compute_capability": None,
        "supported_architectures": [],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "probe_succeeded": False,
        "amp_enabled": False,
    }


def _cuda_failure(
    requested: str,
    reason: str,
    exception: Exception | None,
    **details: Any,
) -> tuple[torch.device, dict[str, Any]]:
    if exception is not None:
        details["probe_error"] = f"{type(exception).__name__}: {exception}"
    if requested == "auto":
        return _warn_and_fallback(requested, reason, details)
    raise RuntimeError(_cuda_error_message(requested, reason, details))


def _warn_and_fallback(
    requested: str, reason: str, details: dict[str, Any]
) -> tuple[torch.device, dict[str, Any]]:
    info = {
        "requested_device": requested,
        "resolved_device": "cpu",
        "fallback_used": True,
        "fallback_reason": reason,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "probe_succeeded": False,
        "amp_enabled": False,
        **details,
    }
    info["resolved_device"] = "cpu"
    info["fallback_used"] = True
    info["amp_enabled"] = False
    warnings.warn(
        _cuda_error_message(requested, f"{reason}; falling back to CPU", info),
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.device("cpu"), info


def _cuda_error_message(requested: str, reason: str, details: dict[str, Any]) -> str:
    capability = details.get("compute_capability")
    capability_text = (
        f"sm_{capability[0]}{capability[1]}"
        if isinstance(capability, tuple) and len(capability) == 2
        else "unavailable"
    )
    architectures = details.get("supported_architectures")
    architecture_text = ", ".join(architectures) if architectures else "unavailable"
    return (
        f"NeuroTabular cannot use requested device {requested!r}: {reason}. "
        f"GPU={details.get('gpu_name') or 'unavailable'}; "
        f"compute capability={capability_text}; PyTorch={torch.__version__}; "
        f"PyTorch CUDA={torch.version.cuda or 'unavailable'}; "
        f"compiled CUDA architectures={architecture_text}; "
        f"device count={details.get('cuda_device_count', 'unavailable')}; "
        f"probe error={details.get('probe_error', 'none')}."
    )

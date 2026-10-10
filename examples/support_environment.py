"""Small opt-in support snippet; no estimator API, paths, data or host identifiers."""

import json
import re
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

from neurotabular import __version__


def _version(value):
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,79}", value)
        else "UNKNOWN"
    )


def _device(value):
    if value is None:
        return None
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"cpu|auto|cuda(?::[0-9]{1,3})?", value)
        else "INVALID"
    )


def environment_report(*, requested_device="auto", resolved_device=None):
    """Return only allowlisted versions, OS family and optional device labels.

    Device labels are supplied by the caller; this does not resolve a device or
    launch a CUDA kernel. Review the result before posting it publicly.
    """
    system = {"win32": "Windows", "linux": "Linux", "darwin": "Darwin"}.get(
        sys.platform, "Other"
    )
    return {
        "neurotabular": _version(__version__),
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": _version(torch.__version__),
        "numpy": _version(np.__version__),
        "pandas": _version(pd.__version__),
        "scikit_learn": _version(sklearn.__version__),
        "os": system,
        "requested_device": _device(requested_device),
        "resolved_device": _device(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": None
        if torch.version.cuda is None
        else _version(torch.version.cuda),
    }


if __name__ == "__main__":
    print(json.dumps(environment_report(), indent=2))

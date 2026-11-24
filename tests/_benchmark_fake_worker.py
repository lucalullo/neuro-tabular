"""Artificial IPC/receipt load for runner tests. No estimator or training."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--kind", default="serve")
args = parser.parse_args()
job = json.loads(args.job.read_text())
folder = args.job.parent
started_cpu, started_wall = time.process_time(), time.perf_counter()
if args.kind == "serve":
    import numpy as np

    from benchmarks.competitive.data import read_bundle
    from benchmarks.competitive.metrology import serve

    class ArithmeticFixture:
        class_order_ = ["negative", "positive"]

        def predict(self, X):
            return np.where(X["a"].fillna(0).to_numpy() > 0, "positive", "negative")

        def predict_proba(self, X):
            value = np.where(X["a"].fillna(0).to_numpy() > 0, 0.75, 0.25)
            return np.column_stack([1 - value, value])

    X, _, groups, _ = read_bundle(job["input_file"], include_target=False)
    serve(ArithmeticFixture(), X, groups, job, folder, started_cpu, started_wall)
elif args.kind == "receipt":
    from benchmarks.competitive.control import atomic_json

    result = {
        "job_id": job["job_id"],
        "worker_cpu_s": time.process_time() - started_cpu,
    }
    atomic_json(folder / "RESULT.json", result)
    atomic_json(
        folder / "WORKER_DONE.json",
        {
            "job_id": job["job_id"],
            "pid": os.getpid(),
            "result_sha256": hashlib.sha256(
                (folder / "RESULT.json").read_bytes()
            ).hexdigest(),
        },
    )
elif args.kind == "incomplete":
    (folder / "half-written.partial").write_bytes(b"retained incomplete artifact")
    raise SystemExit(3)
else:
    raise ValueError(args.kind)

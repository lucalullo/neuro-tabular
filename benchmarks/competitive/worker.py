"""One isolated fit or replay; no source downloads, HPO loop or implicit refit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


_METER = None
_OUTPUT = None

def main():
    global _METER, _OUTPUT
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    output = args.job.parent
    _OUTPUT = output
    from .work_counters import ForwardCounter
    _METER = ForwardCounter(job["model"], job["max_torch_root_forward_calls"])
    if (output / "WORKER_DONE.json").exists():
        raise ValueError("Do not repeat completed work")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if job.get("nt_import_root"):
        sys.path.insert(0, job["nt_import_root"])
    from importlib.metadata import version

    for name, expected in job.get("package_versions", {}).items():
        if version(name) != expected:
            raise ValueError("Unpinned worker dependency: " + name)
    import psutil
    inherited_affinity = psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([job["logical_cpu"]])
    if psutil.Process().cpu_affinity() != [job["logical_cpu"]]:
        raise RuntimeError("WORKER_AFFINITY_NOT_APPLIED")
    import joblib
    import numpy as np
    import torch

    from .adapters import BenchmarkModel
    from .control import atomic_json
    from .data import digest, read_bundle
    from .metrics import score

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _METER.install()
    optimizer_steps = [0]
    base_init = torch.optim.Optimizer.__init__
    step_handles = []

    def before_step(optimizer, arguments, keywords):
        if (output / "STOP_REQUESTED").exists():
            raise RuntimeError("Controller requested stop")
        if optimizer_steps[0] >= job["max_optimizer_steps"]:
            raise RuntimeError("OPTIMIZER_STEP_LIMIT")
        optimizer_steps[0] += 1
        _METER.optimizer_steps = optimizer_steps[0]

    def watched_init(optimizer, *arguments, **keywords):
        base_init(optimizer, *arguments, **keywords)
        step_handles.append(optimizer.register_step_pre_hook(before_step))

    torch.optim.Optimizer.__init__ = watched_init
    result = {
        "job_id": job["job_id"],
        "job_sha256": digest(args.job),
        "mode": job["mode"],
        "model": job["model"],
        "task": job["task"],
        "seed": job["seed"],
        "optimizer_steps": 0,
        "fit_calls": 0,
        "prediction_API_calls": 0,
    }
    if job["mode"] == "fit":
        if digest(job["train_file"]) != job["train_sha256"]:
            raise ValueError("TRAIN artifact changed")
        X, y, groups, categorical = read_bundle(job["train_file"])
        model = BenchmarkModel(
            job["model"], job["task"], job["seed"], categorical, job["params"]
        )
        wall, cpu = time.perf_counter(), time.process_time()
        _METER.fit_started += 1
        atomic_json(output / "COUNTERS_PARTIAL.json", _METER.snapshot())
        model.fit(X, y)
        _METER.fit_completed += 1
        atomic_json(output / "COUNTERS_PARTIAL.json", _METER.snapshot())
        result.update(
            fit_compute_cpu_s=time.process_time() - cpu,
            fit_compute_wall_s=time.perf_counter() - wall,
            fit_calls=1,
            constructor=model.public_constructor_,
            class_order=model.class_order_,
        )
        model_path = output / "model.joblib"
        # Never pickle local step hooks or instrumentation closures.
        for step_handle in step_handles:
            step_handle.remove()
        step_handles.clear()
        torch.optim.Optimizer.__init__ = base_init
        _METER.close()
        try:
            joblib.dump(model, model_path, compress=0)
        finally:
            _METER.install()
        if model_path.stat().st_size > job["max_model_bytes"]:
            raise ValueError("MODEL_SIZE_LIMIT")
        result.update(
            model_sha256=digest(model_path), model_bytes=model_path.stat().st_size
        )
        result.update(
            fit_cpu_s=time.process_time() - started_cpu,
            fit_wall_s=time.perf_counter() - started_wall,
        )
        if job.get("validation_file"):
            if digest(job["validation_file"]) != job["validation_sha256"]:
                raise ValueError("VAL artifact changed")
            val_X, val_y, _, _ = read_bundle(job["validation_file"])
            values = (
                model.predict(val_X)
                if job["task"] == "regression"
                else model.predict_proba(val_X)
            )
            result["validation_metrics"] = score(
                job["task"], val_y, values, model.class_order_
            )
            result["prediction_API_calls"] = 1
            if job.get("qualification"):
                labels = values if job["task"] == "regression" else model.predict(val_X)
                result["prediction_API_calls"] += job["task"] != "regression"
                if labels.dtype.kind == "O":
                    labels = labels.astype("str")
                with (output / "predictions.npz").open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        labels=labels,
                        values=values,
                        groups=np.asarray([]),
                        classes=np.asarray(model.class_order_ or [], dtype="str"),
                    )
                result["predictions_sha256"] = digest(output / "predictions.npz")
    elif job["mode"] in {"inference", "replay", "serve"}:
        if (
            digest(job["model_file"]) != job["model_sha256"]
            or digest(job["input_file"]) != job["input_sha256"]
        ):
            raise ValueError("Changed sealed model/input")
        _METER.load_started += 1
        model = joblib.load(job["model_file"])
        _METER.load_completed += 1
        X, _, groups, _ = read_bundle(job["input_file"], include_target=False)
        if job["mode"] == "serve":
            from .metrology import serve

            serve(model, X, groups, job, output, started_cpu, started_wall, forward_counter=_METER)
            return
        if job["mode"] == "replay":
            if digest(job["reference_predictions"]) != job["reference_sha256"]:
                raise ValueError("Changed reference predictions")
            labels = model.predict(X)
            values = labels if job["task"] == "regression" else model.predict_proba(X)
            with np.load(job["reference_predictions"], allow_pickle=False) as reference:
                if not np.array_equal(
                    labels, reference["labels"]
                ) or not np.array_equal(values, reference["values"]):
                    raise ValueError("REPLAY_IDENTITY")
            result["prediction_API_calls"] = 1 if job["task"] == "regression" else 2
        else:
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            labels = model.predict(X)
            result["cold_predict"] = {
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
            result["prediction_API_calls"] = 1
            for _ in range(5):
                model.predict(X)
                result["prediction_API_calls"] += 1
            records, previous = [], {}
            for round_index, batch in enumerate([1, 32, "full"] * 4):
                subset = X if batch == "full" else X.iloc[: min(batch, len(X))]
                count = {1: 256, 32: 64, "full": 8}[batch]
                wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                for _ in range(count):
                    repeated = model.predict(subset)
                duration = {
                    "wall_ns": time.perf_counter_ns() - wall,
                    "cpu_ns": time.process_time_ns() - cpu,
                }
                if batch in previous and not np.array_equal(previous[batch], repeated):
                    raise ValueError("NONDETERMINISTIC_PREDICTIONS")
                previous[batch] = repeated.copy()
                records.append(
                    {"round": round_index, "batch": batch, "calls": count, **duration}
                )
                result["prediction_API_calls"] += count
            values = labels if job["task"] == "regression" else model.predict_proba(X)
            if job["task"] != "regression":
                result["prediction_API_calls"] += 1
            if labels.dtype.kind == "O":
                labels = labels.astype("str")
            with (output / "predictions.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    labels=labels,
                    values=values,
                    groups=groups,
                    classes=np.asarray(model.class_order_ or [], dtype="str"),
                )
            result.update(
                timing_blocks=records,
                predictions_sha256=digest(output / "predictions.npz"),
            )
    else:
        raise ValueError("Unknown worker mode")
    result.update(_METER.snapshot())
    if job.get("qualification") and job["model"] in __import__("benchmarks.competitive.work_counters", fromlist=["NEURAL"]).NEURAL and _METER.calls == 0:
        raise RuntimeError("Q24_FORWARD_COUNTER_NOT_EXERCISED")
    result.update(
        optimizer_steps=optimizer_steps[0],
        worker_cpu_s=time.process_time() - started_cpu,
        worker_wall_s=time.perf_counter() - started_wall,
        torch_threads=torch.get_num_threads(),
        inherited_affinity=inherited_affinity,
        effective_affinity=psutil.Process().cpu_affinity(),
    )
    # The receipt and its completion acknowledgement precede normal process exit.
    # The controller still requires exit code 0; the acknowledgement alone is insufficient.
    clean = json.loads(
        json.dumps(
            result,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )
    )
    atomic_json(output / "RESULT.json", clean)
    atomic_json(
        output / "WORKER_DONE.json",
        {
            "job_id": job["job_id"],
            "pid": os.getpid(),
            "result_sha256": hashlib.sha256(
                (output / "RESULT.json").read_bytes()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _OUTPUT is not None:
            from .control import atomic_json
            atomic_json(_OUTPUT / "COUNTERS_PARTIAL.json", {
                "error": repr(error), "pid": os.getpid(),
                "partial_only_not_completion": True,
                **(_METER.snapshot() if _METER is not None else {})})
        raise
    finally:
        if _METER is not None:
            _METER.close()

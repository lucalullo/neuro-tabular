"""Prepare, plan, and explicitly execute the frozen public benchmark.

Planning and --help never import an estimator. Real execution is opt-in. Models
are fitted before the TEST scoring barrier; HPO sees only FIT and validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .control import Budget, atomic_json, host_memory_ready, launch
from .data import digest, prepare_family

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan(config):
    rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for method in config["methods"]:
                rows.append(
                    {
                        "family": dataset["id"],
                        "task": dataset["task"],
                        "seed": seed,
                        "method": method,
                        "default_fits": 1,
                        "hpo_trials": len(config["hpo_candidates"][method]),
                        "hpo_final_fits": 1,
                    }
                )
    return {
        "schema": "NT_PUBLIC_PLAN/1",
        "cases": rows,
        "maxima": config["budget"],
        "real_execution": False,
    }


def choose_trial(trials, metric):
    """Choose from completed validation records only; never accepts TEST scores."""
    import math

    candidates = []
    for index, result in enumerate(trials):
        if result is None or "validation_metrics" not in result:
            continue
        value = result["validation_metrics"].get(metric)
        if value is not None and math.isfinite(value):
            candidates.append((value, index))
    return min(candidates)[1] if candidates else None


def command_for(environment, job_path):
    return [
        environment["python"],
        "-B",
        "-u",
        "-m",
        "benchmarks.competitive.worker",
        "--job",
        str(job_path),
    ]


def worker_environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(HERE.parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def base_job(config, environment, job_id, method, task, seed, mode):
    return {
        "job_id": job_id,
        "mode": mode,
        "model": method,
        "task": task,
        "seed": seed,
        "logical_cpu": config["logical_cpu"],
        "max_optimizer_steps": config["max_optimizer_steps_per_fit"],
        "max_model_bytes": config["max_model_bytes"],
        "params": {},
        "nt_import_root": environment.get("nt_import_roots", {}).get(method),
        "package_versions": config["package_versions"],
    }


def restore_job(folder):
    """Reuse only a complete, hash-bound receipt with normal process exit."""
    if not folder.exists():
        return None
    terminal = folder / "TERMINAL.json"
    if not terminal.exists():
        raise RuntimeError(
            "Interrupted ticket requires reconciliation, never automatic retry: "
            + str(folder)
        )
    cost = load(terminal)
    if cost["reason"] is not None:
        return {"failed": True, "cost": cost}
    result_path = folder / "RESULT.json"
    result, ack, job = (
        load(result_path),
        load(folder / "WORKER_DONE.json"),
        load(folder / "JOB.json"),
    )
    if (
        cost["exit_code"] != 0
        or ack["result_sha256"] != digest(result_path)
        or ack["job_id"] != job["job_id"]
    ):
        raise RuntimeError("Changed terminal receipt")
    if result.get("job_sha256") != digest(folder / "JOB.json"):
        raise RuntimeError("Changed job definition")
    model = folder / "model.joblib"
    if model.exists() and result.get("model_sha256") != digest(model):
        raise RuntimeError("Changed fitted model")
    if result.get("model_sha256") and not model.exists():
        tombstone = folder / "RETIRED_MODEL.json"
        if (
            not tombstone.exists()
            or "-trial" not in job["job_id"]
            or job.get("qualification")
            or load(tombstone)["sha256"] != result["model_sha256"]
        ):
            raise RuntimeError("A required fitted model is missing")
    return result


def dispatch(root, job, limits, environment, budget):
    folder = root / "jobs" / job["job_id"]
    previous = restore_job(folder)
    if previous is not None:
        return previous
    if any(row["job_id"] == job["job_id"] for row in budget.journal.events()):
        raise RuntimeError("Previously started ticket cannot be relaunched")
    from .budget_contract import attach_forward_limit
    operations = limits.get("operations", {})
    attach_forward_limit(job, operations)
    if job["mode"] == "fit":
        kind = "artificial_fits" if job.get("qualification") else "real_hpo_trials" if "-trial" in job["job_id"] else "real_hpo_final_fits" if job["job_id"].startswith("hpo4-") else "real_default_fits"
        operations[kind] = 1
    host_memory_ready(limits["rss_bytes"])
    budget.reserve(
        limits["cpu_s"] + 2, limits["wall_s"] + 6, limits["archive_bytes"], **operations
    )
    folder.mkdir(parents=True)
    atomic_json(folder / "JOB.json", job)
    cost = launch(
        command_for(environment, folder / "JOB.json"),
        folder,
        limits,
        budget.journal,
        job["job_id"],
        worker_environment(),
    )
    budget.checkpoint()
    return (
        {"failed": True, "cost": cost}
        if cost["reason"]
        else load(folder / "RESULT.json")
    )


def fit_limits(config, cpu_s, method, qualification=False, trial=False):
    kind = "artificial_fits" if qualification else "real_hpo_trials" if trial else "real_default_fits" if cpu_s == config["fit_limits"]["default_cpu_s"] else "real_hpo_final_fits"
    return {
        "cpu_s": cpu_s,
        "wall_s": 2 * cpu_s,
        "rss_bytes": config["rss_bytes"],
        "archive_bytes": 128 * 1024**2,
        "operations": {
            "fit_attempts": 1,
            "optimizer_steps": 0
            if method in {"xgboost", "lightgbm", "catboost"}
            else config["max_optimizer_steps_per_fit"],
            "prediction_API_calls": 2 if qualification else int(trial),
            "scoring_artificial": int(qualification),
            "scoring_validation": int(trial),
        },
    }


def artificial_roles(root):
    """Deterministic mixed synthetic inputs. Called only in explicit run, no fit."""
    import numpy as np
    import pandas as pd

    from .data import write_bundle

    folder = root / "artificial"
    if (folder / "READY.json").exists():
        record = load(folder / "READY.json")
        for detail in record.values():
            if (
                digest(detail["fit"]) != detail["fit_sha256"]
                or digest(detail["validation"]) != detail["validation_sha256"]
            ):
                raise RuntimeError("Changed artificial inputs")
        return record
    folder.mkdir()
    rng = np.random.default_rng(418)
    X = pd.DataFrame(rng.normal(size=(160, 5)), columns=list("abcde"))
    X["category"] = np.where(np.arange(160) % 3 == 0, "north", "south")
    X.loc[::19, "a"] = np.nan
    X.loc[::23, "category"] = np.nan
    targets = {
        "binary": pd.Series(np.where(np.arange(160) % 2, "positive", "negative")),
        "multiclass": pd.Series(["class" + str(index % 3) for index in range(160)]),
        "regression": pd.Series(np.nan_to_num(X["a"]) + 0.5 * X["b"]),
    }
    record = {}
    for task, y in targets.items():
        item = {}
        for name, indices in (
            ("fit", np.arange(128)),
            ("validation", np.arange(128, 160)),
        ):
            path = folder / (task + "-" + name + ".npz")
            write_bundle(path, X, y, np.arange(160), indices, ["category"])
            item[name], item[name + "_sha256"] = str(path), digest(path)
        record[task] = item
    atomic_json(folder / "READY.json", record)
    return record


def cold_replay(
    root, job, model_folder, input_file, reference, config, environment, budget
):
    job.update(
        mode="replay",
        model_file=str(model_folder / "model.joblib"),
        model_sha256=load(model_folder / "RESULT.json")["model_sha256"],
        input_file=str(input_file),
        input_sha256=digest(input_file),
        reference_predictions=str(reference),
        reference_sha256=digest(reference),
    )
    limits = {
        **config["replay_limits"],
        "rss_bytes": config["rss_bytes"],
        "archive_bytes": 8 * 1024**2,
        "operations": {
            "cold_replays": 1,
            "model_load_operations": 1,
            "prediction_API_calls": 1 if job["task"] == "regression" else 2,
        },
    }
    return dispatch(root, job, limits, environment, budget)


def qualify(root, config, environment, budget):
    inputs = artificial_roles(root)
    qualified = {}
    for method in config["methods"]:
        for task in ("binary", "multiclass", "regression"):
            job_id = "Q-" + method + "-" + task
            files = inputs[task]
            job = base_job(config, environment, job_id, method, task, 42, "fit")
            job.update(
                train_file=files["fit"],
                train_sha256=files["fit_sha256"],
                validation_file=files["validation"],
                validation_sha256=files["validation_sha256"],
                qualification=True,
            )
            result = dispatch(
                root,
                job,
                fit_limits(
                    config,
                    config["fit_limits"]["artificial_cpu_s"],
                    method,
                    qualification=True,
                ),
                environment,
                budget,
            )
            passed = not result.get("failed", False)
            if passed:
                folder = root / "jobs" / job_id
                replay_job = base_job(
                    config, environment, job_id + "-replay", method, task, 42, "replay"
                )
                replay = cold_replay(
                    root,
                    replay_job,
                    folder,
                    Path(files["validation"]),
                    folder / "predictions.npz",
                    config,
                    environment,
                    budget,
                )
                passed = not replay.get("failed", False)
            qualified[method + ":" + task] = passed
    atomic_json(root / "QUALIFICATION.json", qualified)
    return qualified


def retire_trial(folder):
    """Only our verified trial weight file, after its validation receipt is saved."""
    model = folder / "model.joblib"
    if not model.exists():
        return
    record = load(folder / "RESULT.json")
    if "validation_metrics" not in record or digest(model) != record["model_sha256"]:
        raise RuntimeError("Trial retirement requires a verified validation receipt")
    atomic_json(
        folder / "RETIRED_MODEL.json",
        {
            "sha256": record["model_sha256"],
            "bytes": model.stat().st_size,
            "reason": "Validation selection recorded; final model is a separately counted fit on TRAIN pool.",
        },
    )
    model.unlink()


def train_all(root, data_root, config, environment, budget, qualified):
    models, failures = [], []
    for case in plan(config)["cases"]:
        family, seed, method, task = (
            case["family"],
            case["seed"],
            case["method"],
            case["task"],
        )
        if not qualified.get(method + ":" + task, False):
            for failed_track in config["tracks"]:
                failures.append({**case, "track": failed_track,
                    "reason": "Artificial qualification did not pass; no real fit opened",
                    "failure_stage": "Q24", "natural_fit_started": False})
            continue
        data = data_root / family / str(seed)
        manifest = load(data_root / family / "MANIFEST.json")["splits"][str(seed)][
            "roles"
        ]
        for track in config["tracks"]:
            prefix = f"{track}-{family}-{seed}-{method}"
            params = {}
            tuning_cpu, tuning_wall = 0.0, 0.0
            if track == "hpo4":
                trial_records = []
                for index, candidate in enumerate(config["hpo_candidates"][method]):
                    job = base_job(
                        config,
                        environment,
                        prefix + "-trial" + str(index),
                        method,
                        task,
                        seed,
                        "fit",
                    )
                    job.update(
                        params=candidate,
                        train_file=str(data / "fit.npz"),
                        train_sha256=manifest["fit"]["sha256"],
                        validation_file=str(data / "validation.npz"),
                        validation_sha256=manifest["validation"]["sha256"],
                    )
                    result = dispatch(
                        root,
                        job,
                        fit_limits(
                            config,
                            config["fit_limits"]["trial_cpu_s"],
                            method,
                            trial=True,
                        ),
                        environment,
                        budget,
                    )
                    terminal = load(root / "jobs" / job["job_id"] / "TERMINAL.json")
                    tuning_cpu += terminal["cpu_s"]
                    tuning_wall += terminal["wall_s"]
                    trial_records.append(
                        None if result.get("failed", False) else result
                    )
                selected = choose_trial(
                    trial_records, config["hpo_selection_metric"][task]
                )
                selection = {
                    "case": case,
                    "track": track,
                    "selected_index": selected,
                    "tuning_cpu_s": tuning_cpu,
                    "tuning_wall_s": tuning_wall,
                    "test_information_used": False,
                    "complete_search": all(item is not None for item in trial_records),
                }
                path = root / "selections" / (prefix + ".json")
                if path.exists():
                    if load(path) != selection:
                        raise RuntimeError("Changed HPO selection")
                else:
                    atomic_json(path, selection)
                for index, item in enumerate(trial_records):
                    if item is not None:
                        retire_trial(root / "jobs" / (prefix + "-trial" + str(index)))
                if selected is None:
                    failures.append(
                        {
                            **case,
                            "track": track,
                            "reason": "All four validation trials failed",
                        }
                    )
                    continue
                params = config["hpo_candidates"][method][selected]
            job = base_job(
                config, environment, prefix + "-final", method, task, seed, "fit"
            )
            job.update(
                params=params,
                train_file=str(data / "pool.npz"),
                train_sha256=manifest["pool"]["sha256"],
            )
            result = dispatch(
                root,
                job,
                fit_limits(
                    config,
                    config["fit_limits"][
                        "default_cpu_s" if track == "default" else "hpo_final_cpu_s"
                    ],
                    method,
                ),
                environment,
                budget,
            )
            if result.get("failed", False):
                failures.append(
                    {**case, "track": track, "reason": result["cost"]["reason"]}
                )
                continue
            models.append(
                {
                    **case,
                    "track": track,
                    "job_id": job["job_id"],
                    "model_sha256": result["model_sha256"],
                    "test_file": str(data / "test.npz"),
                    "test_sha256": manifest["test"]["sha256"],
                    "tuning_cpu_s": tuning_cpu,
                    "tuning_wall_s": tuning_wall,
                }
            )
    seal = {
        "models": models,
        "failures": failures,
        "all_final_fits_closed_before_test": True,
    }
    path = root / "MODELS_SEALED.json"
    if path.exists():
        if load(path) != seal:
            raise RuntimeError("Changed final model seal")
    else:
        atomic_json(path, seal)
    return seal


def arrays_equal(first, second):
    import numpy as np

    with (
        np.load(first, allow_pickle=False) as a,
        np.load(second, allow_pickle=False) as b,
    ):
        return set(a.files) == set(b.files) and all(
            np.array_equal(a[key], b[key]) for key in a.files
        )


def verify_pair(pair):
    receipt = load(pair / "PAIR.json")
    if not receipt["both_workers_exited"] or not receipt["PASS_EXECUTION"]:
        raise RuntimeError("Incomplete earlier pair")
    for which in (0, 1):
        folder = pair / str(which)
        ack = load(folder / "WORKER_DONE.json")
        result = load(folder / "RESULT.json")
        predicted = load(folder / "PREDICTIONS_COMPLETE.json")
        job = load(folder / "JOB.json")
        if (
            ack["result_sha256"] != digest(folder / "RESULT.json")
            or ack["job_id"] != job["job_id"]
            or result["job_sha256"] != digest(folder / "JOB.json")
            or predicted["prediction_sha256"] != digest(folder / "predictions.npz")
        ):
            raise RuntimeError("Changed metrology evidence")
    return receipt


def measure_all(root, config, environment, budget, seal):
    from .metrology import run_pair

    lookup = {
        (m["family"], m["seed"], m["track"], m["method"]): m for m in seal["models"]
    }
    primary = {}
    for spec in config["datasets"]:
        for seed_index, seed in enumerate(config["seeds"]):
            for track in config["tracks"]:
                reference = lookup.get((spec["id"], seed, track, "neurotabular210"))
                if reference is None:
                    continue
                for method_index, method in enumerate(
                    [
                        "neurotabular210",
                        *[m for m in config["methods"] if m != "neurotabular210"],
                    ]
                ):
                    alternative = lookup.get((spec["id"], seed, track, method))
                    if alternative is None:
                        continue
                    pair_id = f"{track}-{spec['id']}-{seed}-{method}"
                    pair = root / "measurements" / pair_id
                    receipt_path = pair / "PAIR.json"
                    if (pair / "PAIR_FAILURE.json").exists():
                        continue
                    if receipt_path.exists():
                        verify_pair(pair)
                    else:
                        host_memory_ready(config["rss_bytes"], processes=2)
                        from .work_counters import NEURAL, prediction_forward_limit
                        from .budget_contract import role_rows
                        rows, _ = role_rows(dict(input_file=reference["test_file"]))
                        one_forward_cap = prediction_forward_limit(rows, 1318 + int(spec["task"] != "regression"))
                        pair_forward_cap = one_forward_cap * (1 + int(method in NEURAL))
                        budget.reserve(
                            2 * (config["measurement_limits"]["cpu_s"] + 2),
                            config["measurement_limits"]["wall_s"] + 6,
                            256 * 1024**2,
                            measurement_workers=2, model_load_operations=2,
                            torch_root_forward_calls=pair_forward_cap,
                            prediction_API_calls=2
                            * (1318 + int(spec["task"] != "regression")),
                        )
                        descriptions = []
                        for which, model in enumerate((reference, alternative)):
                            job_id = "M-" + pair_id + "-" + str(which)
                            job = base_job(
                                config,
                                environment,
                                job_id,
                                model["method"],
                                spec["task"],
                                seed,
                                "serve",
                            )
                            job.update(
                                model_file=str(
                                    root / "jobs" / model["job_id"] / "model.joblib"
                                ),
                                model_sha256=model["model_sha256"],
                                input_file=model["test_file"],
                                input_sha256=model["test_sha256"],
                            )
                            from .budget_contract import attach_forward_limit
                            side_operations = dict(measurement_workers=1, model_load_operations=1,
                                prediction_API_calls=1318 + int(spec["task"] != "regression"))
                            attach_forward_limit(job, side_operations)
                            folder = pair / str(which)
                            folder.mkdir(parents=True, exist_ok=True)
                            if (folder / "JOB.json").exists():
                                raise RuntimeError(
                                    "Interrupted pair requires reconciliation; never relaunch"
                                )
                            atomic_json(folder / "JOB.json", job)
                            descriptions.append(
                                {
                                    "job_id": job_id,
                                    "command": command_for(
                                        environment, folder / "JOB.json"
                                    ),
                                    "model_sha256": job["model_sha256"],
                                    "input_sha256": job["input_sha256"],
                                    "operations": side_operations,
                                }
                            )
                        try:
                            limits = {
                                **config["measurement_limits"],
                                "operations": {
                                    "measurement_workers": 1,
                                    "prediction_API_calls": 1318
                                    + int(spec["task"] != "regression"),
                                },
                                "prediction_API_calls": 1318
                                + int(spec["task"] != "regression"),
                            }
                            run_pair(
                                descriptions,
                                pair,
                                budget.journal,
                                limits,
                                reverse=bool((seed_index + method_index) % 2),
                                environment=worker_environment(),
                            )
                        except Exception:
                            budget.checkpoint()
                            continue
                    if method == "neurotabular210" and not arrays_equal(
                        pair / "0/predictions.npz", pair / "1/predictions.npz"
                    ):
                        raise RuntimeError("A/A predictions differ")
                    for which, model in enumerate((reference, alternative)):
                        key = (
                            model["family"],
                            model["seed"],
                            model["track"],
                            model["method"],
                        )
                        path = pair / str(which) / "predictions.npz"
                        if key in primary and not arrays_equal(primary[key], path):
                            raise RuntimeError(
                                "Predictions changed between dedicated pairs"
                            )
                        primary.setdefault(key, path)
                    budget.checkpoint()
    output = []
    for key, path in primary.items():
        model = lookup[key]
        job = base_job(
            config,
            environment,
            "R-" + model["job_id"],
            model["method"],
            model["task"],
            model["seed"],
            "replay",
        )
        replay = cold_replay(
            root,
            job,
            root / "jobs" / model["job_id"],
            Path(model["test_file"]),
            path,
            config,
            environment,
            budget,
        )
        output.append(
            {
                **model,
                "predictions": str(path),
                "predictions_sha256": digest(path),
                "cold_replay_pass": not replay.get("failed", False),
            }
        )
    atomic_json(
        root / "PREDICTIONS_SEALED.json",
        {
            "models": output,
            "failures": seal["failures"],
            "models_seal_sha256": digest(root / "MODELS_SEALED.json"),
        },
    )


def validate_prepared_sources(
    data_root, config, pins_path=HERE / "prepared_sources.json"
):
    pinned = {row["id"]: row for row in load(pins_path)["sources"]}
    for spec in config["datasets"]:
        pin = pinned[spec["id"]]
        manifest = load(data_root / spec["id"] / "MANIFEST.json")
        if (
            manifest["source"]["sha256"] != pin["source_sha256"]
            or manifest["feature_columns"] != pin["features"]
        ):
            raise ValueError(
                "Prepared source identity/schema differs from the frozen public panel"
            )
        for seed in config["seeds"]:
            for name in ("fit", "validation", "pool", "test"):
                expected = pin["splits"][str(seed)]["roles"][name]
                recorded = manifest["splits"][str(seed)]["roles"][name]
                if (
                    recorded["sha256"] != expected["sha256"]
                    or recorded["rows"] != expected["rows"]
                    or digest(data_root / spec["id"] / str(seed) / (name + ".npz"))
                    != expected["sha256"]
                ):
                    raise ValueError(
                        "Prepared statistical role differs from its preregistered hash"
                    )


def execute(root, data_root, config, environment, config_path):
    import psutil

    psutil.Process().cpu_affinity([config["monitor_logical_cpu"]])
    validate_prepared_sources(data_root, config)
    if (
        root.exists()
        and any(root.iterdir())
        and not (root / "RUN_BINDING.json").exists()
    ):
        raise ValueError("A new benchmark requires an empty work directory")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve().is_relative_to(HERE.parents[1].resolve()):
        raise ValueError("Keep datasets/models/logs outside the public repository")
    binding = {
        "protocol_sha256": digest(config_path),
        "environment_sha256": digest(environment["_path"]),
        "data_manifests": {
            d["id"]: digest(data_root / d["id"] / "MANIFEST.json")
            for d in config["datasets"]
        },
        "runner_sources": {p.name: digest(p) for p in HERE.glob("*.py")},
    }
    binding.update(environment.get("_content_fields", {}))
    path = root / "RUN_BINDING.json"
    if path.exists():
        if load(path) != binding:
            raise RuntimeError("Frozen run changed; do not resume")
    else:
        atomic_json(path, binding)
    for directory in ("jobs", "selections", "measurements"):
        (root / directory).mkdir(exist_ok=True)
    budget = Budget(root, config["budget"])
    try:
        qualified = qualify(root, config, environment, budget)
        seal = train_all(root, data_root, config, environment, budget, qualified)
        measure_all(root, config, environment, budget, seal)
    finally:
        budget.checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "run"))
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        parser.error("Private0422 operations require the content-bound RUN_BOUND.py entrypoint")
    config = load(args.protocol)
    if args.action == "plan" or not args.execute:
        print(json.dumps(plan(config), indent=2))
        return
    if args.action == "prepare":
        if args.data is None:
            parser.error("prepare requires --data outside the repository")
        if args.data.resolve().is_relative_to(HERE.parents[1].resolve()):
            parser.error("Data must be outside the repository")
        for spec in config["datasets"]:
            prepare_family(spec, args.data / spec["id"], config["seeds"])
    else:
        if args.work is None or args.data is None or args.environment is None:
            parser.error("run requires --work, --data and --environment")
        environment = load(args.environment)
        environment["_path"] = str(args.environment.resolve())
        execute(
            args.work.resolve(), args.data.resolve(), config, environment, args.protocol
        )


if __name__ == "__main__":
    main()

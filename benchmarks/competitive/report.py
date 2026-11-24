"""Verify fixed predictions, retain failures, export tables and measured plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .control import Budget, atomic_json
from .data import digest
from .metrics import score
from .run import HERE, load, verify_pair

_SCORING_CPU_DEADLINE = None

def _check_score_deadline():
    import time
    if _SCORING_CPU_DEADLINE is not None and time.process_time() > _SCORING_CPU_DEADLINE:
        raise RuntimeError("SCORING_LOCAL_CPU_LIMIT")


def paired_interval(first, second, orders=None, seed=417, replicates=5000):
    """Geometric paired ratio, with order-stratified resampling of warm blocks."""
    import numpy as np

    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (
        len(first) != len(second)
        or len(first) < 4
        or (first <= 0).any()
        or (second <= 0).any()
    ):
        return {
            "status": "INCONCLUSIVE",
            "reason": "Insufficient blocks or clock quantization",
        }
    difference = np.log(second / first)
    order = np.zeros(len(first), dtype=int) if orders is None else np.asarray(orders)
    strata = [np.flatnonzero(order == value) for value in np.unique(order)]
    rng = np.random.default_rng(seed)
    samples = np.zeros(replicates)
    for indices in strata:
        selected = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        samples += difference[selected].sum(axis=1) / len(first)
    low, high = np.exp(np.quantile(samples, [0.025, 0.975]))
    return {
        "status": "MEASURED",
        "ratio_B_over_A": float(np.exp(difference.mean())),
        "ci95": [float(low), float(high)],
        "paired_blocks": len(first),
        "method": "order-stratified paired bootstrap of log ratios",
    }


def pair_summary(receipt, config, is_aa):
    results = {}
    for batch in (1, 32, "full"):
        subset = [
            [row for row in side if row["event"] == "BLOCK" and row["batch"] == batch]
            for side in receipt["records"]
        ]
        orders = [receipt["balanced_block_order"][row["round"]][0] for row in subset[0]]
        results[str(batch)] = {
            clock: paired_interval(
                [r[clock] for r in subset[0]],
                [r[clock] for r in subset[1]],
                orders,
                config["bootstrap_seed"],
                config["bootstrap_replicates"],
            )
            for clock in ("cpu_ns", "wall_ns")
        }
    valid = True
    if is_aa:
        low, high = config["AA_interval_band"]
        valid = all(
            value["status"] == "MEASURED"
            and low <= value["ci95"][0] <= value["ci95"][1] <= high
            for clocks in results.values()
            for value in clocks.values()
        )
    return {
        "A_A": is_aa,
        "status": "VALID" if valid else "INCONCLUSIVE",
        "blocks": results,
        "uncertainty_scope": "four repeated blocks per workload; conditional on this CPU/process pair",
    }


def quality_interval(
    task, y, values, classes, labels, units, seed=417, replicates=5000
):
    """Conditional unit-bootstrap intervals for primary loss/decision metrics.

    A single held-out simulation study has no independent-study interval.
    Seed results and overlapping held-out samples are never treated as extra
    independent families. AUC and macro-F1 remain point metrics in this pilot.
    """
    import numpy as np

    y, units = np.asarray(y), np.asarray(units)
    unique, inverse = np.unique(units, return_inverse=True)
    if len(unique) < 2:
        return {
            "status": "UNAVAILABLE",
            "reason": "Fewer than two independent declared units",
        }
    rng = np.random.default_rng(seed)
    if task == "regression":
        errors = (np.asarray(values).reshape(-1) - y.astype(float)) ** 2
        numerator = np.bincount(inverse, weights=errors)
        count = np.bincount(inverse)
        samples = []
        for _ in range(replicates):
            _check_score_deadline()
            chosen = rng.integers(0, len(unique), len(unique))
            samples.append(
                float(np.sqrt(numerator[chosen].sum() / count[chosen].sum()))
            )
        return {
            "status": "MEASURED",
            "rmse_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "units": len(unique),
        }
    lookup = {label: index for index, label in enumerate(classes)}
    positions = np.asarray([lookup[label] for label in y])
    losses = -np.log(np.clip(values[np.arange(len(y)), positions], 1e-15, 1))
    loss_sum = np.bincount(inverse, weights=losses)
    count = np.bincount(inverse)
    supports = np.zeros((len(unique), len(classes)))
    correct = np.zeros_like(supports)
    np.add.at(supports, (inverse, positions), 1)
    np.add.at(correct, (inverse, positions), np.asarray(labels) == y)
    nll_samples, ba_samples = [], []
    for _ in range(replicates):
        _check_score_deadline()
        chosen = rng.integers(0, len(unique), len(unique))
        nll_samples.append(float(loss_sum[chosen].sum() / count[chosen].sum()))
        support = supports[chosen].sum(axis=0)
        if (support > 0).all():
            ba_samples.append(float((correct[chosen].sum(axis=0) / support).mean()))
    return {
        "status": "MEASURED",
        "nll_ci95": np.quantile(nll_samples, [0.025, 0.975]).tolist(),
        "balanced_accuracy_ci95": np.quantile(ba_samples, [0.025, 0.975]).tolist()
        if len(ba_samples) >= 0.9 * replicates
        else None,
        "valid_balanced_bootstraps": len(ba_samples),
        "units": len(unique),
    }


def paired_quality(
    task, y, values, classes, labels, reference, units, replicates=5000, seed=417
):
    """Paired conditional intervals on the same declared TEST units."""
    import numpy as np

    unique, inverse = np.unique(units, return_inverse=True)
    if len(unique) < 2:
        return {
            "status": "UNAVAILABLE",
            "reason": "Fewer than two declared independent units",
        }
    counts = np.bincount(inverse)
    if task == "regression":
        first = np.bincount(inverse, weights=(np.asarray(values).reshape(-1) - y) ** 2)
        second = np.bincount(
            inverse, weights=(reference["values"].reshape(-1) - y) ** 2
        )
    else:
        first_lookup = {label: i for i, label in enumerate(classes)}
        second_lookup = {
            label: i for i, label in enumerate(reference["classes"].tolist())
        }
        first_positions = np.asarray([first_lookup[label] for label in y])
        second_positions = np.asarray([second_lookup[label] for label in y])
        first = -np.log(np.clip(values[np.arange(len(y)), first_positions], 1e-15, 1))
        second = -np.log(
            np.clip(reference["values"][np.arange(len(y)), second_positions], 1e-15, 1)
        )
        delta = np.bincount(inverse, weights=first - second)
        supports = np.zeros((len(unique), len(classes)))
        correct_delta = np.zeros_like(supports)
        np.add.at(supports, (inverse, first_positions), 1)
        np.add.at(
            correct_delta,
            (inverse, first_positions),
            (np.asarray(labels) == y).astype(float)
            - (reference["labels"] == y).astype(float),
        )
    rng = np.random.default_rng(seed)
    primary, decisions = [], []
    for _ in range(replicates):
        _check_score_deadline()
        chosen = rng.integers(0, len(unique), len(unique))
        if task == "regression":
            a, b = first[chosen].sum(), second[chosen].sum()
            if b > 0:
                primary.append(float(np.sqrt(a / b)))
        else:
            primary.append(float(delta[chosen].sum() / counts[chosen].sum()))
            support = supports[chosen].sum(axis=0)
            if (support > 0).all():
                decisions.append(
                    float((correct_delta[chosen].sum(axis=0) / support).mean())
                )
    return {
        "status": "MEASURED" if primary else "UNAVAILABLE",
        "reference": "neurotabular210",
        "primary": "rmse_ratio" if task == "regression" else "nll_difference",
        "ci95": np.quantile(primary, [0.025, 0.975]).tolist() if primary else None,
        "balanced_accuracy_difference_ci95": np.quantile(
            decisions, [0.025, 0.975]
        ).tolist()
        if len(decisions) >= 0.9 * replicates
        else None,
        "scope": "conditional on frozen models and declared TEST units; not independent source-family uncertainty",
    }


def score_model(model, reference=None):
    import numpy as np

    if (
        digest(model["predictions"]) != model["predictions_sha256"]
        or digest(model["test_file"]) != model["test_sha256"]
    ):
        raise ValueError("Changed predictions or TEST rows")
    with (
        np.load(model["test_file"], allow_pickle=False) as source,
        np.load(model["predictions"], allow_pickle=False) as predicted,
    ):
        if not np.array_equal(source["groups"], predicted["groups"]):
            raise ValueError("Misaligned TEST groups")
        y, values, classes, labels = (
            source["y"],
            predicted["values"],
            predicted["classes"].tolist(),
            predicted["labels"],
        )
        metrics = score(model["task"], y, values, classes, native_labels=labels)
        interval = quality_interval(
            model["task"], y, values, classes, labels, source["units"]
        )
        paired = {"status": "UNAVAILABLE_REFERENCE"}
        reference_labels = None
        if reference is not None and reference["cold_replay_pass"]:
            if (
                reference["test_sha256"] != model["test_sha256"]
                or digest(reference["predictions"]) != reference["predictions_sha256"]
            ):
                raise ValueError("Paired baseline is not the same sealed TEST")
            with np.load(reference["predictions"], allow_pickle=False) as baseline:
                reference_labels = baseline["labels"].copy()
                paired = paired_quality(
                    model["task"], y, values, classes, labels, baseline, source["units"]
                )
        if model["task"] != "regression":
            from .supplementary_metrics import classification_details, average_precision
            # Explicit frozen label rule, independent of probability-column
            # order and TEST outcomes: greatest Unicode TRAIN class label.
            positive_label = max(classes, key=str) if model["task"] == "binary" else None
            details = classification_details(y.tolist(), values.tolist(), classes,
                native_labels=labels.tolist(), positive_label=positive_label,
                reference_native_labels=None if reference_labels is None else reference_labels.tolist())
            details["uncertainty"] = "POINT_ONLY_DESCRIPTIVE_NO_AP_OR_PER_CLASS_CI"
            details["positive_label_rule"] = "MAX_UNICODE_TRAIN_CLASS_LABEL_NOT_COLUMN_ORDER" if model["task"] == "binary" else None
            metrics["average_precision"] = details["average_precision"]
            if model["task"] == "binary":
                details["average_precision_by_class"] = [dict(label=label,
                    support=sum(t == label for t in y.tolist()),
                    average_precision=average_precision([t == label for t in y.tolist()],
                        [row[index] for row in values.tolist()])) for index, label in enumerate(classes)]
            metrics["classification_details"] = details
    return metrics, interval, paired


def counter_profile(folder):
    """Read raw receipts only; missing/partial work is never fabricated as zero."""
    complete, partial = folder / "RESULT.json", folder / "COUNTERS_PARTIAL.json"
    source = complete if complete.exists() else partial
    if not source.exists():
        return {"status": "NO_OBSERVED_COUNTER_RECEIPT", "forward_calls": None}
    value = load(source)
    tree = value.get("forward_applicability") == "NOT_APPLICABLE_TREE"
    return dict(status="COMPLETE" if source == complete else "PARTIAL_OBSERVATION",
        forward_applicability=value.get("forward_applicability"),
        forward_calls=None if tree else value.get("torch_root_forward_calls"),
        forward_root_classes=value.get("forward_root_classes", {}),
        optimizer_steps=value.get("optimizer_steps"),
        fit_calls_started=value.get("fit_calls_started"),
        fit_calls_completed=value.get("fit_calls_completed"),
        model_load_calls_started=value.get("model_load_calls_started"),
        model_load_operations=value.get("model_load_operations"),
        blocked_refit_requests=value.get("blocked_refit_requests"),
        scope="Observed counters only; reservations and failed-ticket charges are separate")


def compile_results(root, config, budget, config_path=HERE / "protocol.json"):
    seal = load(root / "PREDICTIONS_SEALED.json")
    if load(root / "RUN_BINDING.json")["runner_sources"] != {
        p.name: digest(p) for p in HERE.glob("*.py")
    }:
        raise ValueError("Frozen model/scoring implementation changed")
    if load(root / "RUN_BINDING.json")["protocol_sha256"] != digest(config_path):
        raise ValueError("Scoring protocol differs from the frozen run")
    if seal["models_seal_sha256"] != digest(root / "MODELS_SEALED.json"):
        raise ValueError("Changed pre-TEST model seal")
    metrology = {}
    for pair in sorted((root / "measurements").iterdir()):
        if (pair / "PAIR.json").exists():
            receipt = verify_pair(pair)
            metrology[pair.name] = pair_summary(
                receipt, config, pair.name.endswith("-neurotabular210")
            )
        elif (pair / "PAIR_FAILURE.json").exists():
            metrology[pair.name] = {
                "status": "FAILED",
                "failure": load(pair / "PAIR_FAILURE.json"),
            }
    records = []
    references = {
        (m["family"], m["seed"], m["track"]): m
        for m in seal["models"]
        if m["method"] == "neurotabular210"
    }
    for model in seal["models"]:
        if not model["cold_replay_pass"]:
            records.append(
                {
                    **{
                        k: model[k]
                        for k in ("family", "seed", "task", "track", "method")
                    },
                    "status": "FAILED_COLD_REPLAY",
                    "failure_stage": "COLD_REPLAY_IDENTITY_OR_RESOURCE_STOP",
                }
            )
            continue
        job_id = "S-" + model["job_id"]
        folder = root / "scores" / job_id
        if folder.exists():
            if not (folder / "SCORE.json").exists():
                raise RuntimeError(
                    "Interrupted scoring requires reconciliation, no retry"
                )
            record = load(folder / "SCORE.json")
            if record["predictions_sha256"] != model["predictions_sha256"]:
                raise ValueError("Changed scored predictions")
            events = [row for row in budget.journal.events() if row["job_id"] == job_id]
            if len(events) != 2 or events[1]["event"] != "COMPLETED":
                raise RuntimeError(
                    "Atomic score is retained but its journal needs reconciliation; never rescore automatically"
                )
        else:
            budget.reserve(20, 40, 1024**2, scoring_test=1)
            folder.mkdir(parents=True)
            budget.journal.start(
                job_id, {"cpu_s": 20, "wall_s": 40, "operations": {"scoring_test": 1}}
            )
            import time

            global _SCORING_CPU_DEADLINE
            started = time.process_time()
            _SCORING_CPU_DEADLINE = started + 19
            metrics, interval, paired = score_model(
                model, references.get((model["family"], model["seed"], model["track"]))
            )
            _check_score_deadline()
            _SCORING_CPU_DEADLINE = None
            fit = load(root / "jobs" / model["job_id"] / "RESULT.json")
            terminal = load(root / "jobs" / model["job_id"] / "TERMINAL.json")
            pair_id = (
                f"{model['track']}-{model['family']}-{model['seed']}-{model['method']}"
            )
            pair = load(root / "measurements" / pair_id / "PAIR.json")
            aa_id = (
                f"{model['track']}-{model['family']}-{model['seed']}-neurotabular210"
            )
            side = 0 if model["method"] == "neurotabular210" else 1
            blocks = [
                row
                for row in pair["records"][side]
                if row["event"] == "BLOCK" and row["batch"] == "full"
            ]
            import statistics

            inference_wall = statistics.median(
                row["wall_ns"] / row["calls"] for row in blocks
            )
            inference_cpu = statistics.median(
                row["cpu_ns"] / row["calls"] for row in blocks
            )
            inference_rss = next(
                row["peak_rss_bytes"] for row in pair["costs"] if row["which"] == side
            )
            cold = next(row for row in pair["records"][side] if row["event"] == "COLD")
            record = {
                **{k: model[k] for k in ("family", "seed", "task", "track", "method")},
                "status": "MEASURED",
                "metrics": metrics,
                "quality_uncertainty": interval,
                "paired_quality_uncertainty": paired,
                "fit_cpu_s": terminal["cpu_s"],
                "fit_wall_s": terminal["wall_s"],
                "fit_worker_cpu_s": fit["fit_cpu_s"],
                "fit_compute_cpu_s": fit["fit_compute_cpu_s"],
                "tuning_cpu_s": model["tuning_cpu_s"],
                "tuning_wall_s": model["tuning_wall_s"],
                "training_total_cpu_s": terminal["cpu_s"] + model["tuning_cpu_s"],
                "training_total_wall_s": terminal["wall_s"] + model["tuning_wall_s"],
                "fit_peak_rss_bytes": terminal["peak_rss_bytes"],
                "model_bytes": fit["model_bytes"],
                "warm_full_predict_wall_ns": inference_wall,
                "warm_full_predict_cpu_ns": inference_cpu,
                "cold_predict_wall_ns": cold["wall_ns"],
                "inference_peak_rss_bytes": inference_rss,
                "inference_A_A_status": metrology.get(aa_id, {}).get(
                    "status", "UNAVAILABLE"
                ),
                "public_constructor": fit["constructor"],
                "predictions_sha256": model["predictions_sha256"],
                "model_sha256": model["model_sha256"],
                "test_sha256": model["test_sha256"],
                "class_order": fit["class_order"],
                "fit_torch_root_forward_calls": fit.get("torch_root_forward_calls") if fit.get("forward_applicability") != "NOT_APPLICABLE_TREE" else None,
                "forward_scope": fit.get("forward_scope"),
                "forward_applicability": fit.get("forward_applicability"),
                "fit_optimizer_steps": fit.get("optimizer_steps"),
                "internal_refits": fit.get("internal_refits"),
                "operation_profiles": {
                    "final_fit": counter_profile(root / "jobs" / model["job_id"]),
                    "cold_replay": counter_profile(root / "jobs" / ("R-" + model["job_id"])),
                    "reported_measurement_resident": counter_profile(root / "measurements" / pair_id / str(side)),
                    "measurement_instrumentation": "Learned-entry wrapper is active inside timing; nonzero method-specific overhead, not equal FLOPs",
                },
            }
            if model["track"] == "hpo4":
                prefix = model["job_id"].removesuffix("-final")
                selection = load(root / "selections" / (prefix + ".json"))
                record["hpo_complete_search"] = selection["complete_search"]
                record["hpo_selected_index"] = selection["selected_index"]
                record["operation_profiles"]["tuning_trials"] = [
                    counter_profile(root / "jobs" / (prefix + "-trial" + str(index)))
                    for index in range(len(config["hpo_candidates"][model["method"]]))
                ]
            atomic_json(folder / "SCORE.json", record)
            # Scoring is inside this controller, already included in its CPU.
            budget.journal.finish(
                job_id,
                "COMPLETED",
                {
                    "cpu_s": 0,
                    "scoring_test": 1,
                    "controller_cpu_measured_s": time.process_time() - started,
                },
            )
            budget.checkpoint()
        records.append(record)
    for failed in seal["failures"]:
        records.append(
            {
                **{
                    k: failed.get(k)
                    for k in ("family", "seed", "task", "track", "method")
                },
                "status": "FAILED",
                "reason": failed["reason"],
            }
        )
    # Missing measurements remain visible, including methods whose fit succeeded
    # but whose inference exhausted its declared resource budget.
    present = {(r["family"], r["seed"], r["track"], r["method"]) for r in records}
    for model in load(root / "MODELS_SEALED.json")["models"]:
        key = tuple(model[k] for k in ("family", "seed", "track", "method"))
        if key not in present:
            records.append(
                {
                    **{
                        k: model[k]
                        for k in ("family", "seed", "task", "track", "method")
                    },
                    "status": "FAILED_MEASUREMENT",
                    "failure_stage": "METROLOGY_INCOMPLETE_OR_RESOURCE_STOP",
                }
            )
    return {
        "schema": "NT_PUBLIC_RESULTS/1",
        "synthetic": any(
            r.get("public_constructor", {}).get("SYNTHETIC_PROTOCOL_FIXTURE", False)
            for r in records
        ),
        "protocol_sha256": digest(config_path),
        "run_binding_sha256": digest(root / "RUN_BINDING.json"),
        "records": records,
        "metrology": metrology,
        "claims": "Prospective six-family CPU comparison. No universal superiority or independent recipe confirmation.",
        "all_failures_retained": True,
    }


def descriptive_summary(records):
    import math
    import statistics

    measured = [
        r
        for r in records
        if r["status"] == "MEASURED" and r.get("hpo_complete_search", True)
    ]
    lookup = {(r["family"], r["seed"], r["track"], r["method"]): r for r in measured}
    rows = []
    for track, task, method in sorted(
        {(r["track"], r["task"], r["method"]) for r in measured}
    ):
        families = {
            r["family"] for r in measured if r["track"] == track and r["task"] == task
        }
        family_values = []
        pairs = 0
        for family in sorted(families):
            paired = []
            for (source, seed, source_track, source_method), row in lookup.items():
                if (source, source_track, source_method) != (family, track, method):
                    continue
                baseline = lookup.get((source, seed, track, "neurotabular210"))
                if baseline is None:
                    continue
                metric = "rmse" if task == "regression" else "nll"
                if baseline["metrics"][metric] <= 0 or row["metrics"][metric] <= 0:
                    continue
                paired.append(
                    (
                        math.log(row["metrics"][metric] / baseline["metrics"][metric]),
                        None
                        if task == "regression"
                        else row["metrics"]["balanced_accuracy"]
                        - baseline["metrics"]["balanced_accuracy"],
                    )
                )
            pairs += len(paired)
            if len(paired) == 3:
                family_values.append(
                    (
                        statistics.median(item[0] for item in paired),
                        None
                        if task == "regression"
                        else statistics.median(item[1] for item in paired),
                    )
                )
        rows.append(
            {
                "track": track,
                "task": task,
                "method": method,
                "paired_cells": pairs,
                "complete_families": len(family_values),
                "geometric_loss_ratio_to_nt210": math.exp(
                    statistics.mean(item[0] for item in family_values)
                )
                if family_values
                else None,
                "mean_family_balanced_accuracy_delta": statistics.mean(
                    item[1] for item in family_values
                )
                if family_values and task != "regression"
                else None,
                "scope": "descriptive only; median of three seeds, equal family weights, at most two families per task",
            }
        )
    task_families = {task: {r["family"] for r in records if r["task"] == task}
                     for task in ("binary", "multiclass", "regression")}
    for row in rows:
        complete = row["complete_families"] == len(task_families[row["task"]]) == 2 and row["paired_cells"] == 6
        row["aggregate_ranking_allowed"] = complete
        if not complete:
            for name in list(row):
                if name.startswith("geometric_") or name.startswith("mean_family_"):
                    row[name] = None
            row["incompleteness_reason"] = "Full two-family, three-seed paired panel required; no surviving-subset ranking"
    return rows


def decision_support(records):
    """Pure descriptive comparisons/frontiers; no new selection or model call."""
    import math
    lookup = {(r['family'], r['seed'], r['track'], r['method']): r for r in records}
    costs = ('training_total_cpu_s', 'training_total_wall_s', 'fit_peak_rss_bytes',
             'inference_peak_rss_bytes', 'warm_full_predict_wall_ns')
    comparisons, frontiers = [], []
    for row in records:
        if row['method'] == 'neurotabular210':
            continue
        baseline = lookup.get((row['family'], row['seed'], row['track'], 'neurotabular210'))
        item = {k:row[k] for k in ('family','seed','track','task','method')}
        item['primary'] = 'rmse' if row['task'] == 'regression' else 'nll'
        valid = row['status'] == 'MEASURED' and baseline is not None and baseline['status'] == 'MEASURED'
        full = valid and (row['track'] != 'hpo4' or
                         all(r.get('hpo_complete_search') is True for r in (row, baseline)))
        item['eligible_for_paired_claim'] = bool(full)
        item['status'] = 'INCONCLUSIVE_OR_MISSING_PRIMARY_INTERVAL'
        if full:
            interval = row.get('paired_quality_uncertainty', {})
            bounds = interval.get('ci95')
            null = 1.0 if row['task'] == 'regression' else 0.0
            if interval.get('status') == 'MEASURED' and bounds is not None:
                item['primary_ci95'] = bounds
                if bounds[0] > null:
                    item['status'] = 'NT210_BETTER_PRIMARY_CONDITIONAL_INTERVAL'
                elif bounds[1] < null:
                    item['status'] = 'COMPETITOR_BETTER_PRIMARY_CONDITIONAL_INTERVAL'
                else:
                    item['status'] = 'INCONCLUSIVE_PRIMARY_CONDITIONAL_INTERVAL'
            item['cost_ratios_to_nt210_point_only'] = {
                cost:row[cost]/baseline[cost]
                for cost in costs
                if row.get(cost) is not None and baseline.get(cost,0) > 0 and
                (cost != 'warm_full_predict_wall_ns' or
                 all(r.get('inference_A_A_status') == 'VALID' for r in (row,baseline)))}
            if row['task'] != 'regression':
                item['balanced_accuracy_delta_point_only'] = row['metrics']['balanced_accuracy'] - baseline['metrics']['balanced_accuracy']
        else:
            item['reason'] = 'Missing/failed pair or incomplete hpo4 remains visible; no winner inferred'
        comparisons.append(item)
    cells = sorted({(r['family'],r['seed'],r['track'],r['task']) for r in records})
    for family, seed, track, task in cells:
        loss = 'rmse' if task == 'regression' else 'nll'
        eligible = [r for r in records if (r['family'],r['seed'],r['track']) == (family,seed,track)
            and r['status'] == 'MEASURED' and (track != 'hpo4' or r.get('hpo_complete_search') is True)]
        for cost in costs:
            observed = [r for r in eligible if isinstance(r.get(cost),(int,float)) and
                math.isfinite(r[cost]) and r[cost] >= 0 and
                (cost != 'warm_full_predict_wall_ns' or r.get('inference_A_A_status') == 'VALID')]
            front = [r['method'] for r in observed if not any(
                other['metrics'][loss] <= r['metrics'][loss] and other[cost] <= r[cost]
                and (other['metrics'][loss] < r['metrics'][loss] or other[cost] < r[cost])
                for other in observed)]
            frontiers.append(dict(family=family,seed=seed,track=track,task=task,loss=loss,cost=cost,
                observed_methods=[r['method'] for r in observed],frontier_methods=front,
                scope='Local descriptive observed-method point frontier, not a confidence-qualified or global rank'))
    return dict(comparisons=comparisons,pareto_frontiers=frontiers,
        uncertainty='Primary intervals are conditional on frozen models/TEST units, no multiplicity-adjusted universal claim',
        failures_and_incomplete_searches_are_not_wins=True, AP_and_costs_are_point_only=True)


def write_report(results, output, budget=None):
    """Only sanitized numerical results; no raw rows, weights or private receipts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if results.get("synthetic") and output.resolve().is_relative_to(
        HERE.parents[1].resolve()
    ):
        raise ValueError(
            "Synthetic fixtures cannot be exported as public benchmark results"
        )
    output.mkdir(parents=True, exist_ok=True)
    if budget is not None:
        budget.reserve(0, 0, 64*1024**2)
    atomic_json(output / "results.json", results)
    atomic_json(output / "summary.json", descriptive_summary(results["records"]))
    decisions = decision_support(results['records'])
    atomic_json(output / 'decision_support.json', decisions)
    guide = ['# When to choose NeuroTabular', '',
        'Use the following paired pilot evidence for the named family, seed and budget regime only.',
        'The CPU profile is instrumented. Default and hpo4 are separate; equal fit counts are not equal compute.',
        'Classification primary evidence is NLL, not an assertion of higher accuracy. Regression uses RMSE.',
        'Costs, AP, per-class metrics and Pareto membership are descriptive point values; no universal rank follows.',
        'Warm latency ratios require valid A/A controls. Failures, missing pairs and incomplete tuning are inconclusive.', '',
        '| Family | Seed | Regime | Competitor | Primary evidence versus NT210 |',
        '|---|---:|---|---|---|']
    for item in decisions['comparisons']:
        guide.append(f"| {item['family']} | {item['seed']} | {item['track']} | {item['method']} | {item['status']} |")
    guide += ['', 'For a quality-first decision, inspect the named primary interval together with balanced accuracy, AUC, macro metrics and regressions.',
        'For a constrained deployment, inspect the local quality/CPU, quality/wall, fit-RAM, inference-RAM and valid warm-latency frontiers in decision_support.json.',
        'A point on a local frontier may still be statistically indistinguishable or unsuitable for another family.',
        'The API remains one final model for NeuroTabular. Check each competitor constructor and tuning costs before comparing ease or expense.',
        'No inference is drawn from a failed competitor as if it were a zero-quality or infinite-cost measured model.']
    (output / 'WHEN_TO_CHOOSE.md').write_text('\n'.join(guide)+'\n', encoding='utf-8')
    rows = []
    for record in results["records"]:
        rows.append(
            {
                **{
                    k: v
                    for k, v in record.items()
                    if k
                    not in {
                        "metrics",
                        "quality_uncertainty",
                        "paired_quality_uncertainty",
                        "class_order",
                        "public_constructor",
                    }
                },
                **record.get("metrics", {}),
            }
        )
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    title = (
        "SYNTHETIC TEST — not benchmark evidence"
        if results.get("synthetic")
        else "Measured public benchmark"
    )
    lines = [
        "# " + title,
        "",
        "Default and tuning are separate. All resource failures remain in the results.",
        "",
        "| Family | Seed | Track | Method | Status | Balanced accuracy / RMSE | NLL | Total CPU (s) | Fit RSS (MiB) |",
        "|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = (
            row.get("rmse")
            if row.get("task") == "regression"
            else row.get("balanced_accuracy")
        )

        def number(value):
            return "—" if value is None else f"{value:.5g}"

        lines.append(
            f"| {row['family']} | {row['seed']} | {row['track'] + (' (incomplete search)' if row.get('hpo_complete_search') is False else '')} | {row['method']} | {row['status']} | {number(metric)} | {number(row.get('nll'))} | {number(row.get('training_total_cpu_s'))} | {number(row.get('fit_peak_rss_bytes', 0) / 1024**2) if row.get('fit_peak_rss_bytes') else '—'} |"
        )
    lines += [
        "",
        "Intervals and all point metrics are in `results.json`. The two families per task support a limited pilot, not a global ranking.",
        "Warm inference claims require a valid dedicated A/A control; cold timing and fit cost are separate measurements.",
        "The three seeds are repetitions, not three independent dataset families.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frontier_keys = {(r['family'],r['seed'],r['track'],method,r['loss'],r['cost'])
                     for r in decisions['pareto_frontiers'] for method in r['frontier_methods']}
    for track in ("default", "hpo4"):
        for task in ("binary", "multiclass", "regression"):
            selected = [
                r
                for r in results["records"]
                if r["status"] == "MEASURED"
                and r["track"] == track
                and r["task"] == task
            ]
            if not selected:
                continue
            primary = "rmse" if task == "regression" else "balanced_accuracy"
            cost_specs = (
                ("training_total_cpu_s", "Total training CPU (s)"),
                ("training_total_wall_s", "Total training wall (s)"),
                ("fit_peak_rss_bytes", "Peak fit RSS (MiB)"),
                ("inference_peak_rss_bytes", "Peak inference RSS (MiB)"),
                (
                    "warm_full_predict_wall_ns",
                    "Warm full TEST predict (ms); A/A validity in JSON",
                ),
            )
            for primary, cost, label in [(metric,cost,label)
                    for metric in ([primary] if task == 'regression' else [primary,'nll'])
                    for cost,label in cost_specs]:
                if budget is not None:
                    # Fixed150dpi, at most2 panels: uncompressed RGBA <6MiB;
                    #8MiB headroom before rendering covers each PNG plus headers.
                    budget.reserve(0, 0, 8*1024**2)
                families = sorted({r["family"] for r in selected})
                fig, axes = plt.subplots(
                    1,
                    len(families),
                    figsize=(6 * len(families), 5),
                    squeeze=False,
                    layout="constrained",
                )
                for axis, family in zip(axes[0], families):
                    for method in sorted({r["method"] for r in selected}):
                        subset = [
                            r
                            for r in selected
                            if r["method"] == method and r["family"] == family
                        ]
                        method_label = method + (' — contains incomplete hpo4' if any(r.get('hpo_complete_search') is False for r in subset) else '')
                        divisor = (
                            1024**2
                            if cost in {"fit_peak_rss_bytes", "inference_peak_rss_bytes"}
                            else 1e6
                            if cost == "warm_full_predict_wall_ns"
                            else 1
                        )
                        if cost == "warm_full_predict_wall_ns":
                            for valid in (True, False):
                                part = [r for r in subset if (r.get("inference_A_A_status") == "VALID") == valid]
                                if not part:
                                    continue
                                axis.scatter([r[cost] / divisor for r in part], [r["metrics"][primary] for r in part],
                                    label=method_label + ("" if valid else " — A/A INCONCLUSIVE"), alpha=.8,
                                    **({} if valid else {"facecolors": "none", "edgecolors": "gray"}))
                        else:
                            x = [r[cost] / divisor for r in subset]
                            axis.scatter(x, [r["metrics"][primary] for r in subset], label=method_label, alpha=.8)
                        frontier = [r for r in subset if (r['family'],r['seed'],r['track'],method,primary,cost) in frontier_keys]
                        if frontier:
                            axis.scatter([r[cost]/divisor for r in frontier], [r['metrics'][primary] for r in frontier],
                                marker='*',s=130,facecolors='none',edgecolors='black',label='_local point frontier')
                    axis.set(xlabel=label, ylabel=primary, title=family)
                    axis.legend(fontsize=7, ncol=2)
                    axis.grid(alpha=0.2)
                fig.suptitle(f"{title}: {task}, {track}", fontsize=10)
                fig.savefig(output / f"{track}-{task}-{primary}-{cost}.png", dpi=150)
                plt.close(fig)
                if budget is not None:
                    budget.reserve(0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    args = parser.parse_args()
    parser.error("Private0422 scoring/rendering requires RUN_BOUND.py --execute --report and the authenticated receipt gate")
    if not args.output.resolve().is_relative_to(args.work.resolve()):
        parser.error(
            "Keep the generated report inside --work so its archives are included in the run budget"
        )
    config = load(args.protocol)
    budget = Budget(args.work, config["budget"])
    try:
        results = compile_results(args.work, config, budget, args.protocol)
        write_report(results, args.output)
    finally:
        budget.checkpoint()


if __name__ == "__main__":
    main()

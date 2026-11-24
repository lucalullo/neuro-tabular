"""Fit-free mathematical, split, adapter and actual process-controller checks."""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks.competitive.adapters import (
    FeatureAdapter,
    category_token,
    estimator_spec,
)
from benchmarks.competitive.control import (
    Budget,
    Journal,
    archive_bytes,
    atomic_json,
    launch,
)
from benchmarks.competitive.data import (
    linked_groups,
    make_roles,
    read_bundle,
    write_bundle,
)
from benchmarks.competitive.metrics import score
from benchmarks.competitive.metrology import run_pair
from benchmarks.competitive.report import (
    paired_interval,
    paired_quality,
    quality_interval,
)
from benchmarks.competitive.run import choose_trial, plan, worker_environment

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkMathTests(unittest.TestCase):
    def test_class_order_and_native_tie_decision(self):
        y = np.asarray(["a", "b", "a", "b"])
        probability = np.asarray([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        result = score("binary", y, probability, ["b", "a"], native_labels=y)
        self.assertEqual(result["auc"], 1)
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["nll"], -np.log([0.9, 0.9, 0.8, 0.8]).mean())
        tied = score(
            "binary",
            ["a", "b"],
            [[0.5, 0.5], [0.5, 0.5]],
            ["a", "b"],
            native_labels=["a", "b"],
        )
        self.assertEqual(tied["accuracy"], 1)
        self.assertEqual(tied["auc"], 0.5)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            score("binary", ["a"], [[1.1, -0.1]], ["a", "b"])
        with self.assertRaises(ValueError):
            score("binary", ["absent"], [[0.5, 0.5]], ["a", "b"])

    def test_missing_single_unit_has_no_invented_interval(self):
        result = quality_interval(
            "regression",
            np.arange(3),
            np.arange(3),
            [],
            None,
            np.zeros(3),
            replicates=20,
        )
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_paired_cost_resamples_order_and_rejects_zero_clock(self):
        result = paired_interval(
            [1, 2, 4, 8], [2, 4, 8, 16], [0, 1, 0, 1], replicates=100
        )
        self.assertAlmostEqual(result["ratio_B_over_A"], 2)
        np.testing.assert_allclose(result["ci95"], [2, 2])
        self.assertEqual(paired_interval([0] * 4, [1] * 4)["status"], "INCONCLUSIVE")

    def test_paired_quality_uses_same_units_and_class_columns(self):
        y = np.asarray(["a", "b", "a", "b"])
        values = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        reference = {
            "values": np.asarray([[0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]]),
            "classes": np.asarray(["b", "a"]),
            "labels": y,
        }
        result = paired_quality(
            "binary",
            y,
            values,
            ["a", "b"],
            y,
            reference,
            np.repeat(np.arange(2), 2),
            replicates=100,
        )
        np.testing.assert_allclose(result["ci95"], [np.log(0.8 / 0.9)] * 2)
        np.testing.assert_allclose(result["balanced_accuracy_difference_ci95"], [0, 0])

    def test_validation_selection_never_accepts_test_only_metrics(self):
        trials = [
            {"test_metrics": {"nll": 0}},
            {"validation_metrics": {"nll": 2}},
            {"validation_metrics": {"nll": 2}},
        ]
        self.assertEqual(choose_trial(trials, "nll"), 1)
        self.assertIsNone(choose_trial([None, {"test_metrics": {"nll": 0}}], "nll"))

    def test_frozen_plan_counts(self):
        config = json.loads((ROOT / "benchmarks/competitive/protocol.json").read_text())
        result = plan(config)
        self.assertEqual(len(result["cases"]), 144)
        self.assertEqual(sum(row["hpo_trials"] for row in result["cases"]), 576)
        self.assertFalse(result["real_execution"])


class AdapterTests(unittest.TestCase):
    def test_native_constructor_has_only_seed(self):
        module, name, kwargs = estimator_spec("neurotabular210", "binary", 42)
        self.assertEqual(
            (module, name, kwargs),
            ("neurotabular", "NeuroTabularClassifier", {"random_state": 42}),
        )

    def test_resources_cannot_be_tuned(self):
        with self.assertRaises(ValueError):
            estimator_spec("realmlp", "binary", 42, {"n_refit": 1})

    def test_imputation_and_vocabulary_use_only_train(self):
        X = pd.DataFrame({"n": [1.0, 3.0, np.nan], "cat": ["known", np.nan, "known"]})
        future = pd.DataFrame({"n": [np.nan, 999.0], "cat": ["new", "known"]})
        numeric = FeatureAdapter("realmlp", ["cat"]).fit(X).transform(future)
        self.assertEqual(numeric.loc[0, "n"], 2)
        for method in ("xgboost", "lightgbm"):
            with self.subTest(method=method):
                adapter = FeatureAdapter(method, ["cat"]).fit(X)
                categorical = adapter.transform(future)
                self.assertTrue(pd.isna(categorical.loc[0, "cat"]))
                self.assertEqual(categorical.loc[1, "cat"], "str:known")
                self.assertEqual(list(categorical["cat"].cat.categories), ["str:known"])
                np.testing.assert_array_equal(categorical["cat"].cat.codes, [-1, 0])
                fitted = adapter.transform(X)
                np.testing.assert_array_equal(fitted["cat"].cat.codes, [0, -1, 0])
                self.assertEqual(adapter.vocabulary_["cat"], ["str:known"])
        self.assertNotEqual(category_token(1), category_token("1"))
        self.assertNotEqual(category_token(np.nan), category_token("missing:"))
        native = FeatureAdapter("neurotabular210", ["cat"]).fit(X)
        self.assertIs(native.transform(future), future)


class SplitTests(unittest.TestCase):
    def test_declared_groups_and_duplicates_are_transitive(self):
        X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "cat": ["x", "x", "y", np.nan]})
        groups = linked_groups(X, [10, 20, 20, 30])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_temporal_purge_never_moves_future_rows_into_train(self):
        X = pd.DataFrame({"a": np.arange(30, dtype=float)})
        X.loc[0, "a"] = X.loc[25, "a"]
        y = pd.Series(["a", "b"] * 15)
        groups = linked_groups(X)
        roles, purged = make_roles(
            X, y, groups, X, {"split_policy": "ordered_contacts", "task": "binary"}, 42
        )
        self.assertIn(0, purged)
        self.assertIn(25, roles["test"])
        self.assertLess(max(roles["fit"]), 18)
        self.assertFalse(set(groups[roles["pool"]]) & set(groups[roles["test"]]))

    def test_blind_bundle_reader_does_not_open_target(self):
        X = pd.DataFrame({"a": [1.0, np.nan], "cat": ["known", np.nan]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.npz"
            write_bundle(
                path, X, pd.Series(["a", "b"]), np.arange(2), np.arange(2), ["cat"]
            )
            # Rewrite the NPZ without y. The blind path must still work.
            with np.load(path, allow_pickle=False) as original:
                saved = {key: original[key] for key in original.files if key != "y"}
            with path.open("wb") as stream:
                np.savez(stream, **saved)
            actual, target, _, _ = read_bundle(path, include_target=False)
            self.assertIsNone(target)
            pd.testing.assert_frame_equal(actual, X)


class ControllerTests(unittest.TestCase):
    def test_insufficient_host_memory_stops_before_any_dispatch(self):
        from types import SimpleNamespace

        from benchmarks.competitive.control import host_memory_ready

        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1)):
            with self.assertRaisesRegex(RuntimeError, "PAUSED_BEFORE_DISPATCH"):
                host_memory_ready(1024**3, processes=2)

    def test_collision_is_forced_between_enumeration_and_stat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "payload.partial", root / "payload.json"
            old.write_bytes(b"x" * 937)
            seen = threading.Event()
            renamed = threading.Event()
            real_scandir = os.scandir

            class Entry:
                def __init__(self, entry):
                    self.entry, self.path = entry, entry.path

                def stat(self, **kwargs):
                    if self.path == str(old):
                        seen.set()
                        self.assert_wait()
                    return self.entry.stat(**kwargs)

                def assert_wait(self):
                    if not renamed.wait(5):
                        raise RuntimeError("Synthetic rename did not occur")

            class Entries:
                def __init__(self, path):
                    self.original = real_scandir(path)

                def __enter__(self):
                    return (Entry(entry) for entry in self.original)

                def __exit__(self, *_):
                    self.original.close()

            def rename():
                if seen.wait(5):
                    old.replace(new)
                    renamed.set()

            thread = threading.Thread(target=rename)
            thread.start()
            with patch("benchmarks.competitive.control.os.scandir", Entries):
                self.assertEqual(archive_bytes(root), 937)
            thread.join(5)
            self.assertTrue(renamed.is_set())
            with patch(
                "benchmarks.competitive.control.os.scandir",
                side_effect=PermissionError("denied"),
            ):
                with self.assertRaises(PermissionError):
                    archive_bytes(root)
            with patch(
                "benchmarks.competitive.control.os.scandir",
                side_effect=FileNotFoundError("race"),
            ):
                with self.assertRaises(RuntimeError):
                    archive_bytes(root)

    @unittest.skipUnless(
        os.name == "nt", "Recorded worker custody uses Windows Job Objects"
    )
    def test_failed_worker_is_retained_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "job"
            folder.mkdir()
            job = {"job_id": "failed"}
            atomic_json(folder / "JOB.json", job)
            command = [
                sys.executable,
                "-B",
                str(ROOT / "tests/_benchmark_fake_worker.py"),
                "--job",
                str(folder / "JOB.json"),
                "--kind",
                "incomplete",
            ]
            limits = {
                "cpu_s": 5,
                "wall_s": 15,
                "rss_bytes": 1024**3,
                "archive_bytes": 1024**2,
            }
            journal = Journal(root / "journal.jsonl")
            result = launch(
                command, folder, limits, journal, "failed", worker_environment()
            )
            self.assertEqual(result["reason"], "WORKER_INCOMPLETE")
            # Recorded owned-worker accounting retains a two-second failure
            # closure reservation in addition to the worker's CPU ceiling.
            self.assertEqual(result["cpu_s"], limits["cpu_s"] + 2)
            self.assertTrue(result["owned_family"]["family_exit_verified"])
            self.assertTrue((folder / "half-written.partial").exists())
            with self.assertRaises(ValueError):
                launch(command, folder, limits, journal, "failed", worker_environment())

    def test_budget_charges_unresolved_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            budget = Budget(
                root,
                {
                    "cpu_s": 50,
                    "wall_s": 100,
                    "archive_bytes": 1024**2,
                    "fit_attempts": 1,
                },
            )
            budget.journal.start(
                "one", {"cpu_s": 40, "operations": {"fit_attempts": 1}}
            )
            self.assertGreaterEqual(budget.usage()["cpu_s"], 40)
            with self.assertRaises(RuntimeError):
                budget.reserve(1, 1, fit_attempts=1)

    @unittest.skipUnless(
        os.name == "nt", "Recorded worker custody uses Windows Job Objects"
    )
    def test_valid_receipt_from_observed_venv_worker_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "job"
            folder.mkdir()
            atomic_json(folder / "JOB.json", {"job_id": "complete"})
            command = [
                sys.executable,
                "-B",
                str(ROOT / "tests/_benchmark_fake_worker.py"),
                "--job",
                str(folder / "JOB.json"),
                "--kind",
                "receipt",
            ]
            limits = {
                "cpu_s": 5,
                "wall_s": 15,
                "rss_bytes": 1024**3,
                "archive_bytes": 1024**2,
            }
            result = launch(
                command,
                folder,
                limits,
                Journal(root / "journal.jsonl"),
                "complete",
                worker_environment(),
            )
            self.assertIsNone(result["reason"])
            self.assertEqual(result["exit_code"], 0)

    @unittest.skipUnless(
        os.name == "nt", "Recorded worker custody uses Windows Job Objects"
    )
    def test_actual_dedicated_pair_has_equal_load_and_normal_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "input.npz"
            X = pd.DataFrame({"a": np.arange(40, dtype=float) - 20})
            write_bundle(
                data, X, pd.Series(["a", "b"] * 20), np.arange(40), np.arange(40), []
            )
            descriptions = []
            pair = root / "pair"
            for which in (0, 1):
                folder = pair / str(which)
                folder.mkdir(parents=True)
                job = {
                    "job_id": str(which),
                    "task": "binary",
                    "model_sha256": "artificial",
                    "input_sha256": "artificial",
                    "input_file": str(data),
                }
                atomic_json(folder / "JOB.json", job)
                descriptions.append(
                    {
                        "job_id": str(which),
                        "command": [
                            sys.executable,
                            "-B",
                            "-u",
                            str(ROOT / "tests/_benchmark_fake_worker.py"),
                            "--job",
                            str(folder / "JOB.json"),
                        ],
                    }
                )
            result = run_pair(
                descriptions,
                pair,
                Journal(root / "journal.jsonl"),
                {
                    "cpu_s": 15,
                    "wall_s": 40,
                    "rss_bytes": 1024**3,
                    "archive_bytes": 16 * 1024**2,
                },
                reverse=True,
                environment=worker_environment(),
            )
            self.assertTrue(result["both_workers_exited"])
            self.assertEqual(
                [row["prediction_API_calls"] for row in result["costs"]], [1319, 1319]
            )
            self.assertEqual(result["balanced_block_order"].count([0, 1]), 6)
            self.assertEqual(result["balanced_block_order"].count([1, 0]), 6)
            self.assertEqual(
                [row["event"] for row in result["records"][0]].count("BLOCK"), 12
            )


if __name__ == "__main__":
    unittest.main()

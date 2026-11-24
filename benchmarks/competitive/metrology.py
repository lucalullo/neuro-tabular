"""Dedicated, equally loaded A/A and A/B workers with interleaved warm blocks."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .control import archive_bytes, atomic_json, isolated_environment

BLOCKS = ((1, 256), (32, 64), ("full", 8)) * 4
PROTOCOL = "NT_BENCH_IPC/1"


def serve(model, X, groups, job, output, started_cpu, started_wall, forward_counter=None):
    """A ready worker never predicts until the controller asks it to."""
    import numpy as np

    calls = 0
    blocks = []
    previous = {}
    warmed = False

    def reply(value):
        print(
            json.dumps(
                {"protocol": PROTOCOL, "job_id": job["job_id"], **value},
                default=lambda v: v.item() if hasattr(v, "item") else str(v),
            ),
            flush=True,
        )

    reply(
        {
            "event": "READY",
            "pid": os.getpid(),
            "model_sha256": job["model_sha256"],
            "input_sha256": job["input_sha256"],
        }
    )
    for line in __import__("sys").stdin:
        command = json.loads(line)
        kind = command["command"]
        if kind == "cold":
            if calls:
                raise ValueError("Cold call must be first")
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            labels = model.predict(X)
            measured = {
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
            calls += 1
            values = labels if job["task"] == "regression" else model.predict_proba(X)
            calls += job["task"] != "regression"
            saved_labels = labels.astype("str") if labels.dtype.kind == "O" else labels
            temporary = output / "predictions.npz.partial"
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    labels=saved_labels,
                    values=values,
                    groups=groups,
                    classes=np.asarray(model.class_order_ or [], dtype="str"),
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(output / "predictions.npz")
            value = {
                "event": "COLD",
                "prediction_sha256": hashlib.sha256(
                    (output / "predictions.npz").read_bytes()
                ).hexdigest(),
                **measured,
            }
            atomic_json(output / "PREDICTIONS_COMPLETE.json", value)
            reply(value)
        elif kind == "warm":
            if calls not in {1, 2}:
                raise ValueError("Warm-up order")
            for _ in range(5):
                model.predict(X)
                calls += 1
            warmed = True
            reply({"event": "WARM", "calls": 5})
        elif kind == "block":
            index = command["round"]
            if not warmed or index != len(blocks) or not 0 <= index < len(BLOCKS):
                raise ValueError("Unregistered/repeated block")
            batch, count = BLOCKS[index]
            subset = X if batch == "full" else X.iloc[: min(batch, len(X))]
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            for _ in range(count):
                repeated = model.predict(subset)
            timing = {
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
            if batch in previous and not np.array_equal(previous[batch], repeated):
                raise ValueError("Non-deterministic warm predictions")
            previous[batch] = repeated.copy()
            calls += count
            value = {
                "event": "BLOCK",
                "round": index,
                "batch": batch,
                "calls": count,
                **timing,
            }
            blocks.append(value)
            reply(value)
        elif kind == "close":
            if len(blocks) != 12:
                raise ValueError("Incomplete metrology")
            value = {
                "job_id": job["job_id"],
                "prediction_API_calls": calls,
                "timing_blocks": blocks,
                "job_sha256": hashlib.sha256(
                    (output / "JOB.json").read_bytes()
                ).hexdigest(),
                "worker_cpu_s": time.process_time() - started_cpu,
                "worker_wall_s": time.perf_counter() - started_wall,
                "fit_calls": 0,
                "optimizer_steps": 0,
            }
            if forward_counter is not None:
                value.update(forward_counter.snapshot())
            atomic_json(output / "RESULT.json", value)
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
            reply({"event": "DONE"})
            return
        else:
            raise ValueError("Unknown metrology command")
    raise ValueError("Controller disconnected before completion")


class Resident:
    """A process plus a bounded line reader; never shares an interpreter/version."""

    def __init__(self, command, folder, job_id, environment):
        import psutil

        self.folder, self.job_id = Path(folder), job_id
        self.owned_processes = []
        self.stderr = (self.folder / "stderr.log").open("xb")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        from .windows_job import OwnedPopen
        self.process = OwnedPopen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            env=isolated_environment(environment, self.folder),
            cwd=self.folder,
            creationflags=flags,
            bufsize=1,
            registry=self.owned_processes.append,
            resume=False,
        )
        self.ps = psutil.Process(self.process.pid)
        self.birth = self.ps.create_time()
        self.process.check_identity(self.process.pid, self.birth)
        self.process.resume()
        self.worker_pid = None
        self.identities = {self.process.pid: self.birth}
        self.known_cpu = {}
        self.peak_rss = 0
        self.cpu = 0.0
        self.messages = queue.Queue(maxsize=64)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                with (self.folder / "stdout.log").open("a", encoding="utf-8") as stream:
                    stream.write(line)
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("protocol") == PROTOCOL:
                    self.messages.put(item)
        finally:
            self.messages.put({"event": "EOF"})

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def sample(self):
        import psutil

        self.cpu = max(self.cpu, self.process.query()["CPU_s"])
        try:
            # Membership comes from the owned kernel job, including after the
            # direct launcher exits. psutil is RSS telemetry, not ownership.
            for process in [psutil.Process(pid) for pid in self.process.active_pids()]:
                identity = process.pid, process.create_time()
                self.identities[identity[0]] = identity[1]
                values = process.cpu_times()
                self.known_cpu[identity] = values.user + values.system
                memory = process.memory_info()
                self.peak_rss = max(
                    self.peak_rss, memory.rss, getattr(memory, "peak_wset", 0)
                )
        except psutil.NoSuchProcess:
            pass

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()

    def close_handles(self):
        self.reader.join(timeout=2)
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.stdout.close()
        self.stderr.close()
        self.process.close()


def run_pair(descriptions, root, journal, limits, reverse=False, environment=None):
    """Both residents perform cold, five warm-ups and exactly the same 12 blocks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    residents = {}
    started = time.perf_counter()
    previous_bytes = archive_bytes(root)
    order = [1, 0] if reverse else [0, 1]
    sequence = []

    def check():
        nonlocal previous_bytes
        for resident in residents.values():
            resident.sample()
            if (
                resident.cpu >= limits["cpu_s"] - 2
                or resident.peak_rss > limits["rss_bytes"]
            ):
                raise RuntimeError("Metrology CPU/RSS limit")
        if time.perf_counter() - started >= limits["wall_s"] - 10:
            raise RuntimeError("Metrology wall limit")
        previous_bytes = max(previous_bytes, archive_bytes(root))
        if previous_bytes > limits["archive_bytes"]:
            raise RuntimeError("Metrology archive limit")

    def receive(which, event):
        while True:
            check()
            try:
                message = residents[which].messages.get(timeout=0.05)
            except queue.Empty:
                continue
            if (
                message.get("job_id") != descriptions[which]["job_id"]
                or message.get("event") != event
            ):
                raise RuntimeError("Unexpected/incomplete worker acknowledgement")
            return message

    records = [[], []]
    try:
        for which in order:
            item = descriptions[which]
            folder = root / str(which)
            folder.mkdir(exist_ok=True)
            if (folder / "WORKER_DONE.json").exists():
                raise ValueError("Do not relaunch completed resident")
            journal.start(item["job_id"], {**limits, "operations": item.get("operations", limits.get("operations", {}))})
            residents[which] = Resident(
                item["command"], folder, item["job_id"], environment
            )
            ready = receive(which, "READY")
            residents[which].sample()
            if ready["pid"] not in residents[which].identities:
                raise RuntimeError("READY pid is outside the launched process tree")
            residents[which].worker_pid = ready["pid"]
            for key in ("model_sha256", "input_sha256"):
                if item.get(key) and ready.get(key) != item[key]:
                    raise RuntimeError("Wrong resident model/input")
        for command, event in (("cold", "COLD"), ("warm", "WARM")):
            for which in reversed(order):
                residents[which].send({"command": command})
                records[which].append(receive(which, event))
        for round_index in range(12):
            block_order = order if round_index % 2 == 0 else list(reversed(order))
            for which in block_order:
                residents[which].send({"command": "block", "round": round_index})
                records[which].append(receive(which, "BLOCK"))
            sequence.append(block_order)
        for which in order:
            residents[which].send({"command": "close"})
        for which in order:
            receive(which, "DONE")
        # Wait for exit independently in parallel, not sequentially spending
        # one timeout per resident. DONE alone is never an exit certificate.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                which: pool.submit(resident.process.wait, timeout=8)
                for which, resident in residents.items()
            }
            codes = {which: future.result() for which, future in futures.items()}
        if any(code != 0 for code in codes.values()):
            raise RuntimeError("Metrology process exit failed")
        outcomes = []
        for which, resident in residents.items():
            final_job_accounting = resident.process.query()
            resident.cpu = max(resident.cpu, final_job_accounting["CPU_s"])
            result_path = resident.folder / "RESULT.json"
            ack = json.loads(
                (resident.folder / "WORKER_DONE.json").read_text(encoding="utf-8")
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                ack["job_id"] != resident.job_id
                or ack["pid"] != resident.worker_pid
                or ack["result_sha256"]
                != hashlib.sha256(result_path.read_bytes()).hexdigest()
            ):
                raise RuntimeError("Completion receipt identity")
            cost = {
                "cpu_s": max(resident.cpu, result["worker_cpu_s"]),
                "peak_rss_bytes": resident.peak_rss,
                "prediction_API_calls": result["prediction_API_calls"],
                "model_load_operations": 1,
                "torch_root_forward_calls": result.get("torch_root_forward_calls", 0),
                "exit_code": codes[which],
                "measurement_workers": 1,
                "wall_s": (time.perf_counter() - started) / 2,
                "which": which,
                "job_id": resident.job_id,
                "launcher_pid": resident.process.pid,
                "worker_pid": resident.worker_pid,
                "observed_identities": resident.identities,
                "owned_family_active_processes": final_job_accounting["ActiveProcesses"],
                "kernel_job_CPU_s": final_job_accounting["CPU_s"],
                "psutil_CPU_not_added_to_kernel_job_CPU": True,
            }
            journal.finish(resident.job_id, "COMPLETED", cost)
            outcomes.append(cost)
        receipt = {
            "PASS_EXECUTION": True,
            "records": records,
            "balanced_block_order": sequence,
            "costs": outcomes,
            "both_workers_exited": True,
            "archive_bytes_charged": previous_bytes,
            "retry": False,
        }
        atomic_json(root / "PAIR.json", receipt)
        return receipt
    except BaseException as exc:
        for resident in residents.values():
            resident.stop()
        for which, resident in residents.items():
            try:
                code = resident.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                resident.process.kill()
                code = resident.process.wait(timeout=5)
            resident.cpu = max(resident.cpu, resident.process.query()["CPU_s"])
            if sum(row["job_id"] == resident.job_id for row in journal.events()) == 1:
                journal.finish(
                    resident.job_id,
                    "FAILED",
                    {
                        "cpu_s": max(limits["cpu_s"] + 2, resident.cpu),
                        "model_load_operations": 1,
                        "torch_root_forward_calls": descriptions[which].get("operations", {}).get("torch_root_forward_calls", 0),
                        "reservation_is_not_actual_operations": True,
                        "reason": str(exc),
                        "exit_code": code,
                        "measurement_workers": 1,
                        "owned_family_active_processes": resident.process.query()["ActiveProcesses"],
                        "kernel_job_CPU_s": resident.cpu,
                        "prediction_API_calls": limits.get(
                            "prediction_API_calls", 1319
                        ),
                    },
                )
        # Constructor failures occur after STARTED but before the object is
        # assigned into residents. Close that ticket too; never treat it as new.
        for which, description in enumerate(descriptions):
            events = [e for e in journal.events() if e["job_id"] == description["job_id"]]
            if len(events) == 1 and events[0]["event"] == "STARTED":
                failure = root / str(which) / "CTOR_FAILURE.json"
                detail = json.loads(failure.read_text()) if failure.exists() else {}
                journal.finish(description["job_id"], "FAILED", dict(
                    cpu_s=limits["cpu_s"] + 2, reason="RESIDENT_CONSTRUCTION_INCOMPLETE",
                    error=repr(exc), retry=False, worker_exited=detail.get("worker_exited", False),
                    reservation_is_not_actual_operations=True, **description.get("operations", {})))
        atomic_json(
            root / "PAIR_FAILURE.json",
            {"PASS_EXECUTION": False, "error": repr(exc), "retry": False},
        )
        raise
    finally:
        for resident in residents.values():
            resident.close_handles()


_ResidentImpl = Resident

class Resident(_ResidentImpl):
    """Own the new process even if construction fails after Popen succeeds."""
    def __init__(self, command, folder, job_id, environment):
        try:
            super().__init__(command, folder, job_id, environment)
        except BaseException as error:
            process = getattr(self, "process", None)
            ownership = [owner.close() for owner in getattr(self, "owned_processes", [])]
            for name in ("stdin", "stdout"):
                stream = getattr(process, name, None)
                if stream is not None:
                    stream.close()
            stream = getattr(self, "stderr", None)
            if stream is not None:
                stream.close()
            atomic_json(Path(folder) / "CTOR_FAILURE.json", dict(
                job_id=job_id, error=repr(error), retry=False,
                owned_PID=getattr(process, "pid", None),
                owned_family_receipts=ownership,
                worker_exited=all(row["family_exit_verified"] for row in ownership)))
            raise

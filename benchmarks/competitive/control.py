"""Single-controller tickets, atomic receipts and conservative resource limits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    raw = json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def archive_bytes(root, attempts=3):
    """Retry the entire *census* on rename races; never silently count ENOENT as 0."""
    for attempt in range(attempts):
        try:
            total = 0
            pending = [Path(root)]
            while pending:
                with os.scandir(pending.pop()) as entries:
                    for entry in entries:
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode):
                            total += info.st_size
                        else:
                            raise ValueError("Linked/special archive entry")
            return total
        except FileNotFoundError:
            if attempt == attempts - 1:
                raise RuntimeError("Archive census remained unstable") from None
    raise AssertionError("Unreachable")


class Journal:
    def __init__(self, path):
        self.path = Path(path)

    def events(self):
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]

    def start(self, job_id, reservation):
        if any(row["job_id"] == job_id for row in self.events()):
            raise ValueError("Already-started jobs cannot be retried")
        self._append({"job_id": job_id, "event": "STARTED", "reservation": reservation})

    def finish(self, job_id, outcome, cost):
        found = [row for row in self.events() if row["job_id"] == job_id]
        if len(found) != 1 or found[0]["event"] != "STARTED":
            raise ValueError("Missing start or duplicate terminal receipt")
        self._append({"job_id": job_id, "event": outcome, "cost": cost})

    def _append(self, value):
        with self.path.open("ab") as stream:
            stream.write(
                json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())


def isolated_environment(environment, folder):
    environment = dict(os.environ if environment is None else environment)
    temporary = Path(folder).resolve() / "tmp"
    temporary.mkdir(exist_ok=True)
    for key in (
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
    ):
        environment[key] = str(temporary)
    return environment


def host_memory_ready(rss_bytes, processes=1):
    import psutil

    if psutil.virtual_memory().available < processes * rss_bytes + 1024**3:
        raise RuntimeError(
            "PAUSED_BEFORE_DISPATCH: insufficient free host RAM; no new ticket started"
        )


def _launch_impl(command, folder, limits, journal, job_id, environment=None, owned_registry=None):
    """Monitor a fresh worker. A failure is retained and never relaunched."""
    import psutil

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "TERMINAL.json").exists():
        raise ValueError("Terminal job")
    reserved = dict(limits)
    journal.start(job_id, reserved)
    start = time.perf_counter()
    ceiling_bytes = archive_bytes(folder)
    last_cpu = 0.0
    peaks = {}
    identities = {}
    reason = None
    with (
        (folder / "stdout.log").open("xb") as stdout,
        (folder / "stderr.log").open("xb") as stderr,
    ):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        from .windows_job import OwnedPopen
        process = OwnedPopen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=isolated_environment(environment, folder),
            cwd=folder,
            creationflags=flags,
            registry=owned_registry,
            resume=False,
        )
        monitored = psutil.Process(process.pid)
        identities[process.pid] = monitored.create_time()
        process.check_identity(process.pid, identities[process.pid])
        process.resume()
        known_cpu = {}
        while process.poll() is None:
            try:
                # Kernel job accounting includes short-lived descendants after
                # they exit. Do not add it to sampled psutil CPU a second time.
                last_cpu = max(last_cpu, process.query()["CPU_s"])
                family = [psutil.Process(pid) for pid in process.active_pids()]
                for member in family:
                    identity = member.pid, member.create_time()
                    identities[member.pid] = identity[1]
                    cpu = member.cpu_times()
                    known_cpu[identity] = cpu.user + cpu.system
                    memory = member.memory_info()
                    observed = max(memory.rss, getattr(memory, "peak_wset", 0))
                    peaks[identity] = max(peaks.get(identity, 0), observed)
                    if observed > limits["rss_bytes"]:
                        reason = "MEMORY_LIMIT"
            except psutil.NoSuchProcess:
                pass
            except Exception:
                reason = "MONITOR_ERROR"
            # An exited launcher / transient psutil race must never bypass
            # the wall, archive or kernel CPU limits of its still-live family.
            if last_cpu >= max(0.1, limits["cpu_s"] - limits.get("closure_margin_cpu_s", 2)):
                reason = reason or "CPU_LIMIT"
            if time.perf_counter() - start >= max(0.1, limits["wall_s"] - limits.get("closure_margin_wall_s", 10)):
                reason = reason or "WALL_LIMIT"
            ceiling_bytes = max(ceiling_bytes, archive_bytes(folder))
            if ceiling_bytes > limits["archive_bytes"]:
                reason = reason or "ARCHIVE_LIMIT"
            if reason:
                (folder / "STOP_REQUESTED").write_text(reason, encoding="utf-8")
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                break
            time.sleep(0.05)
        try:
            code = process.wait(
                timeout=max(2, limits.get("closure_margin_wall_s", 10) - 2)
            )
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=5)
            reason = reason or "EXIT_TIMEOUT"
        final_job_accounting = process.query()
        last_cpu = max(last_cpu, final_job_accounting["CPU_s"])
        ownership = process.close()
    result = folder / "RESULT.json"
    ack = folder / "WORKER_DONE.json"
    if code != 0 or not result.is_file() or not ack.is_file():
        reason = reason or "WORKER_INCOMPLETE"
    if reason is None:
        received = json.loads(result.read_text(encoding="utf-8"))
        done = json.loads(ack.read_text(encoding="utf-8"))
        if (
            done.get("job_id") != job_id
            or done.get("pid") not in identities
            or received.get("job_id") != job_id
            or done.get("result_sha256")
            != hashlib.sha256(result.read_bytes()).hexdigest()
        ):
            reason = "RECEIPT_IDENTITY"
        else:
            last_cpu = max(last_cpu, received.get("worker_cpu_s", 0))
    # Missing evidence consumes its reservation, not a fabricated zero cost.
    charged_cpu = max(limits["cpu_s"] + 2, math.ceil(last_cpu * 1000) / 1000) if reason else math.ceil(last_cpu * 1000) / 1000
    cost = {
        "cpu_s": charged_cpu,
        "wall_s": time.perf_counter() - start,
        "peak_rss_bytes": max(peaks.values(), default=0),
        "archive_bytes_charged": max(ceiling_bytes, archive_bytes(folder)),
        "exit_code": code,
        "reason": reason,
        "worker_exited": True,
        "retry": False,
        "owned_family": ownership,
        "kernel_job_CPU_s": final_job_accounting["CPU_s"],
        "psutil_CPU_not_added_to_kernel_job_CPU": True,
    }
    cost["observed_process_identities"] = identities
    cost["reservation_is_not_actual_operations"] = reason is not None
    partial = folder / "COUNTERS_PARTIAL.json"
    if partial.exists():
        cost["partial_observed_operations"] = json.loads(partial.read_text())
    for name, maximum in limits.get("operations", {}).items():
        result_key = {"fit_attempts": "fit_calls"}.get(name, name)
        cost[name] = (
            maximum
            if reason or name == "fit_attempts"
            else received.get(result_key, maximum)
        )
    journal.finish(job_id, "FAILED" if reason else "COMPLETED", cost)
    atomic_json(folder / "TERMINAL.json", cost)
    return cost


class Budget:
    """One journal, conservative unresolved reservations, controller costs included."""

    def __init__(self, root, maxima):
        self.root, self.maxima = Path(root), dict(maxima)
        self.journal = Journal(self.root / "journal.jsonl")
        self.started_cpu, self.started_wall = time.process_time(), time.perf_counter()
        self.previous = {"cpu_s": 0.0, "wall_s": 0.0}
        path = self.root / "CONTROLLER_COST.json"
        if path.exists():
            self.previous = json.loads(path.read_text(encoding="utf-8"))

    def usage(self):
        starts, finishes = {}, {}
        for row in self.journal.events():
            (starts if row["event"] == "STARTED" else finishes)[row["job_id"]] = row
        total = {key: 0 for key in self.maxima}
        for job_id, row in starts.items():
            charge = (
                finishes[job_id]["cost"]
                if job_id in finishes
                else {**row["reservation"], **row["reservation"].get("operations", {})}
            )
            for key in total:
                if key not in {"archive_bytes", "wall_s"}:
                    total[key] += charge.get(key, 0)
        total["cpu_s"] += (
            self.previous["cpu_s"] + time.process_time() - self.started_cpu
        )
        # Active wall is actual controller lifetime; concurrent residents are not
        # charged twice. Previous sessions are persisted before every dispatch.
        total["wall_s"] = (
            self.previous["wall_s"] + time.perf_counter() - self.started_wall
        )
        total["archive_bytes"] = archive_bytes(self.root)
        return total

    def reserve(self, cpu_s, wall_s, archive_bytes=0, **operations):
        used = self.usage()
        request = {
            "cpu_s": cpu_s,
            "wall_s": wall_s,
            "archive_bytes": archive_bytes,
            **operations,
        }
        for key, maximum in self.maxima.items():
            if used.get(key, 0) + request.get(key, 0) > maximum:
                raise RuntimeError("Cumulative budget would be exceeded: " + key)
        self.checkpoint()

    def checkpoint(self):
        cost = {
            "cpu_s": self.previous["cpu_s"] + time.process_time() - self.started_cpu,
            "wall_s": self.previous["wall_s"] + time.perf_counter() - self.started_wall,
        }
        atomic_json(self.root / "CONTROLLER_COST.json", cost)
        atomic_json(self.root / "USAGE.json", self.usage())


def launch(command, folder, limits, journal, job_id, environment=None):
    """Retain and close an owned worker even on KeyboardInterrupt/Popen failure."""
    tracked = []
    started = time.perf_counter()
    try:
        return _launch_impl(command, folder, limits, journal, job_id, environment, tracked.append)
    except BaseException as error:
        folder = Path(folder)
        (folder / "STOP_REQUESTED").write_text(type(error).__name__, encoding="utf-8")
        ownership = [process.close() for process in tracked]
        events = [row for row in journal.events() if row["job_id"] == job_id]
        if len(events) == 1 and events[0]["event"] == "STARTED":
            kernel_cpu = sum(row.get("final_job_accounting", {}).get("CPU_s", 0) for row in ownership)
            cost = dict(cpu_s=max(limits["cpu_s"] + 2, kernel_cpu), wall_s=time.perf_counter() - started,
                reason="CONTROLLER_EXCEPTION:" + type(error).__name__, error=repr(error),
                retry=False, worker_exited=all(row["family_exit_verified"] for row in ownership),
                owned_family_receipts=ownership, reservation_is_not_actual_operations=True,
                **limits.get("operations", {}))
            journal.finish(job_id, "FAILED", cost)
            atomic_json(folder / "TERMINAL.json", cost)
        raise
    finally:
        for process in tracked:
            process.close()

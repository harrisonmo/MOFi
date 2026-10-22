"""Unified compute scheduling: one job contract, local or Slurm execution, persistent session budgets.

Scientific tools describe *what* to run (argv, working directory, resources). This layer owns *how*:
thread and GPU pinning, wall-time enforcement, process-tree cleanup, batch submission, capacity
sharing between concurrent controllers, and a conservative cost ledger that survives restarts.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PYTHON_TOKEN = "{python}"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CONTROLLED_ENV = set(THREAD_VARIABLES) | {"CUDA_VISIBLE_DEVICES"}
SPEC_KEYS = {"name", "argv", "cwd", "env", "resources", "log"}
RESOURCE_KEYS = {"cpus", "gpus", "memory_mb", "wall_seconds"}
BUDGET_KEYS = {"max_jobs", "max_cpu_seconds", "max_gpu_seconds", "max_wall_seconds"}
ACTIVE = {"reserved", "running", "queued"}
# Measured outcomes are charged by elapsed time; anything else is charged at the full reservation.
MEASURED = {"completed", "failed", "timeout", "cancelled"}
TERMINAL = MEASURED | {"interrupted", "unknown", "queue_timeout"}
SLURM_ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED",
                "STAGE_OUT", "SIGNALING", "REQUEUE_HOLD", "REQUEUE_FED"}
SLURM_FINAL = {"COMPLETED": "completed", "FAILED": "failed", "NODE_FAIL": "failed", "OUT_OF_MEMORY": "failed",
               "BOOT_FAIL": "failed", "DEADLINE": "timeout", "TIMEOUT": "timeout", "PREEMPTED": "cancelled",
               "CANCELLED": "cancelled", "REVOKED": "cancelled"}


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_keys(value, allowed, required, where):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{where} must contain {sorted(required)} and only {sorted(allowed)}")


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def validate_spec(spec):
    """Check a backend-neutral job description and return a normalized copy."""
    exact_keys(spec, SPEC_KEYS, {"name", "argv", "cwd", "resources"}, "Job spec")
    spec = copy.deepcopy(spec)
    if not isinstance(spec["name"], str) or not spec["name"] or not all(c.isalnum() or c in "-_." for c in spec["name"]):
        raise ValueError("Job name must be a non-empty [A-Za-z0-9._-] string")
    if not isinstance(spec["argv"], list) or not spec["argv"] or not all(isinstance(a, str) for a in spec["argv"]):
        raise ValueError("argv must be a non-empty list of strings; shells are never invoked implicitly")
    # A Slurm cwd is a cluster path, so a POSIX absolute path is accepted on any controller.
    if not isinstance(spec["cwd"], str) or not (Path(spec["cwd"]).is_absolute()
                                                or PurePosixPath(spec["cwd"]).is_absolute()):
        raise ValueError("cwd must be an absolute path")
    spec.setdefault("env", {})
    if not isinstance(spec["env"], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in spec["env"].items()):
        raise ValueError("env must map strings to strings")
    if CONTROLLED_ENV & set(spec["env"]):
        raise ValueError(f"Thread and GPU variables are set from resources: {sorted(CONTROLLED_ENV)}")
    spec.setdefault("log", f"{spec['name']}.log")
    if not isinstance(spec["log"], str) or Path(spec["log"]).is_absolute() or ".." in Path(spec["log"]).parts:
        raise ValueError("log must be a relative path inside cwd")
    resources = spec["resources"]
    exact_keys(resources, RESOURCE_KEYS, {"cpus", "wall_seconds"}, "Resources")
    resources.setdefault("gpus", 0)
    resources.setdefault("memory_mb", None)
    integer(resources["cpus"], "cpus", 1)
    integer(resources["gpus"], "gpus")
    if resources["memory_mb"] is not None:
        integer(resources["memory_mb"], "memory_mb", 1)
    positive(resources["wall_seconds"], "wall_seconds")
    return spec


def validate_config(config):
    exact_keys(config, {"schema_version", "backends", "session_budget"},
               {"schema_version", "backends", "session_budget"}, "HPC config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported HPC config schema {config['schema_version']}")
    if not isinstance(config["backends"], dict) or not config["backends"]:
        raise ValueError("At least one backend is required")
    for name, backend in config["backends"].items():
        kind = backend.get("type") if isinstance(backend, dict) else None
        if kind == "local":
            exact_keys(backend, {"type", "max_cpus", "gpu_ids", "max_memory_mb"}, {"type", "max_cpus"}, name)
            integer(backend["max_cpus"], f"{name}.max_cpus", 1)
            gpus = backend.get("gpu_ids", [])
            if not isinstance(gpus, list) or len(set(gpus)) != len(gpus) or not all(isinstance(g, str) for g in gpus):
                raise ValueError(f"{name}.gpu_ids must be distinct device strings")
            if backend.get("max_memory_mb") is not None:
                integer(backend["max_memory_mb"], f"{name}.max_memory_mb", 1)
        elif kind == "slurm":
            allowed = {"type", "partition", "account", "qos", "max_cpus", "max_gpus", "max_memory_mb",
                       "max_wall_seconds", "max_queue_seconds", "poll_seconds", "python", "env", "setup",
                       "extra_directives"}
            exact_keys(backend, allowed, {"type", "partition", "max_cpus", "max_wall_seconds",
                                          "max_queue_seconds"}, name)
            integer(backend["max_cpus"], f"{name}.max_cpus", 1)
            integer(backend.get("max_gpus", 0), f"{name}.max_gpus")
            for key in ("max_wall_seconds", "max_queue_seconds", "poll_seconds"):
                if key in backend:
                    positive(backend[key], f"{name}.{key}")
            for key in ("setup", "extra_directives"):
                if not all(isinstance(line, str) and "\n" not in line for line in backend.get(key, [])):
                    raise ValueError(f"{name}.{key} must be single-line strings")
            if any(not d.startswith("--") for d in backend.get("extra_directives", [])):
                raise ValueError(f"{name}.extra_directives must be sbatch long options")
            env = backend.get("env", {})
            if not isinstance(env, dict) or CONTROLLED_ENV & set(env) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise ValueError(f"{name}.env must map strings to strings and leave thread/GPU variables alone")
        else:
            raise ValueError(f"Backend {name} must have type 'local' or 'slurm'")
    budget = config["session_budget"]
    exact_keys(budget, BUDGET_KEYS, BUDGET_KEYS, "session_budget")
    integer(budget["max_jobs"], "max_jobs", 1)
    for key in BUDGET_KEYS - {"max_jobs"}:
        positive(budget[key], key)
    return config


def thread_environment(cpus, gpu_ids):
    env = {name: str(cpus) for name in THREAD_VARIABLES}
    # An empty list hides every device, so CPU-only jobs cannot silently use a GPU.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return env


def stop_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied still means the process exists.
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def slurm_time(seconds):
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes // 1440}-{minutes % 1440 // 60:02d}:{minutes % 60:02d}:00"


def render_sbatch(spec, backend):
    """Return a self-contained batch script; every value is quoted, and argv is exec'd without a shell eval."""
    r = spec["resources"]
    python = backend.get("python", "python3")
    argv = [python if a == PYTHON_TOKEN else a for a in spec["argv"]]
    directives = [f"--job-name={spec['name']}", f"--output={PurePosixPath(spec['cwd'], spec['log'])}",
                  "--ntasks=1", f"--cpus-per-task={r['cpus']}", f"--time={slurm_time(r['wall_seconds'])}",
                  f"--partition={backend['partition']}"]
    for key in ("account", "qos"):
        if backend.get(key):
            directives.append(f"--{key}={backend[key]}")
    if r["memory_mb"] is not None:
        directives.append(f"--mem={r['memory_mb']}M")
    if r["gpus"]:
        directives.append(f"--gres=gpu:{r['gpus']}")
    directives += backend.get("extra_directives", [])
    # sbatch reads directives itself, without shell unquoting, so values must be plain tokens.
    if any(set(d) & set(" \t\n'\"\\") for d in directives):
        raise ValueError("Slurm directives (including cwd and log) cannot contain whitespace or quotes")
    lines = ["#!/bin/bash"] + [f"#SBATCH {d}" for d in directives] + ["set -euo pipefail"]
    env = {**backend.get("env", {}), **spec["env"], **{k: str(r["cpus"]) for k in THREAD_VARIABLES}}
    lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    if not r["gpus"]:
        lines.append("export CUDA_VISIBLE_DEVICES=''")
    # The working directory may not exist yet on the compute node, so it is created there.
    lines.append(f"mkdir -p {shlex.quote(spec['cwd'])} && cd {shlex.quote(spec['cwd'])}")
    lines += backend.get("setup", [])
    lines.append("exec " + shlex.join(argv))
    return "\n".join(lines) + "\n"


def default_runner(argv, timeout=60):
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        raise RuntimeError(f"{argv[0]} failed ({done.returncode}): {done.stderr.strip()[:500]}")
    return done.stdout


@contextmanager
def locked(path, timeout=60.0, stale_seconds=600.0):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_seconds:
                    lock.unlink()  # The ledger lock is held for milliseconds; an old one is from a crash.
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Ledger lock is held: {lock}") from None
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def charge(job):
    r = job["resources"]
    wall = job["elapsed_seconds"] if job["status"] in MEASURED and job.get("elapsed_seconds") is not None \
        else r["wall_seconds"]
    return {"jobs": 1, "cpu_seconds": r["cpus"] * wall, "gpu_seconds": r["gpus"] * wall, "wall_seconds": wall}


class Scheduler:
    """Owned by a controller; tools pass job specs, never backend choices or budgets."""

    def __init__(self, config_path, session_root, runner=default_runner, clock=time.time, sleep=time.sleep):
        self.config_path = Path(config_path).resolve()
        self.config = validate_config(read_json(self.config_path))
        self.session_root = Path(session_root).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.session_root / "hpc-ledger.json"
        self.runner, self.clock, self.sleep = runner, clock, sleep
        self.host = socket.gethostname()
        self._local = {}
        binding = {"config_sha256": sha256(self.config_path), "source_sha256": sha256(__file__)}
        with locked(self.ledger_path):
            if self.ledger_path.exists():
                ledger = read_json(self.ledger_path)
                if ledger["binding"]["config_sha256"] != binding["config_sha256"]:
                    raise ValueError("HPC config changed after the session started; a budget cannot be reset")
            else:
                ledger = {"schema_version": SCHEMA_VERSION, "binding": binding,
                          "limits": copy.deepcopy(self.config["session_budget"]), "jobs": []}
            self._reconcile(ledger)
            write_json(self.ledger_path, ledger)

    # Ledger helpers -------------------------------------------------------------------------------

    def _reconcile(self, ledger):
        """Local jobs whose controller died can no longer be measured; charge their full reservation."""
        for job in ledger["jobs"]:
            if (job["backend_type"] == "local" and job["status"] in ACTIVE and job["host"] == self.host
                    and job["controller_pid"] != os.getpid() and not pid_alive(job["controller_pid"])):
                job.update(status="unknown", finished_at=self.clock(),
                           error="Controller exited before the job was measured")
                job["charged"] = charge(job)

    def _occupies(self, job):
        if job["status"] in ACTIVE:
            return True
        # An orphaned local child still holds its CPUs and GPUs even though its cost is already charged.
        return (job["backend_type"] == "local" and job["status"] == "unknown" and job["host"] == self.host
                and pid_alive(job.get("process_pid")))

    def usage(self, ledger=None):
        ledger = ledger or self._read()
        total = {"jobs": 0, "cpu_seconds": 0.0, "gpu_seconds": 0.0, "wall_seconds": 0.0}
        for job in ledger["jobs"]:
            cost = job["charged"] if job["status"] in TERMINAL else charge({**job, "status": "reserved"})
            for key in total:
                total[key] += cost[key]
        return total

    def _admit(self, ledger, backend_name, spec):
        backend = self.config["backends"][backend_name]
        r = spec["resources"]
        if r["gpus"] and backend["type"] == "local" and r["gpus"] > len(backend.get("gpu_ids", [])):
            raise ValueError(f"Backend {backend_name} has fewer GPUs than requested")
        if backend["type"] == "slurm":
            if r["cpus"] > backend["max_cpus"] or r["gpus"] > backend.get("max_gpus", 0):
                raise ValueError(f"Job exceeds per-job limits of backend {backend_name}")
            if r["wall_seconds"] > backend["max_wall_seconds"]:
                raise ValueError(f"Job wall time exceeds backend {backend_name} limit")
            if r["memory_mb"] and backend.get("max_memory_mb") and r["memory_mb"] > backend["max_memory_mb"]:
                raise ValueError(f"Job memory exceeds backend {backend_name} limit")
        used = self.usage(ledger)
        cost = charge({"resources": r, "status": "reserved"})
        limits = ledger["limits"]
        for key in ("jobs", "cpu_seconds", "gpu_seconds", "wall_seconds"):
            if used[key] + cost[key] > limits[f"max_{key}"] + 1e-9:
                raise ValueError(f"HPC session budget exhausted: {key}")
        if backend["type"] != "local":
            return []
        busy = [j for j in ledger["jobs"] if j["backend"] == backend_name and self._occupies(j)]
        if sum(j["resources"]["cpus"] for j in busy) + r["cpus"] > backend["max_cpus"]:
            return None
        memory = backend.get("max_memory_mb")
        if memory and r["memory_mb"] and sum(j["resources"]["memory_mb"] or 0 for j in busy) + r["memory_mb"] > memory:
            return None
        taken = {g for j in busy for g in j.get("gpu_ids", [])}
        free = [g for g in backend.get("gpu_ids", []) if g not in taken]
        return free[:r["gpus"]] if len(free) >= r["gpus"] else None

    def _update(self, job_id, **fields):
        with locked(self.ledger_path):
            ledger = read_json(self.ledger_path)
            job = next(j for j in ledger["jobs"] if j["id"] == job_id)
            job.update(fields)
            if job["status"] in TERMINAL and "charged" not in job:
                job["charged"] = charge(job)
            write_json(self.ledger_path, ledger)
            return copy.deepcopy(job)

    def _read(self):
        # Windows cannot replace a file another process is reading, so readers take the lock too.
        with locked(self.ledger_path):
            return read_json(self.ledger_path)

    def job(self, job_id):
        return next(j for j in self._read()["jobs"] if j["id"] == job_id)

    # Public API -----------------------------------------------------------------------------------

    def submit(self, spec, backend=None, capacity_timeout=0.0):
        """Reserve budget and capacity, then launch. Waits up to capacity_timeout for local slots."""
        spec = validate_spec(spec)
        backend_name = backend or next(iter(self.config["backends"]))
        if backend_name not in self.config["backends"]:
            raise ValueError(f"Unknown backend {backend_name}")
        kind = self.config["backends"][backend_name]["type"]
        deadline = time.monotonic() + capacity_timeout
        while True:
            with locked(self.ledger_path):
                ledger = read_json(self.ledger_path)
                self._reconcile(ledger)
                gpu_ids = self._admit(ledger, backend_name, spec)
                if gpu_ids is not None:
                    job = {"id": f"job-{len(ledger['jobs']) + 1:04d}", "name": spec["name"], "backend": backend_name,
                           "backend_type": kind, "host": self.host, "controller_pid": os.getpid(),
                           "resources": spec["resources"], "gpu_ids": gpu_ids, "cwd": spec["cwd"],
                           "log": spec["log"], "argv": spec["argv"],
                           "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                           "status": "reserved", "submitted_at": self.clock(), "elapsed_seconds": None}
                    ledger["jobs"].append(job)
                    write_json(self.ledger_path, ledger)
                    break
                write_json(self.ledger_path, ledger)
            if time.monotonic() >= deadline:
                raise ValueError(f"Backend {backend_name} has no free capacity for {spec['name']}")
            self.sleep(0.5)
        try:
            launch = self._launch_local if kind == "local" else self._launch_slurm
            return launch(job, spec)
        except BaseException as exc:
            # Launch never reached a measurable state, so the reservation is charged in full.
            self._update(job["id"], status="unknown", finished_at=self.clock(), error=f"{type(exc).__name__}: {exc}")
            raise

    def _launch_local(self, job, spec):
        cwd = Path(spec["cwd"])
        if not cwd.is_absolute():
            raise ValueError("A local job needs an absolute cwd on this host")
        cwd.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable if a == PYTHON_TOKEN else a for a in spec["argv"]]
        env = dict(os.environ, **spec["env"], **thread_environment(spec["resources"]["cpus"], job["gpu_ids"]))
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        log = (cwd / spec["log"]).open("w", encoding="utf-8")
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
            else {"start_new_session": True}
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        except BaseException:
            log.close()
            raise
        self._local[job["id"]] = {"process": process, "log": log, "start": time.monotonic(),
                                  "deadline": time.monotonic() + spec["resources"]["wall_seconds"]}
        return self._update(job["id"], status="running", started_at=self.clock(), process_pid=process.pid)

    def _launch_slurm(self, job, spec):
        backend = self.config["backends"][job["backend"]]
        # sbatch copies the script at submission, so it stays with the controller's session records.
        script = self.session_root / "slurm" / f"{job['id']}.{spec['name']}.sbatch"
        script.parent.mkdir(exist_ok=True)
        script.write_text(render_sbatch(spec, backend), encoding="utf-8", newline="\n")
        native = self.runner(["sbatch", "--parsable", str(script)]).strip().split(";")[0]
        if not native.isdigit():
            raise RuntimeError(f"Unexpected sbatch reply: {native!r}")
        return self._update(job["id"], status="queued", native_id=native, script=str(script),
                            script_sha256=sha256(script))

    def poll(self, job_id):
        job = self.job(job_id)
        if job["status"] in TERMINAL:
            return job
        return self._poll_local(job) if job["backend_type"] == "local" else self._poll_slurm(job)

    def _poll_local(self, job):
        handle = self._local.get(job["id"])
        if handle is None:
            raise ValueError(f"{job['id']} is a local job owned by controller pid {job['controller_pid']}")
        process = handle["process"]
        code = process.poll()
        if code is None and time.monotonic() < handle["deadline"]:
            return job
        if code is None:
            stop_process_tree(process)
            status = "timeout"
        else:
            status = "completed" if code == 0 else "failed"
        return self._finish_local(job["id"], status, process.returncode)

    def _finish_local(self, job_id, status, code):
        handle = self._local.pop(job_id)
        handle["log"].close()
        return self._update(job_id, status=status, exit_code=code, finished_at=self.clock(),
                            elapsed_seconds=time.monotonic() - handle["start"])

    def _poll_slurm(self, job):
        backend = self.config["backends"][job["backend"]]
        native = job["native_id"]
        reply = self.runner(["sacct", "-j", native, "-X", "-n", "-P", "-o", "JobIDRaw,State,ExitCode,ElapsedRaw"])
        rows = [line.split("|") for line in reply.splitlines() if line.startswith(native + "|")]
        state = rows[0][1].split()[0] if rows else None
        if state is None:
            # Accounting can lag submission; squeue still knows a live job.
            state = self.runner(["squeue", "-h", "-j", native, "-o", "%T"]).strip() or None
        now = self.clock()
        if state in SLURM_FINAL:
            exit_code = int(rows[0][2].split(":")[0]) if rows and rows[0][2] else None
            elapsed = float(rows[0][3]) if rows and rows[0][3] else None
            status = SLURM_FINAL[state]
            if status == "completed" and exit_code:
                status = "failed"
            return self._update(job["id"], status=status, native_state=state, exit_code=exit_code,
                                elapsed_seconds=elapsed, finished_at=now)
        if state == "RUNNING" and not job.get("started_at"):
            job = self._update(job["id"], status="running", started_at=now, native_state=state)
        elif state and state not in SLURM_ACTIVE:
            return self._update(job["id"], status="unknown", native_state=state, finished_at=now,
                                error=f"Unrecognized Slurm state {state}")
        if not job.get("started_at") and now - job["submitted_at"] > backend["max_queue_seconds"]:
            self.runner(["scancel", native])
            return self._update(job["id"], status="queue_timeout", native_state=state, finished_at=now)
        limit = backend["max_queue_seconds"] + job["resources"]["wall_seconds"] + 600
        if now - job["submitted_at"] > limit:
            self.runner(["scancel", native])
            return self._update(job["id"], status="unknown", native_state=state, finished_at=now,
                                error="Slurm did not report a final state within the reserved window")
        return job

    def wait(self, job_id):
        """Block until the job is final. Interrupting kills a local job; a Slurm job keeps running."""
        job = self.job(job_id)
        interval = self.config["backends"][job["backend"]].get("poll_seconds", 30) \
            if job["backend_type"] == "slurm" else 0.2
        try:
            while True:
                job = self.poll(job_id)
                if job["status"] in TERMINAL:
                    return job
                self.sleep(interval)
        except BaseException:
            if job_id in self._local:
                stop_process_tree(self._local[job_id]["process"])
                self._finish_local(job_id, "interrupted", self._local[job_id]["process"].returncode)
            raise

    def run(self, spec, backend=None, capacity_timeout=0.0):
        return self.wait(self.submit(spec, backend, capacity_timeout)["id"])

    def cancel(self, job_id):
        job = self.job(job_id)
        if job["status"] in TERMINAL:
            return job
        if job["backend_type"] == "slurm":
            self.runner(["scancel", job["native_id"]])
            return self._poll_slurm(job)
        stop_process_tree(self._local[job_id]["process"])
        return self._finish_local(job_id, "cancelled", self._local[job_id]["process"].returncode)

    def status(self):
        ledger = self._read()
        return {"limits": ledger["limits"], "usage": self.usage(ledger),
                "jobs": [{k: j.get(k) for k in ("id", "name", "backend", "status", "native_id", "gpu_ids", "cwd",
                                                "elapsed_seconds", "charged")} for j in ledger["jobs"]]}


def run_tool_job(scheduler, spec, backend=None, capacity_timeout=0.0):
    """Run one tool worker and map its final state onto the tools' exception contract.

    Tools keep their own manifests: a completed job returns its ledger record, a timeout raises
    TimeoutError, and any other final state raises RuntimeError. Interrupting the wait still kills a
    local job (KeyboardInterrupt propagates), as in the tools' own subprocess paths.
    """
    job = scheduler.run(spec, backend, capacity_timeout)
    if job["status"] == "completed":
        return job
    detail = f"{spec['name']} ended as {job['status']} on {job['backend']} ({job['id']})"
    if job["status"] in {"timeout", "queue_timeout"}:
        raise TimeoutError(detail)
    raise RuntimeError(f"{detail}; exit code {job.get('exit_code')}")


def probe():
    """Describe locally visible compute without launching scientific work."""
    record = {"host": socket.gethostname(), "platform": sys.platform, "logical_cpus": os.cpu_count(),
              "python": sys.executable, "slurm": {c: shutil.which(c) for c in ("sbatch", "sacct", "squeue", "scancel")},
              "gpus": []}
    if shutil.which("nvidia-smi"):
        try:
            out = default_runner(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
            record["gpus"] = [dict(zip(("index", "name", "memory_mib"), (p.strip() for p in line.split(","))))
                              for line in out.splitlines() if line.strip()]
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            record["gpu_error"] = str(exc)
    return record

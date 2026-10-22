"""Durable reservations and bounded, stdin-gated scientific subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rsi.backends.codex_cli import _WindowsJob
from rsi.experiment import write_json

ROOT = Path(__file__).resolve().parents[2]


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"entries": [], "blocked": None}
        self.save()

    def save(self):
        write_json(self.path, self.data)

    def reserve(self, stage, label, seconds):
        rows = self.data["entries"]
        if sum(r.get("evaluations", 1) for r in rows) >= 1000:
            raise RuntimeError("Scientific action-count limit")
        if self.data["blocked"] or any(r["status"] == "reserved" for r in rows):
            raise RuntimeError("Blocked or unresolved budget reservation")
        caps = {"source": 0, "reference": 48, "registered_reference": 48, "small": 256}
        units = 2 if stage == "reference" else 1
        if stage in caps and sum(r.get("evaluations", 2 if stage == "reference" else 1)
                                 for r in rows if r["stage"] == stage and r.get("work_started", True)) + units > caps[stage]:
            raise RuntimeError("Stage budget exhausted")
        if sum(r.get("wall_seconds", 0) for r in rows) + seconds > 600:
            raise RuntimeError("Scientific wall-time budget exhausted")
        row = dict(stage=stage, label=label, status="reserved", timeout_seconds=seconds)
        row["evaluations"] = units
        rows.append(row)
        self.save()
        return row

    def run(self, stage, label, payload, seconds=30):
        row = self.reserve(stage, label, seconds)
        start = time.monotonic()
        process = job = None
        result = None
        try:
            if os.name != "nt":
                raise RuntimeError("This controller requires Windows Job memory containment")
            libraries = str(Path(sys.prefix) / "Lib/site-packages")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "Crystal-Builder"), libraries]), PYTHONUTF8="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            job = _WindowsJob()
            info = job.jobs.QueryInformationJobObject(job.handle, job.jobs.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] |= (
                job.jobs.JOB_OBJECT_LIMIT_PROCESS_MEMORY | job.jobs.JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
            info["BasicLimitInformation"]["ActiveProcessLimit"] = 1
            info["ProcessMemoryLimit"] = 2 * 1024 ** 3
            job.jobs.SetInformationJobObject(job.handle, job.jobs.JobObjectExtendedLimitInformation, info)
            process = subprocess.Popen([sys._base_executable, "-m", "rsi.q2c_v2_explicit.worker"], cwd=ROOT, env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8")
            job.assign(process)
            out, err = process.communicate(json.dumps(payload, allow_nan=False), timeout=seconds)
            info = job.jobs.QueryInformationJobObject(job.handle, job.jobs.JobObjectExtendedLimitInformation)
            row["peak_process_memory_bytes"] = info["PeakProcessMemoryUsed"]
            row["stderr"] = err[-4000:]
            if process.returncode:
                raise RuntimeError(f"Worker exited {process.returncode}: {err[-2000:]}")
            result = json.loads(out)
            row["status"] = "ok" if result.get("ok") else "error"
            row["error"] = result.get("error")
        except Exception as exc:
            row.update(status="error", error=str(exc))
        finally:
            if job:
                job.close()
            if process and process.poll() is None:
                process.kill()
            if process:
                process.communicate()
            row["wall_seconds"] = time.monotonic() - start
            self.save()
        if row["status"] != "ok":
            raise RuntimeError(row["error"])
        return result["value"]

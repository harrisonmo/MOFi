"""Unified scheduler: job contract, local enforcement, Slurm protocol and conservative budgets."""

import json

import pytest

from rsi.hpc import core


PRINT_ENV = ("import os, json; print(json.dumps({k: os.environ.get(k) for k in "
             "('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'EXTRA')}))")


def config(tmp_path, **budget):
    value = {"schema_version": 1,
             "backends": {"local": {"type": "local", "max_cpus": 4, "gpu_ids": ["0", "1"]},
                          "cluster": {"type": "slurm", "partition": "shared", "account": "proj", "max_cpus": 8,
                                      "max_gpus": 2, "max_wall_seconds": 3600, "max_queue_seconds": 600,
                                      "poll_seconds": 1, "python": "/opt/py/bin/python",
                                      "env": {"PYTHONPATH": "/home/u/repo"}, "setup": ["module load gcc"]}},
             "session_budget": {"max_jobs": 10, "max_cpu_seconds": 1000, "max_gpu_seconds": 100,
                                "max_wall_seconds": 1000, **budget}}
    path = tmp_path / "hpc.json"
    core.write_json(path, value)
    return path


def spec(tmp_path, code="print('ok')", name="probe", **resources):
    return {"name": name, "argv": [core.PYTHON_TOKEN, "-c", code], "cwd": str(tmp_path / "work"),
            "resources": {"cpus": 1, "wall_seconds": 20, **resources}}


@pytest.mark.parametrize("change", [{"argv": "python -c 1"}, {"argv": []}, {"cwd": "relative"},
    {"env": {"OMP_NUM_THREADS": "64"}}, {"log": "../escape.log"}, {"name": "bad name"},
    {"resources": {"cpus": 0, "wall_seconds": 1}}, {"resources": {"cpus": 1, "wall_seconds": float("inf")}},
    {"resources": {"cpus": True, "wall_seconds": 1}}, {"shell": True}])
def test_spec_rejects_ambiguous_jobs(tmp_path, change):
    with pytest.raises(ValueError):
        core.validate_spec({**spec(tmp_path), **change})


def test_local_job_is_pinned_logged_and_charged_by_measurement(tmp_path):
    scheduler = core.Scheduler(config(tmp_path), tmp_path / "session")
    job_spec = spec(tmp_path, PRINT_ENV, cpus=2, gpus=1)
    job_spec["env"] = {"EXTRA": "yes"}
    job = scheduler.run(job_spec)
    assert job["status"] == "completed" and job["exit_code"] == 0 and job["gpu_ids"] == ["0"]
    seen = json.loads((tmp_path / "work" / "probe.log").read_text(encoding="utf-8"))
    assert seen == {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0", "EXTRA": "yes"}
    assert job["charged"]["cpu_seconds"] == pytest.approx(2 * job["elapsed_seconds"])
    assert job["elapsed_seconds"] < 20


def test_cpu_only_job_cannot_see_gpus(tmp_path):
    scheduler = core.Scheduler(config(tmp_path), tmp_path / "session")
    scheduler.run(spec(tmp_path, PRINT_ENV))
    assert json.loads((tmp_path / "work" / "probe.log").read_text())["CUDA_VISIBLE_DEVICES"] == ""


def test_timeout_kills_process_tree_and_failure_is_recorded(tmp_path):
    scheduler = core.Scheduler(config(tmp_path), tmp_path / "session")
    child = ("import pathlib, subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', "
             "'import time; time.sleep(60)']); pathlib.Path('grandchild.pid').write_text(str(p.pid)); time.sleep(60)")
    job = scheduler.run(spec(tmp_path, child, wall_seconds=3))
    assert job["status"] == "timeout" and job["elapsed_seconds"] < 30
    assert not core.pid_alive(int((tmp_path / "work" / "grandchild.pid").read_text()))
    failed = scheduler.run(spec(tmp_path, "raise SystemExit(3)", name="bad"))
    assert failed["status"] == "failed" and failed["exit_code"] == 3


def test_budget_is_checked_before_launch_and_persists(tmp_path):
    path = config(tmp_path, max_cpu_seconds=30)
    scheduler = core.Scheduler(path, tmp_path / "session")
    with pytest.raises(ValueError, match="cpu_seconds"):
        scheduler.submit(spec(tmp_path, cpus=2, wall_seconds=20))
    assert not (tmp_path / "work").exists() and scheduler.status()["jobs"] == []
    scheduler.run(spec(tmp_path, wall_seconds=20))
    reopened = core.Scheduler(path, tmp_path / "session")
    assert reopened.status()["usage"]["jobs"] == 1


def test_config_change_cannot_reset_session(tmp_path):
    path = config(tmp_path)
    core.Scheduler(path, tmp_path / "session")
    config(tmp_path, max_jobs=99)
    with pytest.raises(ValueError, match="cannot be reset"):
        core.Scheduler(path, tmp_path / "session")


def test_concurrent_local_jobs_share_cpus_and_distinct_gpus(tmp_path):
    scheduler = core.Scheduler(config(tmp_path), tmp_path / "session")
    sleeper = "import time; time.sleep(3)"
    first = scheduler.submit(spec(tmp_path, sleeper, name="a", cpus=2, gpus=1))
    second = scheduler.submit(spec(tmp_path, sleeper, name="b", cpus=2, gpus=1))
    assert {tuple(first["gpu_ids"]), tuple(second["gpu_ids"])} == {("0",), ("1",)}
    with pytest.raises(ValueError, match="no free capacity"):
        scheduler.submit(spec(tmp_path, sleeper, name="c", cpus=1))
    assert scheduler.wait(first["id"])["status"] == "completed"
    third = scheduler.submit(spec(tmp_path, "pass", name="c", gpus=1))
    assert third["gpu_ids"] == first["gpu_ids"]
    assert scheduler.wait(second["id"])["status"] == scheduler.wait(third["id"])["status"] == "completed"


def test_dead_controller_reservation_is_charged_in_full(tmp_path):
    path = config(tmp_path)
    scheduler = core.Scheduler(path, tmp_path / "session")
    ledger = core.read_json(scheduler.ledger_path)
    ledger["jobs"].append({"id": "job-0001", "name": "lost", "backend": "local", "backend_type": "local",
                           "host": scheduler.host, "controller_pid": 2 ** 30, "process_pid": 2 ** 30,
                           "resources": {"cpus": 4, "gpus": 0, "memory_mb": None, "wall_seconds": 50},
                           "gpu_ids": [], "status": "running", "submitted_at": 0, "elapsed_seconds": None})
    core.write_json(scheduler.ledger_path, ledger)
    job = core.Scheduler(path, tmp_path / "session").job("job-0001")
    assert job["status"] == "unknown" and job["charged"]["cpu_seconds"] == 200
    # The orphan pid is gone, so its CPUs are free again.
    assert scheduler.run(spec(tmp_path, cpus=4))["status"] == "completed"


class FakeSlurm:
    def __init__(self, states):
        self.states, self.calls = list(states), []

    def __call__(self, argv, timeout=60):
        self.calls.append(argv)
        if argv[0] == "sbatch":
            return "4242;cluster\n"
        if argv[0] == "sacct":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return state and f"4242|{state}|0:0|37\n"
        return ""


def test_sbatch_script_is_quoted_and_pinned(tmp_path):
    backend = core.validate_config(core.read_json(config(tmp_path)))["backends"]["cluster"]
    job = core.validate_spec({**spec(tmp_path, cpus=4, gpus=1, memory_mb=8000), "cwd": "/scratch/u/run",
                              "argv": [core.PYTHON_TOKEN, "-c", "print('a b; rm -rf ~')"]})
    script = core.render_sbatch(job, backend)
    for line in ("#SBATCH --cpus-per-task=4", "#SBATCH --gres=gpu:1", "#SBATCH --mem=8000M",
                 "#SBATCH --time=0-00:01:00", "#SBATCH --account=proj", "export OMP_NUM_THREADS=4",
                 "module load gcc", "#SBATCH --output=/scratch/u/run/probe.log",
                 "mkdir -p /scratch/u/run && cd /scratch/u/run", "exec /opt/py/bin/python -c 'print('\"'\"'a b; rm -rf ~'\"'\"')'"):
        assert line in script.splitlines()
    with pytest.raises(ValueError, match="whitespace"):
        core.render_sbatch({**job, "cwd": "/scratch/my run"}, backend)


@pytest.mark.parametrize("final,status", [("COMPLETED", "completed"), ("TIMEOUT", "timeout"),
                                          ("OUT_OF_MEMORY", "failed"), ("CANCELLED by 7", "cancelled")])
def test_slurm_lifecycle_and_restart(tmp_path, final, status):
    path = config(tmp_path)
    fake = FakeSlurm(["", "PENDING", "RUNNING", final])
    scheduler = core.Scheduler(path, tmp_path / "session", runner=fake, sleep=lambda s: None)
    job = scheduler.submit({**spec(tmp_path, cpus=2), "cwd": "/scratch/u/run"}, backend="cluster")
    assert job["status"] == "queued" and job["native_id"] == "4242"
    # A fresh controller can resume waiting because Slurm, not the process, owns the job.
    restarted = core.Scheduler(path, tmp_path / "session", runner=fake, sleep=lambda s: None)
    done = restarted.wait(job["id"])
    assert done["status"] == status and done["elapsed_seconds"] == 37
    assert done["charged"]["cpu_seconds"] == 74


def test_slurm_queue_timeout_cancels_and_charges_reservation(tmp_path):
    times = [0, 0, 700]
    clock = lambda: times.pop(0) if len(times) > 1 else times[0]
    fake = FakeSlurm(["PENDING"])
    scheduler = core.Scheduler(config(tmp_path), tmp_path / "session", runner=fake, clock=clock, sleep=lambda s: None)
    job = scheduler.submit({**spec(tmp_path, wall_seconds=60), "cwd": "/scratch/u/run"},
                           backend="cluster")
    done = scheduler.wait(job["id"])
    assert done["status"] == "queue_timeout" and ["scancel", "4242"] in fake.calls
    assert done["charged"]["wall_seconds"] == 60

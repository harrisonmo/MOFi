# Unified HPC scheduling layer

`rsi.hpc` is the shared execution layer for scientific tools. A tool describes
*what* to run; the controller-owned scheduler decides *where* and *how*, and
records what it cost. It is engineering infrastructure: it makes no scientific
claim and does not change frozen M7/M11 actions or any registered tool.

## Job contract

A job spec is backend-neutral JSON:

```json
{"name": "mof-relax-01",
 "argv": ["{python}", "-m", "rsi.dft.worker", "/abs/run", "attempt-001"],
 "cwd": "/abs/run/attempt-001",
 "env": {"EXTRA_SETTING": "1"},
 "log": "worker.log",
 "resources": {"cpus": 8, "gpus": 0, "memory_mb": 16000, "wall_seconds": 3600}}
```

- `argv` is a list and is never passed through a shell. `{python}` becomes the
  controller interpreter locally and the backend's `python` on Slurm.
- `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS` are set from `cpus`, and
  `CUDA_VISIBLE_DEVICES` from the GPUs actually allocated (empty for CPU-only
  jobs). Specs may not set these variables themselves.
- `log` is relative to `cwd`; stdout and stderr are merged into it.

## Backends

Configured by the controller (example: [configs/hpc-backends.json](../configs/hpc-backends.json)).

| | local | slurm |
|---|---|---|
| Launch | `Popen` in a new process group | `sbatch --parsable` of a rendered script |
| Wall limit | Controller kills the whole process tree | `--time`, plus a controller safety window |
| Capacity | `max_cpus`, `gpu_ids`, optional `max_memory_mb` shared by every controller using the same session | Per-job limits; the cluster queues |
| Restart | A dead controller's job becomes `unknown` | Any controller can resume `wait` from Slurm accounting |

Memory is advisory locally (admission only); Slurm enforces `--mem`. The
Slurm controller must run where `sbatch/sacct/squeue/scancel` exist (a login
node). Rendered scripts are kept under `<session>/slurm/` with their hashes in
the ledger. Directive values (job name, cwd, log, partition, extras) must not
contain whitespace or quotes; command arguments are shell-quoted.

Slurm states map to `completed`, `failed` (including `OUT_OF_MEMORY`, `NODE_FAIL`
and a nonzero exit code), `timeout`, or `cancelled`. A job still pending after
`max_queue_seconds` is cancelled as `queue_timeout`. A job with no final state
by queue limit + wall + 600 s is cancelled as `unknown`.

## Session ledger and budget

Each session directory holds `hpc-ledger.json`, bound to the config hash. A
changed config cannot reopen the session, so a budget cannot be reset.
Admission happens under a file lock before anything launches:

- active jobs count at their full reservation (`cpus x wall`, `gpus x wall`, `wall`, one job);
- `completed/failed/timeout/cancelled` are charged by measured elapsed time;
- `interrupted`, `unknown` and `queue_timeout` are charged at the full reservation.

Concurrent controllers (for example several research threads on one
workstation) can share one session: CPUs and GPU IDs are allocated without
overlap, and `capacity_timeout` lets a submission wait for a free slot. An
orphaned local child whose controller died keeps its CPUs and GPUs until its
process exits.

## Usage

```bash
python -m rsi.hpc probe
python -m rsi.hpc run --config configs/hpc-backends.json --session runs/hpc-session --spec job.json
python -m rsi.hpc status --config configs/hpc-backends.json --session runs/hpc-session
python -m rsi.hpc submit --config cluster.json --session runs/s --spec job.json --backend slurm-example
python -m rsi.hpc wait --config cluster.json --session runs/s --job job-0001
```

In Python:

```python
from rsi.hpc import Scheduler
scheduler = Scheduler("configs/hpc-backends.json", "runs/hpc-session")
job = scheduler.run(spec)                    # blocking
handle = scheduler.submit(spec, "local", capacity_timeout=600)
job = scheduler.wait(handle["id"])           # local jobs: same Scheduler object
```

Interrupting `wait` kills a local job (recorded `interrupted`); a Slurm job keeps
running and can be awaited again from a new controller.

## Adopting the layer in tools

GCMC, DFT, Zeo++ and OpenMM accept an optional scheduler. Without one, each keeps its original local
subprocess path, so frozen protocols and their tests are unchanged.

| Tool | Entry point | Job per | Resources |
|---|---|---|---|
| GCMC | `rsi.gcmc.core.execute(run, scheduler=...)` | pressure x seed worker | 1 CPU, `min(timeout_seconds_per_job, remaining)` |
| DFT | `rsi.dft.core.execute(run, scheduler=...)` | attempt worker | `budget.threads` CPUs, attempt timeout |
| Zeo++ | `rsi.zeopp.core.execute(run, scheduler=...)` | `network` call | 1 CPU, recipe timeout |
| OpenMM | `rsi.openmm_engine.run_isolated(..., scheduler=...)` | minimization child | `threads` CPUs (default 1), `timeout_seconds` |

Final states map onto each tool's own contract: `timeout`/`queue_timeout` raise `TimeoutError`, a
`failed` job is a nonzero exit (Slurm failures that report exit code 0 become -1), and `cancelled`
or `unknown` stop the run as interrupted. The HPC job ID and charge are recorded in the tool manifest.
`rsi.hpc.run_tool_job` wraps this mapping for tools without their own manifest fields. On Slurm the
controller and compute nodes must share the run directories and the backend `setup`/`env` must make
`rsi` importable and engines available. The [science hub](SCIENCE_HUB.md) runs every tool this way.

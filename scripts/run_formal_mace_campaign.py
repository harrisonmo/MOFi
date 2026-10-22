#!/usr/bin/env python3
"""Run the sequential, auditable formal MACE screening campaign.

The launcher is intentionally conservative: it waits for an upstream sentinel,
runs one GPU job at a time, records the exact command/environment, stops on a
failed arm, and never restarts or deletes a run automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    r_max: float
    foundation: str | None = None
    e0s: str = "average"
    lr: float = 0.01
    ema_decay: float = 0.99
    clip_grad: float = 10.0
    lora_rank: int | None = None
    lora_alpha: float | None = None
    freeze: int | None = None
    multiheads: bool = False
    pseudolabel_replay: bool = False


def build_specs(seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in seeds:
        for r_max in (5.0, 6.0, 7.0):
            specs.append(RunSpec("scratch", seed, r_max))
    for seed in seeds:
        specs.extend(
            [
                RunSpec("naive_estimated", seed, 6.0, "medium", "estimated", 1e-3, 0.999, 1.0),
                RunSpec("lora_rank4", seed, 6.0, "medium", "estimated", 5e-3, 0.995, 10.0, 4, 1.0),
                RunSpec("lora_rank16", seed, 6.0, "medium", "estimated", 5e-3, 0.995, 10.0, 16, 4.0),
                RunSpec("lora_rank64", seed, 6.0, "medium", "estimated", 5e-3, 0.995, 10.0, 64, 16.0),
                RunSpec("layer_freeze5", seed, 6.0, "medium", "estimated", 1e-3, 0.999, 1.0, freeze=5),
                RunSpec(
                    "multihead_pseudolabel_replay",
                    seed,
                    6.0,
                    "medium",
                    "estimated",
                    1e-4,
                    0.99999,
                    1.0,
                    multiheads=True,
                    pseudolabel_replay=True,
                ),
            ]
        )
    return specs


def command_for(spec: RunSpec, args: argparse.Namespace, run_dir: Path) -> list[str]:
    name = f"{spec.method}_rmax{spec.r_max:g}_seed{spec.seed}"
    command = [
        args.python,
        str(args.runner),
        "--name",
        name,
        "--work_dir",
        str(run_dir),
        "--log_dir",
        str(run_dir / "logs"),
        "--model_dir",
        str(run_dir / "model"),
        "--checkpoints_dir",
        str(run_dir / "checkpoints"),
        "--results_dir",
        str(run_dir / "results"),
        "--device",
        "cuda",
        "--default_dtype",
        "float32",
        "--r_max",
        str(spec.r_max),
        "--train_file",
        str(args.train_file),
        "--valid_file",
        str(args.valid_file),
        "--test_file",
        str(args.test_file),
        "--E0s",
        spec.e0s,
        "--energy_key",
        "REF_energy",
        "--forces_key",
        "REF_forces",
        "--loss",
        "weighted",
        "--energy_weight",
        "10",
        "--forces_weight",
        "10",
        "--batch_size",
        "1",
        "--valid_batch_size",
        "1",
        "--lr",
        str(spec.lr),
        "--weight_decay",
        "0.0",
        "--ema",
        "--ema_decay",
        str(spec.ema_decay),
        "--clip_grad",
        str(spec.clip_grad),
        "--max_num_epochs",
        str(args.epochs),
        "--seed",
        str(spec.seed),
        "--plot",
        "false",
        "--save_all_checkpoints",
    ]
    foundation = args.foundation_model_path if spec.multiheads else spec.foundation
    if foundation is not None:
        command.extend(["--foundation_model", str(foundation), "--foundation_model_elements", "true"])
        command.extend(["--multiheads_finetuning", str(spec.multiheads).lower()])
    if spec.lora_rank is not None:
        command.extend(["--lora", "true", "--lora_rank", str(spec.lora_rank), "--lora_alpha", str(spec.lora_alpha)])
    if spec.freeze is not None:
        command.extend(["--freeze", str(spec.freeze)])
    if spec.multiheads:
        command.extend(
            [
                "--pseudolabel_replay",
                "true",
                "--pt_train_file",
                str(args.replay_file),
                "--pt_valid_file",
                str(args.replay_file),
            ]
        )
    return command


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--foundation-model-path", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261017, 20261018, 20261019])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    campaign_dir = args.root / "formal_screening"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.seeds)
    manifest = {
        "campaign_id": "mof-mace-formal-screening-20261017",
        "status": "WAITING_FOR_UPSTREAM_SENTINEL",
        "created_at": utc_now(),
        "runtime": {
            "python": args.python,
            "runner": str(args.runner),
            "foundation_model_path": str(args.foundation_model_path),
            "mace_site": str(args.site),
            "python_version": platform.python_version(),
            "host": platform.node(),
            "epochs_per_screening_arm": args.epochs,
            "seeds": args.seeds,
        },
        "upstream_sentinel": str(args.wait_for),
        "source_files": {
            "train": str(args.train_file),
            "valid": str(args.valid_file),
            "test": str(args.test_file),
            "replay": str(args.replay_file),
        },
        "runs": [
            {
                "run_id": f"{spec.method}_rmax{spec.r_max:g}_seed{spec.seed}",
                "spec": asdict(spec),
                "status": "PENDING",
            }
            for spec in specs
        ],
    }
    state_path = campaign_dir / "campaign_state.json"
    write_json(state_path, manifest)

    while not args.wait_for.exists():
        time.sleep(args.poll_seconds)
    manifest["status"] = "RUNNING"
    manifest["upstream_sentinel_observed_at"] = utc_now()
    write_json(state_path, manifest)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.site}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"
    for index, spec in enumerate(specs):
        run_id = f"{spec.method}_rmax{spec.r_max:g}_seed{spec.seed}"
        run_dir = campaign_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("logs", "model", "checkpoints", "results"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        command = command_for(spec, args, run_dir)
        write_json(run_dir / "config.json", {"run_id": run_id, "spec": asdict(spec), "command": command})
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env_payload = {
            "created_at": utc_now(),
            "python": args.python,
            "pythonpath": env["PYTHONPATH"],
            "device": "cuda",
            "campaign_state_sha256_before_run": sha256(state_path),
        }
        write_json(run_dir / "environment.json", env_payload)
        manifest["runs"][index].update({"status": "RUNNING", "started_at": utc_now()})
        write_json(state_path, manifest)
        log_path = run_dir / "train.log"
        exit_code = 1
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"COMMAND: {' '.join(command)}\n")
            log.write(f"STARTED_AT: {utc_now()}\n")
            log.flush()
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            exit_code = completed.returncode
        result = {
            "run_id": run_id,
            "finished_at": utc_now(),
            "exit_code": exit_code,
            "duration_seconds": round(time.time() - started, 3),
            "train_log_sha256": sha256(log_path),
            "training_complete_marker": "Training complete" in log_path.read_text(encoding="utf-8", errors="replace"),
        }
        write_json(run_dir / "result.json", result)
        manifest["runs"][index].update(
            {"status": "COMPLETED" if exit_code == 0 else "FAILED", "finished_at": result["finished_at"], "exit_code": exit_code}
        )
        write_json(state_path, manifest)
        if exit_code != 0:
            manifest["status"] = "FAILED_STOPPED_NO_AUTO_RESTART"
            write_json(state_path, manifest)
            return exit_code

    manifest["status"] = "COMPLETED"
    manifest["completed_at"] = utc_now()
    write_json(state_path, manifest)
    (campaign_dir / "campaign.done").write_text(manifest["completed_at"] + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

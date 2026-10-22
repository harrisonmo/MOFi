#!/usr/bin/env python3
"""Execute one immutable cumulative-training node of the five-round MLIP loop.

This runner resolves a hash-verified precomputed selection-tree prefix.  It
does not receive an independent test path; validation is the only metric input
available before a final evaluator-release barrier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.protocol import require_known_arm
from scripts.run_micro_mlip_round import build_command, model_hashes


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified_file(dataset_dir: Path, entry: Mapping[str, Any], label: str) -> Path:
    relative_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"Invalid {label} receipt")
    path = dataset_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if sha256(path) != expected_hash:
        raise ValueError(f"Hash mismatch for {label}: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "scratch_micro",
            "foundation_naive",
            "lora_rank4",
            "layer_freeze5",
            "multihead_pseudolabel_replay",
        ),
        required=True,
    )
    parser.add_argument("--arm", choices=("fixed", "outer_generated", "manual_control", "removed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--replay-valid-file", type=Path)
    parser.add_argument("--mace-site", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arm = require_known_arm(args.arm)
    config = load_json(args.config)
    if args.method not in config["scientific_environment"]["methods"]:
        raise ValueError(f"Method {args.method} is not registered")
    method = config["scientific_environment"]["methods"][args.method]
    if not str(method.get("status", "")).startswith("READY"):
        raise ValueError(f"Method {args.method} is not ready: {method.get('status')}")
    if args.molecule not in config["molecule_roles"]:
        raise ValueError(f"Molecule {args.molecule} is not registered")
    if args.seed not in config["scientific_environment"]["training"]["seeds"]:
        raise ValueError(f"Seed {args.seed} is not registered")

    manifest_path = args.dataset_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError("Dataset and five-round configuration benchmark IDs differ")
    try:
        molecule_entry = manifest["molecules"][args.molecule]
        prefix_entry = molecule_entry["episodes"]["prefixes"][args.prefix]
        validation_entry = molecule_entry["files"]["valid"]
    except KeyError as exc:
        raise ValueError(f"Unknown registered five-round selection node: {args.molecule}/{args.prefix}") from exc
    if prefix_entry.get("round_index") != args.round_index:
        raise ValueError("Requested round index does not match the immutable selection-tree node")
    train_file = verified_file(args.dataset_dir, prefix_entry, "selected train file")
    valid_file = verified_file(args.dataset_dir, validation_entry, "validation file")

    safe_prefix = args.prefix.replace("__", "-")
    derived_id = "__".join(
        (args.molecule, args.arm, args.method, f"round{args.round_index}", safe_prefix, f"seed{args.seed}")
    )
    run_id = args.run_id or derived_id
    if not run_id.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise ValueError("run-id may contain only letters, numbers, underscores, dashes, and periods")
    run_dir = args.output_root / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
        return 0
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse incomplete run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    for name in ("logs", "model", "checkpoints", "results"):
        (run_dir / name).mkdir()

    training = config["scientific_environment"]["training"]
    command = build_command(args, training, method, run_dir, train_file, valid_file, run_id)
    source_hashes: Dict[str, str] = {
        "config": sha256(args.config),
        "dataset_manifest": sha256(manifest_path),
        "selected_train": sha256(train_file),
        "validation": sha256(valid_file),
        "runner": sha256(args.runner) if args.runner.is_file() else "MISSING_AT_PREPARATION",
    }
    if args.foundation_model is not None:
        source_hashes["foundation_model"] = sha256(args.foundation_model)
    if args.replay_file is not None:
        source_hashes["replay_file"] = sha256(args.replay_file)
    run_receipt = {
        "schema": "micro-mlip-five-round-training-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "benchmark_id": config["benchmark_id"],
        "run_id": run_id,
        "arm": args.arm,
        "arm_definition": arm,
        "molecule": args.molecule,
        "molecule_role": config["molecule_roles"][args.molecule],
        "method": args.method,
        "method_definition": method,
        "round_index": args.round_index,
        "selection_prefix": args.prefix,
        "selection_history": prefix_entry["action_history"],
        "label_budget": prefix_entry["frames"],
        "seed": args.seed,
        "inputs": {
            "selected_train": str(train_file),
            "validation": str(valid_file),
            "test_policy": "NOT_PASSED_TO_TRAINING_COMMAND",
        },
        "source_sha256": source_hashes,
        "command": command,
    }
    write_json(run_dir / "run.json", run_receipt)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    write_json(
        run_dir / "environment.json",
        {
            "created_at": utc_now(),
            "python_executable": args.python,
            "python_version": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "device": args.device,
            "mace_site": str(args.mace_site) if args.mace_site is not None else None,
        },
    )
    if args.dry_run:
        result = {
            "status": "DRY_RUN_ONLY",
            "finished_at": utc_now(),
            "exit_code": None,
            "validation_evaluation": "NOT_STARTED",
            "independent_test_evaluation": "NOT_STARTED",
            "model_sha256": {},
        }
        write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    environment = os.environ.copy()
    if args.mace_site is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(args.mace_site), environment.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep)
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = run_dir / "train.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n")
        log.write(f"STARTED_AT: {utc_now()}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    result = {
        "status": "TRAINING_COMPLETED_PENDING_VALIDATION_EVALUATION"
        if completed.returncode == 0
        else "TRAINING_FAILED_NO_AUTO_RESTART",
        "finished_at": utc_now(),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "train_log_sha256": sha256(log_path),
        "model_sha256": model_hashes(run_dir / "model"),
        "validation_evaluation": "NOT_STARTED",
        "independent_test_evaluation": "NOT_STARTED",
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

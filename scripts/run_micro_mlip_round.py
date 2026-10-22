#!/usr/bin/env python3
"""Prepare or execute one immutable micro-MLIP MACE training round.

The training process receives a selected training file and validation file only.
It never receives the independent test set; use evaluate_micro_mlip_round.py
after a successful run to create test metrics.
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
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.protocol import require_known_arm, require_runnable_method


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_entry(manifest: Mapping[str, Any], molecule: str, name: str) -> Mapping[str, Any]:
    try:
        return manifest["molecules"][molecule]["files"][name]
    except KeyError as exc:
        raise ValueError(f"Dataset manifest is missing {molecule}/{name}") from exc


def selection_entry(
    manifest: Mapping[str, Any], molecule: str, strategy: str, label_budget: int
) -> Mapping[str, Any]:
    try:
        return manifest["molecules"][molecule]["selections"][strategy][str(label_budget)]
    except KeyError as exc:
        raise ValueError(
            f"Dataset manifest is missing {molecule}/{strategy}/{label_budget} selection"
        ) from exc


def verified_file(dataset_dir: Path, entry: Mapping[str, Any], label: str) -> Path:
    relative_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"Invalid {label} entry in dataset manifest")
    path = dataset_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed_hash = sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {label}: {path}")
    return path


def build_command(
    args: argparse.Namespace,
    training: Mapping[str, Any],
    method: Mapping[str, Any],
    run_dir: Path,
    train_file: Path,
    valid_file: Path,
    run_id: str,
) -> List[str]:
    """Build a MACE command without exposing the independent test artifact."""

    command = [
        args.python,
        str(args.runner),
        "--name",
        run_id,
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
        args.device,
        "--default_dtype",
        str(training["default_dtype"]),
        "--train_file",
        str(train_file),
        "--valid_file",
        str(valid_file),
        "--E0s",
        str(method["e0s"]),
        "--energy_key",
        str(training["energy_key"]),
        "--forces_key",
        str(training["forces_key"]),
        "--loss",
        str(training["loss"]),
        "--energy_weight",
        str(training["energy_weight"]),
        "--forces_weight",
        str(training["forces_weight"]),
        "--batch_size",
        str(training["batch_size"]),
        "--valid_batch_size",
        str(training["valid_batch_size"]),
        "--lr",
        str(method["lr"]),
        "--weight_decay",
        "0.0",
        "--clip_grad",
        str(method["clip_grad"]),
        "--max_num_epochs",
        str(training["max_num_epochs"]),
        "--patience",
        str(training["early_stopping_patience"]),
        "--seed",
        str(args.seed),
        "--plot",
        "false",
    ]
    if bool(method.get("ema", False)):
        command.extend(["--ema", "--ema_decay", str(method["ema_decay"])])
    if not bool(method["foundation"]):
        command.extend(
            [
                "--r_max",
                str(method["r_max_A"]),
                "--num_interactions",
                str(method["num_interactions"]),
                "--hidden_irreps",
                str(method["hidden_irreps"]),
            ]
        )
    else:
        if args.foundation_model is None:
            raise ValueError(f"{args.method} requires --foundation-model")
        if not args.foundation_model.is_file():
            raise FileNotFoundError(f"Foundation model does not exist: {args.foundation_model}")
        multiheads = bool(method.get("multiheads_finetuning", False))
        command.extend(
            [
                "--foundation_model",
                str(args.foundation_model),
                "--foundation_model_elements",
                "true",
                "--multiheads_finetuning",
                str(multiheads).lower(),
            ]
        )
        if multiheads:
            if args.replay_file is None:
                raise ValueError(f"{args.method} requires --replay-file")
            if not args.replay_file.is_file():
                raise FileNotFoundError(f"Replay file does not exist: {args.replay_file}")
            if bool(method.get("replay_validation_required", False)) and args.replay_valid_file is None:
                raise ValueError(f"{args.method} requires --replay-valid-file")
            replay_valid_file = args.replay_valid_file or args.replay_file
            if not replay_valid_file.is_file():
                raise FileNotFoundError(f"Replay validation file does not exist: {replay_valid_file}")
            command.extend(
                [
                    "--pseudolabel_replay",
                    "true",
                    "--pt_train_file",
                    str(args.replay_file),
                    "--pt_valid_file",
                    str(replay_valid_file),
                ]
            )
            if bool(method.get("measure_pretraining_forgetting", False)):
                # MACE otherwise skips ``pt_head`` during its final error
                # table.  Retaining it makes the immutable replay validation
                # file an observable forgetting metric rather than an unused
                # provenance attachment.
                command.extend(["--skip_evaluate_heads", ""])
    if "lora_rank" in method:
        command.extend(
            [
                "--lora",
                "true",
                "--lora_rank",
                str(method["lora_rank"]),
                "--lora_alpha",
                str(method["lora_alpha"]),
            ]
        )
    if "freeze" in method:
        command.extend(["--freeze", str(method["freeze"])])
    if "--test_file" in command:
        raise AssertionError("The training command must not receive a test file")
    return command


def model_hashes(model_dir: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(model_dir)): sha256(path)
        for path in sorted(model_dir.rglob("*.model"))
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--selection", choices=("random", "geometry_coverage"), required=True)
    parser.add_argument("--label-budget", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument(
        "--method",
        choices=("scratch_micro", "foundation_naive", "lora_rank4", "layer_freeze5", "multihead_pseudolabel_replay"),
        required=True,
    )
    parser.add_argument(
        "--arm",
        choices=("fixed", "outer_generated", "manual_control", "removed", "development_calibration"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--replay-valid-file", type=Path)
    parser.add_argument("--mace-site", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arm = require_known_arm(args.arm)
    require_runnable_method(args.method)
    config = load_json(args.config)
    if args.method not in config["scientific_environment"]["methods"]:
        raise ValueError(f"Method {args.method} is not registered in {args.config}")
    method = config["scientific_environment"]["methods"][args.method]
    if str(method.get("status", "")).startswith("DEFERRED"):
        raise ValueError(f"Method {args.method} is deferred in the frozen configuration")
    if args.molecule not in config["molecule_roles"]:
        raise ValueError(f"Molecule {args.molecule} is not registered in the frozen configuration")
    if args.seed not in config["scientific_environment"]["training"]["seeds"]:
        raise ValueError(f"Seed {args.seed} is not registered in the frozen configuration")

    manifest_path = args.dataset_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError("Dataset benchmark_id does not match the training configuration")
    train_entry = selection_entry(manifest, args.molecule, args.selection, args.label_budget)
    valid_entry = file_entry(manifest, args.molecule, "valid")
    train_file = verified_file(args.dataset_dir, train_entry, "selected train file")
    valid_file = verified_file(args.dataset_dir, valid_entry, "validation file")

    derived_id = "__".join(
        (args.molecule, args.arm, args.method, args.selection, str(args.label_budget), f"seed{args.seed}")
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
    if args.replay_valid_file is not None:
        source_hashes["replay_validation_file"] = sha256(args.replay_valid_file)
    run_receipt = {
        "schema": "micro-mlip-round-v1",
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
        "selection": args.selection,
        "label_budget": args.label_budget,
        "seed": args.seed,
        "inputs": {
            "selected_train": str(train_file),
            "validation": str(valid_file),
            "pretraining_replay_train": str(args.replay_file) if args.replay_file is not None else None,
            "pretraining_replay_validation": (
                str(args.replay_valid_file)
                if args.replay_valid_file is not None
                else (str(args.replay_file) if args.replay_file is not None else None)
            ),
            "test_policy": "NOT_PASSED_TO_TRAINING_COMMAND",
        },
        "source_sha256": source_hashes,
        "command": command,
    }
    write_json(run_dir / "run.json", run_receipt)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    environment = {
        "created_at": utc_now(),
        "python_executable": args.python,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "device": args.device,
        "mace_site": str(args.mace_site) if args.mace_site is not None else None,
    }
    write_json(run_dir / "environment.json", environment)

    if args.dry_run:
        result = {
            "status": "DRY_RUN_ONLY",
            "finished_at": utc_now(),
            "exit_code": None,
            "test_evaluation": "NOT_STARTED",
            "model_sha256": {},
        }
        write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    environment_variables = os.environ.copy()
    if args.mace_site is not None:
        environment_variables["PYTHONPATH"] = os.pathsep.join(
            (str(args.mace_site), environment_variables.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep)
    environment_variables["PYTHONUNBUFFERED"] = "1"
    log_path = run_dir / "train.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n")
        log.write(f"STARTED_AT: {utc_now()}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment_variables, check=False)
    result = {
        "status": "TRAINING_COMPLETED_PENDING_INDEPENDENT_TEST_EVALUATION"
        if completed.returncode == 0
        else "TRAINING_FAILED_NO_AUTO_RESTART",
        "finished_at": utc_now(),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "train_log_sha256": sha256(log_path),
        "model_sha256": model_hashes(run_dir / "model"),
        "test_evaluation": "NOT_STARTED",
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

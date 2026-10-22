#!/usr/bin/env python3
"""Admit foundation methods for the micro-MLIP model-axis extension.

This remote-side controller is deliberately separate from the P/D/T/W loop.
It runs one small, validation-only rMD17 CPU training for each foundation
method after the formal MOF campaign has released the shared machine.  An
immutable foundation artifact and distinct replay train/validation files are
required.  Multihead replay is admitted only if its MACE result log retains a
pretraining-head validation trace from which a forgetting delta can be read.

No independent rMD17 test set is opened here, no policy is selected, and a
failed admission never falls back to scratch or retries automatically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256


FOUNDATION_METHODS = (
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)
MULTIHEAD_METHOD = "multihead_pseudolabel_replay"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_sha256": sha256(log_path),
    }


def require_success(label: str, receipt: Mapping[str, Any]) -> None:
    if receipt.get("exit_code") != 0:
        raise RuntimeError(f"{label} failed; inspect receipt log {receipt.get('log_sha256')}")


def method_definitions(base_config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    methods = base_config.get("scientific_environment", {}).get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Base configuration has no method definitions")
    result: Dict[str, Dict[str, Any]] = {}
    for method in FOUNDATION_METHODS[:-1]:
        definition = methods.get(method)
        if not isinstance(definition, Mapping) or definition.get("foundation") is not True:
            raise ValueError(f"Base configuration does not define the foundation method {method}")
        copied = dict(definition)
        copied["status"] = "READY_FOR_FOUNDATION_ADMISSION_DRY_RUN_ONLY"
        result[method] = copied
    result[MULTIHEAD_METHOD] = {
        "status": "READY_FOR_FOUNDATION_ADMISSION_DRY_RUN_ONLY",
        "foundation": True,
        "multiheads_finetuning": True,
        "replay_validation_required": True,
        "measure_pretraining_forgetting": True,
        "e0s": "estimated",
        "lr": 0.0001,
        "ema": True,
        "ema_decay": 0.99999,
        "clip_grad": 1.0,
    }
    return result


def staged_config(
    base_config: Mapping[str, Any],
    *,
    foundation_hash: str,
    replay_hash: str,
    replay_validation_hash: str,
    dry_run_epochs: int,
) -> Dict[str, Any]:
    """Create an admission-only config without mutating the frozen base config."""

    if isinstance(dry_run_epochs, bool) or dry_run_epochs <= 0:
        raise ValueError("dry-run epochs must be positive")
    output = json.loads(json.dumps(base_config))
    environment = output.get("scientific_environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("training"), dict):
        raise ValueError("Base configuration has no training environment")
    original_methods = environment.get("methods")
    if not isinstance(original_methods, dict):
        raise ValueError("Base configuration has no mutable methods mapping")
    original_methods.update(method_definitions(base_config))
    environment["training"]["max_num_epochs"] = dry_run_epochs
    environment["training"]["early_stopping_patience"] = min(
        int(environment["training"].get("early_stopping_patience", dry_run_epochs)), dry_run_epochs
    )
    output["status"] = "DERIVED_FOR_FOUNDATION_METHOD_ADMISSION_DRY_RUN_ONLY"
    output["foundation_method_admission_plan"] = {
        "foundation_model_sha256": foundation_hash,
        "replay_train_sha256": replay_hash,
        "replay_validation_sha256": replay_validation_hash,
        "dry_run_epochs": dry_run_epochs,
        "test_access": "FORBIDDEN",
    }
    return output


def admitted_config(
    base_config: Mapping[str, Any],
    *,
    foundation_hash: str,
    replay_hash: str,
    replay_validation_hash: str,
    admission_id: str,
) -> Dict[str, Any]:
    """Create the only config eligible for model-axis five-round campaigns."""

    output = json.loads(json.dumps(base_config))
    environment = output.get("scientific_environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("methods"), dict):
        raise ValueError("Base configuration has no mutable methods mapping")
    methods = environment["methods"]
    for method, definition in method_definitions(base_config).items():
        ready = dict(definition)
        ready["status"] = "READY_AFTER_FOUNDATION_METHOD_ADMISSION"
        methods[method] = ready
    output["status"] = "ADMITTED_FOR_MATCHED_FOUNDATION_METHOD_EXTENSION"
    output["foundation_method_admission"] = {
        "admission_id": admission_id,
        "foundation_model_sha256": foundation_hash,
        "replay_train_sha256": replay_hash,
        "replay_validation_sha256": replay_validation_hash,
        "method_order": list(FOUNDATION_METHODS),
        "boundary": "The config is valid only with the immutable source artifacts recorded in the matching admission receipt.",
    }
    return output


def pretraining_forgetting_metrics(result_dir: Path) -> Dict[str, Any]:
    """Return the initial→final pretraining-head validation change from MACE JSONL."""

    records = []
    for path in sorted((result_dir / "results").glob("*_train.txt")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "eval" and row.get("head") == "pt_head":
                records.append({"path": str(path), "line": line_number, "metrics": row})
    if len(records) < 2:
        raise ValueError("Multihead run did not retain both initial and final pt_head validation metrics")
    initial = records[0]["metrics"]
    final = records[-1]["metrics"]
    fields = ("mae_e_per_atom", "rmse_e_per_atom", "mae_f", "rmse_f")
    delta = {}
    for field in fields:
        before = initial.get(field)
        after = final.get(field)
        if isinstance(before, bool) or not isinstance(before, (int, float)):
            raise ValueError(f"Initial pt_head metric is missing: {field}")
        if isinstance(after, bool) or not isinstance(after, (int, float)):
            raise ValueError(f"Final pt_head metric is missing: {field}")
        delta[f"{field}_final_minus_initial"] = float(after) - float(before)
    return {
        "schema": "micro-mlip-pretraining-forgetting-v1",
        "pretraining_head": "pt_head",
        "initial": initial,
        "final": final,
        "final_minus_initial": delta,
        "source_records": records,
        "interpretation": "Measured on the immutable pretraining-domain replay validation file; positive error change indicates forgetting relative to the initial foundation-head metric.",
    }


def verified_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, required=True)
    parser.add_argument("--replay-valid-file", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--molecule", default="ethanol")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--dry-run-epochs", type=int, default=1)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError("Foundation-method admission roots are immutable")
    for path, label in (
        (args.base_config, "base configuration"),
        (args.foundation_model, "foundation model"),
        (args.replay_file, "replay training file"),
        (args.replay_valid_file, "replay validation file"),
        (args.mace_runner, "MACE runner"),
    ):
        verified_file(path, label)
    if not args.mace_site.is_dir():
        raise FileNotFoundError(f"Missing MACE site directory: {args.mace_site}")
    if sha256(args.replay_file) == sha256(args.replay_valid_file):
        raise ValueError("Replay training and validation artifacts must be distinct")
    base_config = load_json(args.base_config)
    if args.molecule not in base_config.get("molecule_roles", {}):
        raise ValueError("Admission molecule is absent from the frozen five-round configuration")
    if base_config["molecule_roles"][args.molecule] != "development":
        raise ValueError("Foundation admission may use only a development molecule")
    if args.seed not in set(base_config.get("scientific_environment", {}).get("training", {}).get("seeds", [])):
        raise ValueError("Admission seed is absent from the frozen five-round configuration")

    foundation_hash = sha256(args.foundation_model)
    replay_hash = sha256(args.replay_file)
    replay_validation_hash = sha256(args.replay_valid_file)
    args.output_root.mkdir(parents=True)
    manifest_path = args.output_root / "method_admission.json"
    staging_config_path = args.output_root / "staging_method_config.json"
    staging = staged_config(
        base_config,
        foundation_hash=foundation_hash,
        replay_hash=replay_hash,
        replay_validation_hash=replay_validation_hash,
        dry_run_epochs=args.dry_run_epochs,
    )
    write_json(staging_config_path, staging)
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-foundation-method-admission-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "admission_id": args.admission_id,
        "base_config": str(args.base_config),
        "base_config_sha256": sha256(args.base_config),
        "staging_config": str(staging_config_path),
        "staging_config_sha256": sha256(staging_config_path),
        "sources": {
            "foundation_model": {"path": str(args.foundation_model), "sha256": foundation_hash},
            "replay_train": {"path": str(args.replay_file), "sha256": replay_hash},
            "replay_validation": {"path": str(args.replay_valid_file), "sha256": replay_validation_hash},
            "mace_runner": {"path": str(args.mace_runner), "sha256": sha256(args.mace_runner)},
        },
        "dry_run": {
            "molecule": args.molecule,
            "seed": args.seed,
            "prefix": "initial",
            "round_index": 1,
            "epochs": args.dry_run_epochs,
            "test_access": "FORBIDDEN",
        },
        "methods": {},
        "boundary": "A method is admitted only after an actual validation-only CPU run under the staged config. No failed method is retried or replaced.",
    }
    write_json(manifest_path, manifest)

    try:
        manifest["status"] = "RUNNING_VALIDATION_ONLY_DRY_RUNS"
        write_json(manifest_path, manifest)
        for method in FOUNDATION_METHODS:
            method_root = args.output_root / "dry_runs" / method
            gate_path = method_root / "resource_gate.json"
            gate = invoke(
                [
                    args.python,
                    str(args.code_root / "scripts" / "check_micro_mlip_resource_gate.py"),
                    "--formal-launcher-pid", str(args.formal_launcher_pid),
                    "--formal-state", str(args.formal_state),
                    "--formal-done", str(args.formal_done),
                    "--disk-path", str(args.disk_path),
                    "--minimum-free-gib", str(args.minimum_free_gib),
                    "--output", str(gate_path),
                ],
                method_root / "resource_gate.log",
            )
            require_success(f"shared-resource gate for {method}", gate)
            run_root = method_root / "runs"
            command = [
                args.python,
                str(args.code_root / "scripts" / "run_micro_mlip_five_round.py"),
                "--config", str(staging_config_path),
                "--dataset-dir", str(args.dataset_dir),
                "--molecule", args.molecule,
                "--prefix", "initial",
                "--round-index", "1",
                "--method", method,
                "--arm", "fixed",
                "--seed", str(args.seed),
                "--output-root", str(run_root),
                "--python", args.python,
                "--runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--foundation-model", str(args.foundation_model),
            ]
            if method == MULTIHEAD_METHOD:
                command.extend(
                    [
                        "--replay-file", str(args.replay_file),
                        "--replay-valid-file", str(args.replay_valid_file),
                    ]
                )
            training = invoke(command, method_root / "training_controller.log")
            require_success(f"validation-only foundation dry run for {method}", training)
            run_dirs = sorted(run_root.glob("*/run.json"))
            if len(run_dirs) != 1:
                raise ValueError(f"Expected exactly one dry-run receipt for {method}")
            run_dir = run_dirs[0].parent
            result = load_json(run_dir / "result.json")
            if result.get("status") != "TRAINING_COMPLETED_PENDING_VALIDATION_EVALUATION":
                raise ValueError(f"Dry run did not reach validation for {method}")
            validation = invoke(
                [
                    args.python,
                    str(args.code_root / "scripts" / "evaluate_micro_mlip_validation_round.py"),
                    "--run-dir", str(run_dir),
                    "--dataset-dir", str(args.dataset_dir),
                    "--molecule", args.molecule,
                    "--device", args.device,
                    "--default-dtype", "float32",
                ],
                method_root / "validation_controller.log",
            )
            require_success(f"rMD17 validation-only evaluator for {method}", validation)
            validation_receipt = run_dir / "validation_evaluation.json"
            validation_doc = load_json(validation_receipt)
            if validation_doc.get("status") != "COMPLETED" or validation_doc.get("test_access") != "NOT_OPENED_OR_READ":
                raise ValueError(f"Validation receipt violated the test boundary for {method}")
            method_receipt: Dict[str, Any] = {
                "status": "ADMITTED",
                "foundation_model_sha256": foundation_hash,
                "dry_run": {
                    "run_dir": str(run_dir),
                    "run_json_sha256": sha256(run_dir / "run.json"),
                    "result_sha256": sha256(run_dir / "result.json"),
                    "validation_receipt_sha256": sha256(validation_receipt),
                    "training": training,
                    "validation": validation,
                    "test_access": "NOT_OPENED_OR_READ",
                },
            }
            if method == MULTIHEAD_METHOD:
                method_receipt["replay_train_sha256"] = replay_hash
                method_receipt["replay_validation_sha256"] = replay_validation_hash
                method_receipt["forgetting_metric"] = pretraining_forgetting_metrics(run_dir)
            manifest["methods"][method] = method_receipt
            write_json(manifest_path, manifest)

        final_config_path = args.output_root / "admitted_method_config.json"
        final_config = admitted_config(
            base_config,
            foundation_hash=foundation_hash,
            replay_hash=replay_hash,
            replay_validation_hash=replay_validation_hash,
            admission_id=args.admission_id,
        )
        write_json(final_config_path, final_config)
        manifest["admitted_config"] = str(final_config_path)
        manifest["admitted_config_sha256"] = sha256(final_config_path)
        manifest["status"] = "ADMITTED"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "output_root": str(args.output_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Execute one bounded CPU micro-MLIP RSI admission phase without retries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.pilot import development_specs, holdout_specs, read_json, rows_from_run_directories


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256_or_missing(path: Path) -> str:
    return sha256(path) if path.is_file() else "MISSING"


def assert_inputs(
    admission: Mapping[str, Any], benchmark_config: Path, dataset_dir: Path
) -> Dict[str, str]:
    parent = admission["parent_benchmark"]
    if sha256(benchmark_config) != parent["config_sha256"]:
        raise ValueError("Frozen parent benchmark configuration hash mismatch")
    manifest = dataset_dir / "dataset_manifest.json"
    if sha256(manifest) != parent["dataset_manifest_sha256"]:
        raise ValueError("Frozen dataset manifest hash mismatch")
    validation = read_json(dataset_dir / "dataset_validation.json")
    if validation.get("valid") is not True:
        raise ValueError("Dataset validation is absent or failed")
    return {
        "benchmark_config": sha256(benchmark_config),
        "dataset_manifest": sha256(manifest),
        "dataset_validation": sha256(dataset_dir / "dataset_validation.json"),
    }


def invoke(command: List[str], log_path: Path) -> Dict[str, Any]:
    """Run one owned subprocess and persist all controller-visible output."""

    started = time.monotonic()
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "controller_log_sha256": sha256(log_path),
    }


def run_spec(args: argparse.Namespace, spec: Mapping[str, Any], phase_root: Path) -> Dict[str, Any]:
    run_id = "__".join(
        (
            spec["phase"],
            spec["arm"],
            spec["molecule"],
            spec["method"],
            spec["selection"],
            str(spec["label_budget"]),
            f"seed{spec['seed']}",
        )
    )
    run_script = args.code_root / "scripts" / "run_micro_mlip_round.py"
    evaluator_script = args.code_root / "scripts" / "evaluate_micro_mlip_round.py"
    run_root = phase_root / "runs"
    train_command = [
        args.python,
        str(run_script),
        "--config",
        str(args.benchmark_config),
        "--dataset-dir",
        str(args.dataset_dir),
        "--molecule",
        spec["molecule"],
        "--selection",
        spec["selection"],
        "--label-budget",
        str(spec["label_budget"]),
        "--method",
        spec["method"],
        "--arm",
        spec["arm"],
        "--seed",
        str(spec["seed"]),
        "--output-root",
        str(run_root),
        "--run-id",
        run_id,
        "--python",
        args.python,
        "--runner",
        str(args.mace_runner),
        "--mace-site",
        str(args.mace_site),
        "--device",
        "cpu",
    ]
    task_root = phase_root / "controller" / run_id
    task_root.mkdir(parents=True, exist_ok=False)
    train = invoke(train_command, task_root / "train_controller.log")
    run_dir = run_root / run_id
    if train["exit_code"] != 0:
        return {"run_id": run_id, "spec": dict(spec), "training": train, "status": "FAILED"}
    evaluator_command = [
        args.python,
        str(evaluator_script),
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(args.dataset_dir),
        "--molecule",
        spec["molecule"],
        "--device",
        "cpu",
        "--default-dtype",
        "float32",
    ]
    evaluation = invoke(evaluator_command, task_root / "evaluation_controller.log")
    return {
        "run_id": run_id,
        "spec": dict(spec),
        "training": train,
        "evaluation": evaluation,
        "run_directory": str(run_dir),
        "status": "COMPLETED" if evaluation["exit_code"] == 0 else "FAILED",
    }


def initial_manifest(args: argparse.Namespace, admission: Mapping[str, Any], source_hashes: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "schema": "micro-mlip-rsi-cpu-admission-execution-v1",
        "campaign_id": admission["campaign_id"],
        "status": "PREPARED",
        "created_at": utc_now(),
        "admission_config": str(args.admission_config),
        "benchmark_config": str(args.benchmark_config),
        "dataset_dir": str(args.dataset_dir),
        "code_root": str(args.code_root),
        "runtime": {
            "python": args.python,
            "mace_runner": str(args.mace_runner),
            "mace_site": str(args.mace_site),
            "device": "cpu",
        },
        "source_sha256": {
            **source_hashes,
            "admission_config": sha256(args.admission_config),
            "run_script": file_sha256_or_missing(args.code_root / "scripts" / "run_micro_mlip_round.py"),
            "evaluator_script": file_sha256_or_missing(args.code_root / "scripts" / "evaluate_micro_mlip_round.py"),
            "mace_runner": file_sha256_or_missing(args.mace_runner),
        },
        "development_runs": [],
        "holdout_runs": [],
        "cpu_seconds_used": 0.0,
    }


def remaining_budget_ok(manifest: Mapping[str, Any], admission: Mapping[str, Any]) -> bool:
    return float(manifest["cpu_seconds_used"]) < float(admission["budget"]["maximum_cpu_wall_seconds"])


def execute_phase(
    args: argparse.Namespace,
    admission: Mapping[str, Any],
    manifest: Dict[str, Any],
    phase: str,
    specs: List[Dict[str, Any]],
) -> int:
    phase_root = args.campaign_root / phase
    if phase_root.exists():
        raise FileExistsError(f"Refusing to reuse completed or partial phase directory: {phase_root}")
    phase_root.mkdir(parents=True)
    manifest["status"] = f"RUNNING_{phase.upper()}"
    manifest[f"{phase}_started_at"] = utc_now()
    state_path = args.campaign_root / "campaign_manifest.json"
    write_json(state_path, manifest)
    rows: List[Dict[str, Any]] = []
    target_key = "development_runs" if phase == "development" else "holdout_runs"
    try:
        for spec in specs:
            if not remaining_budget_ok(manifest, admission):
                raise RuntimeError("CPU wall-time budget exhausted before next round")
            entry = run_spec(args, spec, phase_root)
            manifest[target_key].append(entry)
            manifest["cpu_seconds_used"] = round(
                float(manifest["cpu_seconds_used"])
                + float(entry["training"]["duration_seconds"])
                + float(entry.get("evaluation", {}).get("duration_seconds", 0.0)),
                3,
            )
            write_json(state_path, manifest)
            if entry["status"] != "COMPLETED":
                raise RuntimeError(f"Round failed without retry: {entry['run_id']}")
        run_directories = [Path(entry["run_directory"]) for entry in manifest[target_key]]
        rows = rows_from_run_directories(run_directories)
        summary = {
            "schema": "micro-mlip-rsi-cpu-admission-phase-summary-v1",
            "campaign_id": admission["campaign_id"],
            "phase": phase,
            "completed_at": utc_now(),
            "rows": rows,
            "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest(),
            "claim_boundary": admission["claim_boundary"],
        }
        write_json(phase_root / "summary.json", summary)
        manifest["status"] = (
            "DEVELOPMENT_COMPLETED_PENDING_FROZEN_POLICIES"
            if phase == "development"
            else "HOLDOUT_COMPLETED_NOT_FOR_PROMOTION"
        )
        manifest[f"{phase}_completed_at"] = utc_now()
        write_json(state_path, manifest)
        if phase == "holdout":
            paired = {
                "schema": "micro-mlip-rsi-cpu-admission-paired-summary-v1",
                "campaign_id": admission["campaign_id"],
                "rows": rows,
                "sorted_by_force_then_energy": sorted(
                    rows,
                    key=lambda row: (
                        row["force_component_mae_meV_per_A"],
                        row["energy_mae_meV_per_atom"],
                    ),
                ),
                "interpretation": "One-seed CPU directional summary only; no promotion or significance claim.",
            }
            write_json(args.campaign_root / "paired_summary.json", paired)
        print(json.dumps({"status": manifest["status"], "rows": len(rows), "cpu_seconds_used": manifest["cpu_seconds_used"]}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = f"FAILED_STOPPED_NO_AUTO_RETRY_{phase.upper()}"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest[f"{phase}_failed_at"] = utc_now()
        write_json(state_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "holdout"), required=True)
    parser.add_argument("--admission-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--policies", type=Path)
    args = parser.parse_args()

    admission = read_json(args.admission_config)
    source_hashes = assert_inputs(admission, args.benchmark_config, args.dataset_dir)
    state_path = args.campaign_root / "campaign_manifest.json"
    if args.mode == "development":
        if args.campaign_root.exists():
            raise FileExistsError(f"Campaign root already exists: {args.campaign_root}")
        args.campaign_root.mkdir(parents=True)
        manifest = initial_manifest(args, admission, source_hashes)
        return execute_phase(args, admission, manifest, "development", development_specs(admission))

    if not state_path.is_file():
        raise FileNotFoundError("Holdout requires a completed development campaign manifest")
    if args.policies is None or not args.policies.is_file():
        raise ValueError("Holdout requires an immutable --policies file")
    manifest = read_json(state_path)
    if manifest.get("status") != "DEVELOPMENT_COMPLETED_PENDING_FROZEN_POLICIES":
        raise ValueError(f"Campaign is not eligible for holdout: {manifest.get('status')}")
    policies = read_json(args.policies)
    manifest["policy_file"] = str(args.policies)
    manifest["policy_file_sha256"] = sha256(args.policies)
    return execute_phase(args, admission, manifest, "holdout", holdout_specs(admission, policies))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independently measure one closed five-round model on validation data only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.evaluation import evaluate_extxyz, resolve_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified_validation_file(dataset_dir: Path, molecule: str) -> Path:
    manifest = load_json(dataset_dir / "dataset_manifest.json")
    try:
        entry = manifest["molecules"][molecule]["files"]["valid"]
        path = dataset_dir / entry["path"]
        expected_hash = entry["sha256"]
    except KeyError as exc:
        raise ValueError(f"Dataset manifest has no validation file for {molecule}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation file: {path}")
    if sha256(path) != expected_hash:
        raise ValueError(f"Validation-file hash mismatch: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", default="float32")
    args = parser.parse_args()

    run = load_json(args.run_dir / "run.json")
    result_path = args.run_dir / "result.json"
    result = load_json(result_path)
    if result.get("status") not in {
        "TRAINING_COMPLETED_PENDING_VALIDATION_EVALUATION",
        "TRAINING_VALIDATED_PENDING_INDEPENDENT_TEST_EVALUATION",
    }:
        raise RuntimeError("Validation evaluation requires a successfully closed five-round training receipt")
    if run.get("molecule") != args.molecule:
        raise ValueError("Requested molecule does not match immutable run receipt")
    output = args.run_dir / "validation_evaluation.json"
    if output.exists():
        print(output.read_text(encoding="utf-8"))
        return 0
    model = resolve_model(args.run_dir, args.model)
    validation_file = verified_validation_file(args.dataset_dir, args.molecule)
    started = time.monotonic()
    metrics = evaluate_extxyz(model, validation_file, args.device, args.default_dtype)
    receipt = {
        "schema": "micro-mlip-five-round-validation-evaluation-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "run_id": run["run_id"],
        "molecule": args.molecule,
        "round_index": run["round_index"],
        "model": str(model),
        "model_sha256": sha256(model),
        "validation_file": str(validation_file),
        "validation_file_sha256": sha256(validation_file),
        "device": args.device,
        "default_dtype": args.default_dtype,
        "python_version": sys.version,
        "platform": platform.platform(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "validation_metrics": metrics,
        "test_access": "NOT_OPENED_OR_READ",
    }
    write_json(output, receipt)
    result["status"] = "TRAINING_VALIDATED_PENDING_INDEPENDENT_TEST_EVALUATION"
    result["validation_evaluation"] = "COMPLETED"
    result["validation_evaluation_sha256"] = sha256(output)
    write_json(result_path, result)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

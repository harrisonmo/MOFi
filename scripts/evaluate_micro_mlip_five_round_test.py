#!/usr/bin/env python3
"""Release a five-round independent test only after a signed four-arm barrier."""

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


def verified_test_file(dataset_dir: Path, molecule: str) -> Path:
    manifest = load_json(dataset_dir / "dataset_manifest.json")
    try:
        entry = manifest["molecules"][molecule]["files"]["test"]
        path = dataset_dir / entry["path"]
        expected_hash = entry["sha256"]
    except KeyError as exc:
        raise ValueError(f"Dataset manifest has no test file for {molecule}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen test file: {path}")
    if sha256(path) != expected_hash:
        raise ValueError(f"Test-file hash mismatch: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", default="float32")
    args = parser.parse_args()

    run = load_json(args.run_dir / "run.json")
    result_path = args.run_dir / "result.json"
    result = load_json(result_path)
    if result.get("status") not in {
        "TRAINING_VALIDATED_PENDING_INDEPENDENT_TEST_EVALUATION",
        "COMPLETED_INDEPENDENT_TEST_EVALUATION",
    }:
        raise RuntimeError("Test release requires a completed training and validation receipt")
    if run.get("molecule") != args.molecule:
        raise ValueError("Requested molecule does not match immutable run receipt")
    barrier = load_json(args.release_manifest)
    if barrier.get("status") != "ALL_FOUR_ARMS_FINAL_TRAINING_VALIDATION_CLOSED":
        raise RuntimeError("The four-arm final-training barrier is not closed")
    eligible = barrier.get("eligible_run_ids")
    if not isinstance(eligible, list) or run["run_id"] not in eligible:
        raise RuntimeError("This run is not named by the closed four-arm release barrier")
    output = args.run_dir / "independent_evaluation.json"
    if output.exists():
        print(output.read_text(encoding="utf-8"))
        return 0
    model = resolve_model(args.run_dir, args.model)
    test_file = verified_test_file(args.dataset_dir, args.molecule)
    started = time.monotonic()
    metrics = evaluate_extxyz(model, test_file, args.device, args.default_dtype)
    receipt = {
        "schema": "micro-mlip-five-round-independent-evaluation-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "run_id": run["run_id"],
        "molecule": args.molecule,
        "round_index": run["round_index"],
        "model": str(model),
        "model_sha256": sha256(model),
        "test_file": str(test_file),
        "test_file_sha256": sha256(test_file),
        "release_manifest": str(args.release_manifest),
        "release_manifest_sha256": sha256(args.release_manifest),
        "device": args.device,
        "default_dtype": args.default_dtype,
        "python_version": sys.version,
        "platform": platform.platform(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "test_metrics": metrics,
    }
    write_json(output, receipt)
    result["status"] = "COMPLETED_INDEPENDENT_TEST_EVALUATION"
    result["independent_test_evaluation"] = "COMPLETED"
    result["independent_test_evaluation_sha256"] = sha256(output)
    write_json(result_path, result)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

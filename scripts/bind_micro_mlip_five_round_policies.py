#!/usr/bin/env python3
"""Bind frozen five-round policies to one audited configuration and data tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from scripts.run_micro_mlip_five_round_campaign import (
    ARMS,
    EXECUTOR_SOURCE_PATHS,
    validated_policies,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def executor_hashes(code_root: Path) -> Dict[str, str]:
    """Hash the controller/evaluator sources that interpret the frozen policy."""

    result: Dict[str, str] = {}
    for relative in EXECUTOR_SOURCE_PATHS:
        source = code_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing required five-round executor source: {source}")
        result[relative] = sha256(source)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite policy/data binding: {args.output}")
    config = load_json(args.config)
    manifest_path = args.dataset_dir / "dataset_manifest.json"
    validation_path = args.dataset_dir / "dataset_validation.json"
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    if manifest.get("benchmark_id") != config.get("benchmark_id"):
        raise ValueError("Dataset benchmark ID does not match the five-round configuration")
    if validation.get("valid") is not True:
        raise ValueError("Dataset audit must pass before policies can bind to it")
    policies = validated_policies(args.policies)
    receipt = {
        "schema": "micro-mlip-five-round-policy-environment-binding-v1",
        "status": "FROZEN_READY_FOR_CAMPAIGN",
        "created_at": utc_now(),
        "benchmark_id": config["benchmark_id"],
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_validation_sha256": sha256(validation_path),
        "policies": str(args.policies),
        "policies_sha256": sha256(args.policies),
        "policy_sha256_by_arm": {arm: policies[arm]["policy_sha256"] for arm in ARMS},
        "executor_source_sha256": executor_hashes(args.code_root),
        "test_release": "Independent test remains forbidden until the campaign controller closes its per-seed four-arm barrier.",
    }
    write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

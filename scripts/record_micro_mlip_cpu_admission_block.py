#!/usr/bin/env python3
"""Close a CPU admission campaign when its frozen outer policy call is blocked."""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    manifest_path = args.campaign_root / "campaign_manifest.json"
    failure_path = args.campaign_root / "policy_generation_failure.json"
    if failure_path.exists():
        print(failure_path.read_text(encoding="utf-8"))
        return 0
    manifest = load_json(manifest_path)
    if manifest.get("status") != "DEVELOPMENT_COMPLETED_PENDING_FROZEN_POLICIES":
        raise ValueError(f"Campaign is not eligible to be blocked: {manifest.get('status')}")
    outer_result = args.policy_root / "outer_generated" / "result.json"
    if not outer_result.is_file():
        raise FileNotFoundError("Missing failed outer-policy receipt")
    outer = load_json(outer_result)
    if outer.get("status") == "COMPLETED":
        raise ValueError("A completed outer-policy receipt cannot be recorded as a block")
    record = {
        "schema": "micro-mlip-rsi-cpu-admission-policy-block-v1",
        "status": "BLOCKED_OUTER_MODEL_CLI_UPGRADE_REQUIRED",
        "recorded_at": utc_now(),
        "reason": args.reason,
        "policy_root": str(args.policy_root),
        "outer_policy_receipt": str(outer_result),
        "outer_policy_receipt_sha256": sha256(outer_result),
        "outer_policy_status": outer.get("status"),
        "outer_policy_exit_code": outer.get("exit_code"),
        "outer_policy_usage": outer.get("usage"),
        "holdout_status": "NOT_STARTED",
        "retry_policy": "No retry, fallback-model substitution, or holdout execution under this frozen campaign.",
    }
    manifest["status"] = record["status"]
    manifest["blocked_at"] = record["recorded_at"]
    manifest["policy_generation_failure"] = record
    write_json(failure_path, record)
    write_json(manifest_path, manifest)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

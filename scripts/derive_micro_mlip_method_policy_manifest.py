#!/usr/bin/env python3
"""Derive a method-matched copy of one frozen five-round policy manifest.

The derivation changes only the policy's registered training-method token. It
does not call an outer model, alter any action table, or use scientific results.
An admitted foundation-method receipt is required so blocked methods cannot be
smuggled into an executable campaign.
"""

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
from rsi.micro_mlip.five_round import POLICY_METHODS, policy_hash
from scripts.run_micro_mlip_five_round_campaign import ARMS, validated_policies


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


def admitted_method(admission: Mapping[str, Any], target_method: str) -> Mapping[str, Any]:
    if admission.get("schema") != "micro-mlip-foundation-method-admission-v1":
        raise ValueError("Unexpected foundation-method admission schema")
    if admission.get("status") != "ADMITTED":
        raise ValueError("Foundation-method admission is not ADMITTED")
    methods = admission.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Foundation-method admission has no method receipts")
    receipt = methods.get(target_method)
    if not isinstance(receipt, Mapping) or receipt.get("status") != "ADMITTED":
        raise ValueError(f"Target method is not admitted: {target_method}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policies", type=Path, required=True)
    parser.add_argument("--target-method", choices=POLICY_METHODS, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite method-derived policies: {args.output}")
    source = load_json(args.source_policies)
    if source.get("schema") != "micro-mlip-five-round-policies-v1":
        raise ValueError("Unexpected source policy schema")
    source_validated = validated_policies(args.source_policies)
    admission = load_json(args.admission)
    method_receipt = admitted_method(admission, args.target_method)
    output = json.loads(json.dumps(source))
    output["status"] = "DERIVED_FROM_FROZEN_POLICY_AFTER_METHOD_ADMISSION"
    output["method_derivation"] = {
        "source_policies": str(args.source_policies),
        "source_policies_sha256": sha256(args.source_policies),
        "source_policy_method_by_arm": {arm: source_validated[arm]["method"] for arm in ARMS},
        "target_method": args.target_method,
        "admission": str(args.admission),
        "admission_sha256": sha256(args.admission),
        "method_receipt": method_receipt,
        "invariant": "Only the registered method token changes; acquisition actions, diagnostic route, split, label budget, and evaluator remain frozen.",
    }
    output["created_at"] = utc_now()
    arms = output.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("Source policy manifest has no complete four-arm block")
    for arm in ARMS:
        entry = arms[arm]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise ValueError(f"Source policy is malformed for {arm}")
        policy = dict(entry["policy"])
        policy["method"] = args.target_method
        entry["policy"] = policy
        entry["policy_sha256"] = policy_hash(policy)
        entry["origin"] = f"method_derived_from:{source_validated[arm]['origin']}"
    write_json(args.output, output)
    print(json.dumps({"status": output["status"], "method": args.target_method, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

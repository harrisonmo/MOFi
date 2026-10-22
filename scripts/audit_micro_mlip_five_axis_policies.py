#!/usr/bin/env python3
"""Audit every runnable five-axis cell policy before any MACE episode starts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_axis import allowed_samplers, assert_policy_matches_cell
from rsi.micro_mlip.five_round import ADAPTIVE_ROUNDS, policy_core, policy_hash


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


def policy_actions(policy: Mapping[str, Any]) -> Iterable[str]:
    for rule in policy_core(policy)["rounds"].values():
        if rule["mode"] == "static":
            yield rule["action"]
        elif rule["mode"] == "conditional":
            yield rule["default_action"]
            yield from rule["by_residual_class"].values()
        elif rule["mode"] == "raw_threshold":
            yield rule["if_true"]
            yield rule["if_false"]
            yield rule["if_missing"]
        else:
            raise AssertionError(f"Unknown registered policy mode: {rule['mode']}")


def check_cell(cell: Mapping[str, Any], policy_root: Path) -> Dict[str, Any]:
    cell_id = str(cell["cell_id"])
    directory = policy_root / cell_id
    manifest_path = directory / "policies.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "micro-mlip-five-round-policies-v1":
        raise ValueError(f"Unexpected policy schema for {cell_id}")
    if manifest.get("factorial_cell_id") != cell_id or manifest.get("factorial_levels") != cell["levels"]:
        raise ValueError(f"Policy manifest does not bind to exact cell levels: {cell_id}")
    evidence = load_json(directory / "development_evidence.json")
    if set(evidence.get("development_molecules", [])) != {"ethanol", "malonaldehyde"}:
        raise ValueError(f"Cell {cell_id} uses a non-development evidence set")
    if "toluene" in json.dumps(evidence, sort_keys=True).lower():
        raise ValueError(f"Cell {cell_id} evidence leaks the policy-holdout molecule")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"fixed", "outer_generated", "manual_control", "removed"}:
        raise ValueError(f"Cell {cell_id} does not contain the matched four arms")
    hashes: Dict[str, str] = {}
    for arm, entry in arms.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise ValueError(f"Cell {cell_id}/{arm} lacks a policy")
        observed = policy_hash(entry["policy"])
        if entry.get("policy_sha256") != observed:
            raise ValueError(f"Cell {cell_id}/{arm} has an invalid policy hash")
        hashes[arm] = observed
    assert_policy_matches_cell(arms["outer_generated"]["policy"], cell["levels"])
    assert_policy_matches_cell(arms["manual_control"]["policy"], cell["levels"])
    allowed = set(allowed_samplers(cell["levels"]))
    for arm in ("fixed", "removed"):
        if not set(policy_actions(arms[arm]["policy"])).issubset(allowed):
            raise ValueError(f"Cell {cell_id}/{arm} violates its Data action boundary")
    if policy_core(arms["fixed"]["policy"])["diagnostic_route"] != "none":
        raise ValueError(f"Cell {cell_id}/fixed is not a no-feedback baseline")
    if policy_core(arms["removed"]["policy"])["diagnostic_route"] != "removed":
        raise ValueError(f"Cell {cell_id}/removed is not an information-removed control")
    receipts = manifest.get("outer_call_receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError(f"Cell {cell_id} lacks outer-call receipts")
    token_total = 0
    duration_total = 0.0
    for arm in ("outer_generated", "removed"):
        receipt = receipts.get(arm)
        if not isinstance(receipt, Mapping) or receipt.get("status") != "COMPLETED" or receipt.get("exit_code") != 0:
            raise ValueError(f"Cell {cell_id}/{arm} outer call is not complete")
        if receipt.get("requested_model") != "gpt-6-astra":
            raise ValueError(f"Cell {cell_id}/{arm} used an unexpected requested model")
        usage = receipt.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError(f"Cell {cell_id}/{arm} has no usage receipt")
        token_total += sum(int(usage.get(key, 0)) for key in ("input_tokens", "output_tokens", "reasoning_output_tokens"))
        duration = receipt.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(f"Cell {cell_id}/{arm} has invalid duration")
        duration_total += float(duration)
    return {
        "cell_id": cell_id,
        "levels": cell["levels"],
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": sha256(manifest_path),
        "policy_sha256_by_arm": hashes,
        "outer_call_token_total": token_total,
        "outer_call_duration_seconds": round(duration_total, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite policy audit: {args.output}")
    config = load_json(args.screen_config)
    registry = load_json(args.registry)
    if registry.get("config_sha256") != sha256(args.screen_config):
        raise ValueError("Registry does not match supplied screen configuration")
    cells = [cell for cell in registry.get("cells", []) if cell.get("status") == "READY"]
    expected_ready = registry.get("ready_cell_count")
    if isinstance(expected_ready, bool) or not isinstance(expected_ready, int) or expected_ready <= 0:
        raise ValueError("Registry has no valid ready-cell count")
    if len(cells) != expected_ready:
        raise ValueError("Ready cell count does not match the registered experimental design")
    records = [check_cell(cell, args.policy_root) for cell in sorted(cells, key=lambda item: item["cell_id"])]
    output = {
        "schema": "micro-mlip-five-axis-policy-audit-v1",
        "status": "COMPLETED_BEFORE_MACE_TRAINING",
        "created_at": utc_now(),
        "screen_config": str(args.screen_config),
        "screen_config_sha256": sha256(args.screen_config),
        "registry": str(args.registry),
        "registry_sha256": sha256(args.registry),
        "registry_schema": registry.get("schema"),
        "policy_root": str(args.policy_root),
        "ready_cell_count": len(records),
        "blocked_cell_count": len([cell for cell in registry.get("cells", []) if cell.get("status") == "BLOCKED"]),
        "cells": records,
        "outer_call_token_total": sum(int(record["outer_call_token_total"]) for record in records),
        "outer_call_duration_seconds": round(sum(float(record["outer_call_duration_seconds"]) for record in records), 3),
        "boundaries": {
            "test_or_holdout_in_policy_generation": "FORBIDDEN_AND_AUDITED",
            "mace_training": "NOT_STARTED_BY_THIS_AUDIT",
            "model_high_methods": "NOT_EXECUTED_OR_REPRESENTED_AS_ZERO_EFFECTS_BY_THIS_POLICY_AUDIT",
        },
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "ready_cells": output["ready_cell_count"],
                "outer_call_tokens": output["outer_call_token_total"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Execute one serial, resource-gated scratch five-axis screening generation.

Each admitted factorial cell runs a complete matched four-arm, five-round
episode.  The runner creates a fresh shared-resource receipt immediately
before every cell, so it cannot begin a later cell on an old assertion that the
formal MOF campaign is absent.  It intentionally does not retry a failed cell
or manufacture a replacement policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_axis import assert_policy_matches_cell
from scripts.run_micro_mlip_five_round_campaign import validated_policies


DEFAULT_SCREEN_CONFIG = ROOT / "configs" / "micro-mlip-five-axis-screen-v1.json"
DEFAULT_FIVE_ROUND_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"
ARMS = ("fixed", "outer_generated", "manual_control", "removed")


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


def runnable_cells(
    screen_config_path: Path,
    registry_path: Path,
    policy_root: Path,
    policy_audit_path: Path,
) -> List[Dict[str, Any]]:
    """Return exactly the policy-audited scratch half-fraction in cell-ID order."""

    screen_config = load_json(screen_config_path)
    registry = load_json(registry_path)
    audit = load_json(policy_audit_path)
    if registry.get("schema") not in {
        "micro-mlip-five-axis-resolution-v-registry-v2",
        "micro-mlip-pdtw-full-scratch-registry-v1",
    }:
        raise ValueError("Unexpected five-axis registry schema")
    if registry.get("config_sha256") != sha256(screen_config_path):
        raise ValueError("Five-axis registry does not match its screen configuration")
    if audit.get("schema") != "micro-mlip-five-axis-policy-audit-v1":
        raise ValueError("Unexpected five-axis policy audit schema")
    if audit.get("status") != "COMPLETED_BEFORE_MACE_TRAINING":
        raise ValueError("Policy audit is not a completed pre-training receipt")
    if audit.get("screen_config_sha256") != sha256(screen_config_path):
        raise ValueError("Policy audit does not match the screen configuration")
    if audit.get("registry_sha256") != sha256(registry_path):
        raise ValueError("Policy audit does not match the five-axis registry")
    audited = {row.get("cell_id"): row for row in audit.get("cells", []) if isinstance(row, Mapping)}
    selected = []
    for cell in registry.get("cells", []):
        if not isinstance(cell, Mapping) or cell.get("status") != "READY":
            continue
        levels = cell.get("levels")
        if not isinstance(levels, Mapping) or levels.get("model") != "low":
            raise ValueError("Only scratch model-low cells may be passed to this runner")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in audited:
            raise ValueError("Every runnable cell must have an audited policy manifest")
        policies_path = policy_root / cell_id / "policies.json"
        if not policies_path.is_file():
            raise FileNotFoundError(f"Missing frozen policies for {cell_id}: {policies_path}")
        if audited[cell_id].get("policy_manifest_sha256") != sha256(policies_path):
            raise ValueError(f"Policy audit hash differs for {cell_id}")
        manifest = load_json(policies_path)
        if manifest.get("factorial_cell_id") != cell_id or manifest.get("factorial_levels") != levels:
            raise ValueError(f"Policy manifest does not bind to the registered cell: {cell_id}")
        policies = validated_policies(policies_path)
        # The adaptive candidate and matched manual policy must obey the cell
        # interface exactly.  ``fixed`` and ``removed`` are deliberately
        # static/no-information controls; their distinct routes are audited
        # by the frozen policy audit rather than being forced to masquerade as
        # a Tool-high static policy.
        for arm in ("outer_generated", "manual_control"):
            assert_policy_matches_cell(policies[arm]["policy"], levels)
        selected.append({"cell_id": cell_id, "levels": dict(levels), "policies": policies_path})
    expected_ready = registry.get("ready_cell_count")
    if isinstance(expected_ready, bool) or not isinstance(expected_ready, int) or expected_ready <= 0:
        raise ValueError("Registry has no valid ready-cell count")
    if len(selected) != expected_ready:
        raise ValueError("Runnable-cell count does not match the registered experimental design")
    return sorted(selected, key=lambda row: row["cell_id"])


def invoke(command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    """Run one owned compiler/controller command and preserve its complete log."""

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


def output_exists_guard(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a screen artifact: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--screen-config", type=Path, default=DEFAULT_SCREEN_CONFIG)
    parser.add_argument("--five-round-config", type=Path, default=DEFAULT_FIVE_ROUND_CONFIG)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--molecule", dest="molecules", action="append", required=True)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, required=True)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--resource-gate-max-age-seconds", type=int, default=900)
    args = parser.parse_args()

    if args.screen_root.exists():
        raise FileExistsError("Screen root already exists; failed or completed screens are immutable")
    config = load_json(args.five_round_config)
    screen_config = load_json(args.screen_config)
    if screen_config.get("parent_benchmark_id") != config.get("benchmark_id"):
        raise ValueError("Screen configuration is not attached to the supplied five-round benchmark")
    if screen_config.get("parent_five_round_config_sha256") != sha256(args.five_round_config):
        raise ValueError("Screen configuration does not match the supplied five-round configuration")
    if set(args.molecules) != {"ethanol", "malonaldehyde"}:
        raise ValueError("A screening generation must use exactly the two registered development molecules")
    if any(config.get("molecule_roles", {}).get(molecule) != "development" for molecule in args.molecules):
        raise ValueError("A screening generation may not use a policy-holdout molecule")
    registered_seeds = set(config.get("scientific_environment", {}).get("training", {}).get("seeds", []))
    if not args.seeds or not set(args.seeds).issubset(registered_seeds):
        raise ValueError("Screen seed is absent from the frozen five-round configuration")
    cells = runnable_cells(args.screen_config, args.registry, args.policy_root, args.policy_audit)
    if not args.mace_runner.is_file() or not args.mace_site.is_dir():
        raise FileNotFoundError("MACE runner or MACE site is unavailable")
    args.screen_root.mkdir(parents=True)
    manifest_path = args.screen_root / "screen_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-five-axis-screen-execution-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "screen_id": args.screen_id,
        "generation": args.generation,
        "source_sha256": {
            "screen_config": sha256(args.screen_config),
            "five_round_config": sha256(args.five_round_config),
            "registry": sha256(args.registry),
            "policy_audit": sha256(args.policy_audit),
            "policy_manifest_by_cell": {row["cell_id"]: sha256(row["policies"]) for row in cells},
        },
        "molecules": args.molecules,
        "seeds": args.seeds,
        "cells": [{"cell_id": row["cell_id"], "levels": row["levels"], "status": "PENDING"} for row in cells],
        "expected_training_runs": len(cells) * len(args.molecules) * len(args.seeds) * len(ARMS) * 5,
        "execution_boundary": "One fresh resource receipt is required before each serial cell; no automatic retry or policy substitution.",
    }
    write_json(manifest_path, manifest)

    try:
        manifest["status"] = "RUNNING"
        write_json(manifest_path, manifest)
        for cell in cells:
            cell_id = cell["cell_id"]
            cell_root = args.screen_root / "cells" / cell_id
            output_exists_guard(cell_root)
            cell_root.mkdir(parents=True)
            entry = next(row for row in manifest["cells"] if row["cell_id"] == cell_id)
            entry["status"] = "BINDING"
            entry["started_at"] = utc_now()
            write_json(manifest_path, manifest)

            binding_path = cell_root / "policy_environment_binding.json"
            binding = invoke(
                [
                    args.python,
                    str(args.code_root / "scripts" / "bind_micro_mlip_five_round_policies.py"),
                    "--config", str(args.five_round_config),
                    "--dataset-dir", str(args.dataset_dir),
                    "--policies", str(cell["policies"]),
                    "--code-root", str(args.code_root),
                    "--output", str(binding_path),
                ],
                cell_root / "binding.log",
            )
            entry["binding"] = binding
            if binding["exit_code"] != 0:
                raise RuntimeError(f"Binding failed for {cell_id}")

            gate_path = cell_root / "resource_gate.json"
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
                cell_root / "resource_gate.log",
            )
            entry["resource_gate"] = gate
            if gate["exit_code"] != 0:
                raise RuntimeError(f"Resource gate failed for {cell_id}")

            campaign_root = cell_root / "campaign"
            campaign_command = [
                args.python,
                str(args.code_root / "scripts" / "run_micro_mlip_five_round_campaign.py"),
                "--campaign-id", f"{args.screen_id}__{cell_id}",
                "--campaign-root", str(campaign_root),
                "--config", str(args.five_round_config),
                "--dataset-dir", str(args.dataset_dir),
                "--policies", str(cell["policies"]),
                "--policy-binding", str(binding_path),
                "--resource-gate", str(gate_path),
                "--resource-gate-max-age-seconds", str(args.resource_gate_max_age_seconds),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--resource-check-note", f"fresh per-cell receipt for {args.screen_id}/{cell_id}",
                "--factorial-cell-id", cell_id,
                "--factorial-levels-json", json.dumps(cell["levels"], sort_keys=True),
            ]
            for molecule in args.molecules:
                campaign_command.extend(["--molecule", molecule])
            for seed in args.seeds:
                campaign_command.extend(["--seed", str(seed)])
            entry["status"] = "RUNNING"
            write_json(manifest_path, manifest)
            campaign = invoke(campaign_command, cell_root / "campaign.log")
            entry["campaign"] = campaign
            if campaign["exit_code"] != 0:
                raise RuntimeError(f"Campaign failed for {cell_id}")

            evidence_path = cell_root / f"generation_{args.generation}_evidence.json"
            evidence_command = [
                args.python,
                str(args.code_root / "scripts" / "collect_micro_mlip_generation_evidence.py"),
                "--generation", args.generation,
                "--output", str(evidence_path),
            ]
            for molecule in args.molecules:
                for seed in args.seeds:
                    evidence_command.extend(
                        [
                            "--input",
                            str(campaign_root / "validation_only_evidence" / f"{molecule}__seed{seed}.json"),
                        ]
                    )
            evidence = invoke(evidence_command, cell_root / "generation_evidence.log")
            entry["generation_evidence"] = evidence
            if evidence["exit_code"] != 0:
                raise RuntimeError(f"Validation-only evidence collection failed for {cell_id}")
            entry["status"] = "COMPLETED"
            entry["completed_at"] = utc_now()
            write_json(manifest_path, manifest)

        manifest["status"] = "COMPLETED_PENDING_G1_POLICY_FREEZE"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "screen_root": str(args.screen_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

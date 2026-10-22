#!/usr/bin/env python3
"""Run the one frozen promoted policy on the toluene holdout.

The holdout launcher accepts a completed validation-only promotion receipt and
the exact policy manifest named by that receipt.  It does not call an outer
model or select among candidates.  As in every inner episode, independent
evaluation opens only after all four arms close their fifth validation round.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-id", required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--five-round-config", type=Path, required=True)
    parser.add_argument("--base-five-round-config", type=Path)
    parser.add_argument(
        "--method",
        choices=("scratch_micro", "foundation_naive", "lora_rank4", "layer_freeze5", "multihead_pseudolabel_replay"),
        default="scratch_micro",
    )
    parser.add_argument("--source-policy-manifest", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--foundation-model", type=Path)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--replay-valid-file", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--resource-gate-max-age-seconds", type=int, default=900)
    args = parser.parse_args()

    if args.holdout_root.exists():
        raise FileExistsError("Holdout root already exists; policy holdout is immutable")
    promotion = load_json(args.promotion)
    if promotion.get("schema") != "micro-mlip-pdtw-promotion-selection-v1":
        raise ValueError("Unexpected promotion selection schema")
    if promotion.get("status") != "FROZEN_BEFORE_POLICY_HOLDOUT":
        raise ValueError("Promotion receipt is not frozen for the policy holdout")
    selection_config = args.base_five_round_config or args.five_round_config
    if promotion.get("five_round_config_sha256") != sha256(selection_config):
        raise ValueError("Promotion receipt does not match the five-round configuration")
    selected = promotion.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Promotion receipt lacks a selected policy")
    policy = load_json(args.policy_manifest)
    if policy.get("generation") != selected.get("generation") or policy.get("factorial_cell_id") != selected.get("factorial_cell_id"):
        raise ValueError("Supplied holdout policy has a different generation/cell identity")
    policies = validated_policies(args.policy_manifest)
    if args.method == "scratch_micro":
        if selected.get("policy_manifest_sha256") != sha256(args.policy_manifest):
            raise ValueError("Supplied holdout policy is not the frozen promoted policy")
        if args.source_policy_manifest is not None or args.admission is not None:
            raise ValueError("Scratch holdout must not receive derived-policy or admission artifacts")
    else:
        if args.source_policy_manifest is None or args.admission is None:
            raise ValueError("Foundation method holdout requires source policy and admission receipts")
        if sha256(args.source_policy_manifest) != selected.get("policy_manifest_sha256"):
            raise ValueError("Derived policy source does not match the frozen promoted policy")
        derivation = policy.get("method_derivation")
        if not isinstance(derivation, Mapping):
            raise ValueError("Foundation method policy lacks a derivation receipt")
        if derivation.get("source_policies_sha256") != selected.get("policy_manifest_sha256"):
            raise ValueError("Method derivation does not bind to the promoted source policy")
        if derivation.get("target_method") != args.method:
            raise ValueError("Method derivation does not match the requested method")
        admission = load_json(args.admission)
        methods = admission.get("methods")
        if admission.get("schema") != "micro-mlip-foundation-method-admission-v1" or admission.get("status") != "ADMITTED":
            raise ValueError("Foundation-method admission is not completed")
        if not isinstance(methods, Mapping) or not isinstance(methods.get(args.method), Mapping):
            raise ValueError("Foundation-method admission lacks the requested method")
        if methods[args.method].get("status") != "ADMITTED":
            raise ValueError("Requested foundation method was not admitted")
        if derivation.get("admission_sha256") != sha256(args.admission):
            raise ValueError("Method derivation was created from a different admission receipt")
        if {policies[arm]["method"] for arm in ARMS} != {args.method}:
            raise ValueError("Derived policy arms do not all carry the requested method token")
    config = load_json(args.five_round_config)
    if config.get("molecule_roles", {}).get("toluene") != "policy_holdout":
        raise ValueError("Five-round configuration does not register toluene as the policy holdout")
    registered_seeds = set(config.get("scientific_environment", {}).get("training", {}).get("seeds", []))
    if args.seed not in registered_seeds:
        raise ValueError("Holdout seed is absent from the five-round configuration")

    method_definition = config.get("scientific_environment", {}).get("methods", {}).get(args.method)
    if not isinstance(method_definition, Mapping) or not str(method_definition.get("status", "")).startswith("READY"):
        raise ValueError("Execution configuration does not admit the requested method")
    if bool(method_definition.get("foundation", False)):
        if args.foundation_model is None or not args.foundation_model.is_file():
            raise ValueError("Foundation method holdout requires a readable foundation model")
    elif args.foundation_model is not None:
        raise ValueError("Scratch holdout must not receive a foundation model")
    if bool(method_definition.get("multiheads_finetuning", False)):
        if args.replay_file is None or not args.replay_file.is_file():
            raise ValueError("Multihead method holdout requires a readable replay file")
        if bool(method_definition.get("replay_validation_required", False)):
            if args.replay_valid_file is None or not args.replay_valid_file.is_file():
                raise ValueError("Multihead method holdout requires a readable replay validation file")
    elif args.replay_file is not None or args.replay_valid_file is not None:
        raise ValueError("Only an admitted multihead method may receive replay artifacts")

    args.holdout_root.mkdir(parents=True)
    manifest_path = args.holdout_root / "holdout_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-five-round-policy-holdout-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "holdout_id": args.holdout_id,
        "promotion": str(args.promotion),
        "promotion_sha256": sha256(args.promotion),
        "selected_generation": selected.get("generation"),
        "selected_factorial_cell_id": selected.get("factorial_cell_id"),
        "policy_manifest": str(args.policy_manifest),
        "policy_manifest_sha256": sha256(args.policy_manifest),
        "source_policy_manifest": str(args.source_policy_manifest) if args.source_policy_manifest is not None else None,
        "source_policy_manifest_sha256": sha256(args.source_policy_manifest) if args.source_policy_manifest is not None else None,
        "method": args.method,
        "execution_config": str(args.five_round_config),
        "execution_config_sha256": sha256(args.five_round_config),
        "selection_config": str(selection_config),
        "selection_config_sha256": sha256(selection_config),
        "admission": str(args.admission) if args.admission is not None else None,
        "admission_sha256": sha256(args.admission) if args.admission is not None else None,
        "foundation_model_sha256": sha256(args.foundation_model) if args.foundation_model is not None else None,
        "replay_file_sha256": sha256(args.replay_file) if args.replay_file is not None else None,
        "replay_validation_file_sha256": sha256(args.replay_valid_file) if args.replay_valid_file is not None else None,
        "molecule": "toluene",
        "seed": args.seed,
        "selection_boundary": "The policy was selected from development validation-only evidence before this holdout launcher was created.",
    }
    write_json(manifest_path, manifest)
    try:
        binding_path = args.holdout_root / "policy_environment_binding.json"
        binding = invoke(
            [
                args.python,
                str(args.code_root / "scripts" / "bind_micro_mlip_five_round_policies.py"),
                "--config", str(args.five_round_config),
                "--dataset-dir", str(args.dataset_dir),
                "--policies", str(args.policy_manifest),
                "--code-root", str(args.code_root),
                "--output", str(binding_path),
            ],
            args.holdout_root / "binding.log",
        )
        manifest["binding"] = binding
        if binding["exit_code"] != 0:
            raise RuntimeError("Holdout policy/data/code binding failed")
        gate_path = args.holdout_root / "resource_gate.json"
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
            args.holdout_root / "resource_gate.log",
        )
        manifest["resource_gate"] = gate
        if gate["exit_code"] != 0:
            raise RuntimeError("Holdout resource gate failed")
        campaign_command = [
            args.python,
            str(args.code_root / "scripts" / "run_micro_mlip_five_round_campaign.py"),
            "--campaign-id", args.holdout_id,
            "--campaign-root", str(args.holdout_root / "campaign"),
            "--config", str(args.five_round_config),
            "--dataset-dir", str(args.dataset_dir),
            "--policies", str(args.policy_manifest),
            "--policy-binding", str(binding_path),
            "--resource-gate", str(gate_path),
            "--resource-gate-max-age-seconds", str(args.resource_gate_max_age_seconds),
            "--code-root", str(args.code_root),
            "--python", args.python,
            "--mace-runner", str(args.mace_runner),
            "--mace-site", str(args.mace_site),
            "--device", args.device,
            "--method", args.method,
            "--molecule", "toluene",
            "--seed", str(args.seed),
            "--resource-check-note", "fresh receipt for the frozen toluene policy holdout",
            "--factorial-cell-id", str(selected["factorial_cell_id"]),
            "--factorial-levels-json", json.dumps(selected["factorial_levels"], sort_keys=True),
        ]
        if args.foundation_model is not None:
            campaign_command.extend(["--foundation-model", str(args.foundation_model)])
        if args.replay_file is not None:
            campaign_command.extend(["--replay-file", str(args.replay_file)])
        if args.replay_valid_file is not None:
            campaign_command.extend(["--replay-valid-file", str(args.replay_valid_file)])
        campaign = invoke(campaign_command, args.holdout_root / "campaign.log")
        manifest["campaign"] = campaign
        if campaign["exit_code"] != 0:
            raise RuntimeError("Toluene policy-holdout campaign failed")
        manifest["status"] = "COMPLETED_PENDING_ANALYSIS"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "holdout_root": str(args.holdout_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

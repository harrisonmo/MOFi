#!/usr/bin/env python3
"""Run matched five-round four-arm micro-MLIP episodes without auto-retries.

The controller is intentionally a fixed interpreter.  It executes only
pre-hashed policy tables, persists each validation observation before the next
choice, and opens test files only through a per-molecule/seed four-arm barrier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from rsi.micro_mlip.generation_evidence import validation_only_document
from rsi.micro_mlip.five_round import (
    ADAPTIVE_ROUNDS,
    ROUND_COUNT,
    child_prefix,
    choose_action,
    initial_prefix,
    policy_core,
    policy_hash,
    raw_validation_context,
    residual_signature,
)


ARMS = ("fixed", "outer_generated", "manual_control", "removed")
DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"
EXECUTOR_SOURCE_PATHS = (
    "scripts/__init__.py",
    "scripts/run_micro_mlip_five_round.py",
    "scripts/run_micro_mlip_round.py",
    "scripts/run_micro_mlip_five_round_campaign.py",
    "scripts/evaluate_micro_mlip_validation_round.py",
    "scripts/evaluate_micro_mlip_five_round_test.py",
    "rsi/micro_mlip/evaluation.py",
    "rsi/micro_mlip/five_round.py",
    "rsi/micro_mlip/five_round_data.py",
    "rsi/micro_mlip/generation_evidence.py",
    "rsi/micro_mlip/protocol.py",
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
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    """Run one owned command and retain controller-visible output."""

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
        "controller_log_sha256": sha256(log_path),
    }


def validated_policies(path: Path) -> Dict[str, Dict[str, Any]]:
    document = load_json(path)
    if document.get("schema") != "micro-mlip-five-round-policies-v1":
        raise ValueError("Unexpected five-round policy manifest schema")
    arms = document.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("Policy manifest must contain exactly the four registered arms")
    result: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        entry = arms[arm]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise ValueError(f"Policy manifest entry is invalid for {arm}")
        core = policy_core(entry["policy"])
        actual_hash = policy_hash(core)
        if entry.get("policy_sha256") != actual_hash:
            raise ValueError(f"Policy hash mismatch for {arm}")
        result[arm] = {
            "policy": core,
            "policy_sha256": actual_hash,
            "method": core["method"],
            "origin": entry.get("origin"),
        }
    return result


def assert_policy_method(policies: Mapping[str, Mapping[str, Any]], method: str) -> None:
    """Reject a campaign whose frozen tables belong to another model recipe."""

    observed = {arm: policies[arm].get("method") for arm in ARMS}
    if set(observed.values()) != {method}:
        raise ValueError(
            f"Frozen policy methods do not match requested campaign method {method}: {observed}"
        )


def assert_inputs(config: Mapping[str, Any], dataset_dir: Path) -> Dict[str, str]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    validation_path = dataset_dir / "dataset_validation.json"
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    if manifest.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError("Dataset benchmark ID does not match five-round configuration")
    if validation.get("valid") is not True:
        raise ValueError("Five-round dataset audit is absent or failed")
    return {
        "dataset_manifest": sha256(manifest_path),
        "dataset_validation": sha256(validation_path),
    }


def assert_policy_binding(
    binding_path: Path,
    config_path: Path,
    dataset_dir: Path,
    policies_path: Path,
    policies: Mapping[str, Mapping[str, Any]],
    code_root: Path,
) -> str:
    """Require a frozen policy/data/config linkage before any training starts."""

    binding = load_json(binding_path)
    if binding.get("schema") != "micro-mlip-five-round-policy-environment-binding-v1":
        raise ValueError("Unexpected policy-environment binding schema")
    if binding.get("status") != "FROZEN_READY_FOR_CAMPAIGN":
        raise ValueError("Policy-environment binding is not frozen for execution")
    config = load_json(config_path)
    if binding.get("benchmark_id") != config.get("benchmark_id"):
        raise ValueError("Policy-environment binding benchmark ID does not match configuration")
    expected = {
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(dataset_dir / "dataset_manifest.json"),
        "dataset_validation_sha256": sha256(dataset_dir / "dataset_validation.json"),
        "policies_sha256": sha256(policies_path),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"Policy-environment binding mismatch for {key}")
    bound_arms = binding.get("policy_sha256_by_arm")
    if not isinstance(bound_arms, Mapping) or {
        arm: bound_arms.get(arm) for arm in ARMS
    } != {arm: policies[arm]["policy_sha256"] for arm in ARMS}:
        raise ValueError("Policy-environment binding does not match the frozen arm policies")
    expected_sources = {
        relative: sha256(code_root / relative) for relative in EXECUTOR_SOURCE_PATHS
    }
    if binding.get("executor_source_sha256") != expected_sources:
        raise ValueError("Policy-environment binding does not match the execution sources")
    return sha256(binding_path)


def assert_resource_gate(gate_path: Path, maximum_age_seconds: int) -> str:
    """Require a fresh affirmative shared-resource receipt before training."""

    gate = load_json(gate_path)
    if gate.get("schema") != "micro-mlip-shared-resource-gate-v1":
        raise ValueError("Unexpected shared-resource gate schema")
    if gate.get("status") != "ADMITTED":
        raise ValueError(f"Shared-resource gate did not admit campaign: {gate.get('blockers')}")
    checked_at = gate.get("checked_at")
    if not isinstance(checked_at, str):
        raise ValueError("Shared-resource gate has no check timestamp")
    try:
        observed = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Shared-resource gate timestamp is invalid") from exc
    if observed.tzinfo is None:
        raise ValueError("Shared-resource gate timestamp lacks a timezone")
    age_seconds = (datetime.now(timezone.utc) - observed).total_seconds()
    if age_seconds < 0 or age_seconds > maximum_age_seconds:
        raise ValueError(
            f"Shared-resource gate is stale ({age_seconds:.1f}s; maximum {maximum_age_seconds}s)"
        )
    return sha256(gate_path)


def episode_key(molecule: str, seed: int, arm: str) -> str:
    return f"{molecule}__seed{seed}__{arm}"


def initial_episode_state(
    molecule: str,
    seed: int,
    arm: str,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": "micro-mlip-five-round-episode-state-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "molecule": molecule,
        "seed": seed,
        "arm": arm,
        "method": policy_core(policy)["method"],
        "policy": policy,
        "policy_sha256": policy_hash(policy),
        "current_prefix": initial_prefix(),
        "rounds": [],
        "decisions": [],
        "test_access": "NOT_OPENED",
    }


def run_id(molecule: str, seed: int, arm: str, method: str, round_index: int, prefix: str) -> str:
    return "__".join(
        (molecule, f"seed{seed}", arm, method, f"round{round_index}", prefix.replace("__", "-"))
    )


def run_stage(
    args: argparse.Namespace,
    state_root: Path,
    state: Dict[str, Any],
    round_index: int,
) -> Dict[str, Any]:
    """Train and independently validate one stage, resuming only closed receipts."""

    prefix = state["current_prefix"]
    identifier = run_id(
        state["molecule"], state["seed"], state["arm"], args.method, round_index, prefix
    )
    existing = next((row for row in state["rounds"] if row["round_index"] == round_index), None)
    if existing is not None:
        if existing.get("run_id") != identifier or existing.get("prefix") != prefix:
            raise ValueError("Checkpoint state conflicts with the registered selection-tree transition")
        validation_receipt = Path(existing["run_directory"]) / "validation_evaluation.json"
        if validation_receipt.is_file():
            return existing
        raise RuntimeError("A prior stage is incomplete; retry is forbidden by the campaign contract")

    run_root = args.campaign_root / "runs"
    controller_root = state_root / f"round_{round_index:02d}"
    train_command = [
        args.python,
        str(args.code_root / "scripts" / "run_micro_mlip_five_round.py"),
        "--config", str(args.config),
        "--dataset-dir", str(args.dataset_dir),
        "--molecule", state["molecule"],
        "--prefix", prefix,
        "--round-index", str(round_index),
        "--method", args.method,
        "--arm", state["arm"],
        "--seed", str(state["seed"]),
        "--output-root", str(run_root),
        "--run-id", identifier,
        "--python", args.python,
        "--runner", str(args.mace_runner),
        "--mace-site", str(args.mace_site),
        "--device", args.device,
    ]
    if args.foundation_model is not None:
        train_command.extend(["--foundation-model", str(args.foundation_model)])
    if args.replay_file is not None:
        train_command.extend(["--replay-file", str(args.replay_file)])
    if args.replay_valid_file is not None:
        train_command.extend(["--replay-valid-file", str(args.replay_valid_file)])
    train = invoke(train_command, controller_root / "train_controller.log")
    run_directory = run_root / identifier
    row: Dict[str, Any] = {
        "round_index": round_index,
        "prefix": prefix,
        "run_id": identifier,
        "run_directory": str(run_directory),
        "training": train,
    }
    if train["exit_code"] != 0:
        row["status"] = "FAILED_NO_AUTO_RETRY"
        return row
    validation_command = [
        args.python,
        str(args.code_root / "scripts" / "evaluate_micro_mlip_validation_round.py"),
        "--run-dir", str(run_directory),
        "--dataset-dir", str(args.dataset_dir),
        "--molecule", state["molecule"],
        "--device", args.device,
        "--default-dtype", "float32",
    ]
    validation = invoke(validation_command, controller_root / "validation_controller.log")
    row["validation"] = validation
    if validation["exit_code"] != 0:
        row["status"] = "FAILED_NO_AUTO_RETRY"
        return row
    receipt = load_json(run_directory / "validation_evaluation.json")
    if receipt.get("status") != "COMPLETED":
        row["status"] = "FAILED_NO_AUTO_RETRY"
        return row
    row["validation_metrics"] = receipt["validation_metrics"]
    row["validation_receipt_sha256"] = sha256(run_directory / "validation_evaluation.json")
    row["status"] = "COMPLETED"
    return row


def state_after_round(
    config: Mapping[str, Any], state: Dict[str, Any], round_row: Mapping[str, Any]
) -> None:
    """Persist a validation observation and the next frozen selection decision."""

    if round_row.get("status") != "COMPLETED":
        state["status"] = "FAILED_NO_AUTO_RETRY"
        state["failure"] = f"round_{round_row.get('round_index')}:{round_row.get('status')}"
        return
    state["rounds"].append(dict(round_row))
    round_index = int(round_row["round_index"])
    if round_index == ROUND_COUNT:
        state["status"] = "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER"
        return
    previous = state["rounds"][-2]["validation_metrics"] if len(state["rounds"]) > 1 else None
    route = policy_core(state["policy"])["diagnostic_route"]
    if route == "validation_residual_v1":
        signature = residual_signature(
            round_row["validation_metrics"],
            previous,
            config["validation_residual_diagnostic"],
        )
    elif route == "raw_validation_v1":
        signature = raw_validation_context(round_row["validation_metrics"], previous)
    else:
        signature = None
    decision = choose_action(state["policy"], round_index + 1, signature)
    next_prefix = child_prefix(state["current_prefix"], decision["selection"])
    state["decisions"].append(
        {
            "after_round": round_index,
            "before_round": round_index + 1,
            "source_prefix": state["current_prefix"],
            "next_prefix": next_prefix,
            "validation_context": signature,
            "decision": decision,
        }
    )
    state["current_prefix"] = next_prefix
    state["status"] = "RUNNING_NEXT_ROUND"


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    mutable = dict(state)
    mutable["updated_at"] = utc_now()
    write_json(path, mutable)


def load_or_prepare_episode(
    args: argparse.Namespace,
    policies: Mapping[str, Mapping[str, Any]],
    molecule: str,
    seed: int,
    arm: str,
) -> Dict[str, Any]:
    state_path = args.campaign_root / "episodes" / episode_key(molecule, seed, arm) / "state.json"
    if state_path.is_file():
        state = load_json(state_path)
        if state.get("policy_sha256") != policies[arm]["policy_sha256"]:
            raise ValueError("An existing episode has a different frozen policy")
        return state
    state = initial_episode_state(molecule, seed, arm, policies[arm]["policy"])
    write_state(state_path, state)
    return state


def release_group(
    args: argparse.Namespace,
    molecule: str,
    seed: int,
    states: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Create a barrier only once every arm has a closed fifth validation receipt."""

    if any(state.get("status") != "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER" for state in states.values()):
        raise RuntimeError("Cannot open independent evaluation before every arm closes final validation")
    eligible: List[str] = []
    final_receipts: Dict[str, str] = {}
    for arm in ARMS:
        final = states[arm]["rounds"][-1]
        if final.get("round_index") != ROUND_COUNT or final.get("status") != "COMPLETED":
            raise RuntimeError("Final validation stage is incomplete")
        eligible.append(final["run_id"])
        final_receipts[arm] = final["validation_receipt_sha256"]
    barrier_path = args.campaign_root / "release_barriers" / f"{molecule}__seed{seed}.json"
    if not barrier_path.exists():
        write_json(
            barrier_path,
            {
                "schema": "micro-mlip-five-round-release-barrier-v1",
                "status": "ALL_FOUR_ARMS_FINAL_TRAINING_VALIDATION_CLOSED",
                "created_at": utc_now(),
                "molecule": molecule,
                "seed": seed,
                "eligible_run_ids": eligible,
                "final_validation_receipts": final_receipts,
                "test_access_before_barrier": "FORBIDDEN",
            },
        )
    return barrier_path


def run_independent_evaluations(
    args: argparse.Namespace,
    molecule: str,
    seed: int,
    states: Dict[str, Dict[str, Any]],
) -> None:
    if all(state.get("status") == "COMPLETED" for state in states.values()):
        return
    barrier = release_group(args, molecule, seed, states)
    for arm in ARMS:
        state = states[arm]
        final = state["rounds"][-1]
        run_dir = Path(final["run_directory"])
        test_command = [
            args.python,
            str(args.code_root / "scripts" / "evaluate_micro_mlip_five_round_test.py"),
            "--run-dir", str(run_dir),
            "--dataset-dir", str(args.dataset_dir),
            "--molecule", molecule,
            "--release-manifest", str(barrier),
            "--device", args.device,
            "--default-dtype", "float32",
        ]
        entry = invoke(test_command, args.campaign_root / "episodes" / episode_key(molecule, seed, arm) / "test_controller.log")
        if entry["exit_code"] != 0:
            state["status"] = "FAILED_NO_AUTO_RETRY"
            state["failure"] = "independent_test_evaluation_failed"
        else:
            state["status"] = "COMPLETED"
            state["test_access"] = "RELEASED_AFTER_FOUR_ARM_BARRIER"
            state["independent_evaluation"] = {
                "path": str(run_dir / "independent_evaluation.json"),
                "sha256": sha256(run_dir / "independent_evaluation.json"),
                "controller": entry,
            }
        write_state(args.campaign_root / "episodes" / episode_key(molecule, seed, arm) / "state.json", state)
        if entry["exit_code"] != 0:
            raise RuntimeError(f"Independent evaluator failed for {molecule}/seed{seed}/{arm}")


def write_validation_only_evidence(
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
    molecule: str,
    seed: int,
    states: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Freeze the outer-loop admissible record before independent release.

    This is intentionally called while every arm is still at
    ``FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER``.  The generated
    document has no independent-evaluation fields, so a later g1 proposal can
    consume the development trajectory even though the inner controller will
    subsequently perform its one allowed release.
    """

    payload = validation_only_document(
        campaign_id=str(campaign["campaign_id"]),
        factorial_cell_id=campaign.get("factorial_cell_id"),
        factorial_levels=campaign.get("factorial_levels"),
        molecule=molecule,
        seed=seed,
        states=states,
    )
    path = args.campaign_root / "validation_only_evidence" / f"{molecule}__seed{seed}.json"
    if path.exists():
        existing = load_json(path)
        if existing != payload:
            raise ValueError("Validation-only evidence differs from the frozen pre-release record")
        return path
    write_json(path, payload)
    return path


def initial_campaign_manifest(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    policies: Mapping[str, Mapping[str, Any]],
    input_hashes: Mapping[str, str],
    binding_hash: str,
    resource_gate_hash: str,
) -> Dict[str, Any]:
    return {
        "schema": "micro-mlip-five-round-campaign-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "campaign_id": args.campaign_id,
        "method": args.method,
        "factorial_cell_id": args.factorial_cell_id,
        "factorial_levels": args.factorial_levels,
        "config": str(args.config),
        "dataset_dir": str(args.dataset_dir),
        "policies": str(args.policies),
        "source_sha256": {
            "config": sha256(args.config),
            "policies": sha256(args.policies),
            "policy_binding": binding_hash,
            "resource_gate": resource_gate_hash,
            "runner": sha256(args.mace_runner),
            "foundation_model": sha256(args.foundation_model) if args.foundation_model is not None else None,
            "replay_file": sha256(args.replay_file) if args.replay_file is not None else None,
            "replay_validation_file": (
                sha256(args.replay_valid_file) if args.replay_valid_file is not None else None
            ),
            **input_hashes,
        },
        "runtime": {
            "python": args.python,
            "mace_runner": str(args.mace_runner),
            "mace_site": str(args.mace_site),
            "device": args.device,
            "foundation_model": str(args.foundation_model) if args.foundation_model is not None else None,
            "replay_file": str(args.replay_file) if args.replay_file is not None else None,
            "replay_validation_file": (
                str(args.replay_valid_file) if args.replay_valid_file is not None else None
            ),
            "resource_check_note": args.resource_check_note,
        },
        "policies_by_arm": {arm: policies[arm]["policy_sha256"] for arm in ARMS},
        "expected_training_runs": len(args.molecules) * len(args.seeds) * len(ARMS) * ROUND_COUNT,
        "completed_training_runs": 0,
        "status_policy": "No automatic restart, replacement, or policy substitution.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--policy-binding", type=Path, required=True)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument("--resource-gate-max-age-seconds", type=int, default=900)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--method",
        choices=(
            "scratch_micro",
            "foundation_naive",
            "lora_rank4",
            "layer_freeze5",
            "multihead_pseudolabel_replay",
        ),
        default="scratch_micro",
    )
    parser.add_argument("--foundation-model", type=Path)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--replay-valid-file", type=Path)
    parser.add_argument("--molecule", dest="molecules", action="append", required=True)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, required=True)
    parser.add_argument("--resource-check-note", required=True)
    parser.add_argument("--factorial-cell-id")
    parser.add_argument("--factorial-levels-json")
    args = parser.parse_args()

    args.factorial_levels = None
    if args.factorial_levels_json is not None:
        try:
            parsed_levels = json.loads(args.factorial_levels_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--factorial-levels-json must be a JSON object") from exc
        if not isinstance(parsed_levels, Mapping):
            raise ValueError("--factorial-levels-json must be a JSON object")
        args.factorial_levels = dict(parsed_levels)
    policy_document = load_json(args.policies)
    manifest_cell_id = policy_document.get("factorial_cell_id")
    manifest_levels = policy_document.get("factorial_levels")
    if args.factorial_cell_id is not None and manifest_cell_id != args.factorial_cell_id:
        raise ValueError("Requested factorial cell does not match the frozen policy manifest")
    if args.factorial_levels is not None and manifest_levels != args.factorial_levels:
        raise ValueError("Requested factorial levels do not match the frozen policy manifest")
    if args.factorial_cell_id is None and manifest_cell_id is not None:
        args.factorial_cell_id = manifest_cell_id
    if args.factorial_levels is None and isinstance(manifest_levels, Mapping):
        args.factorial_levels = dict(manifest_levels)

    config = load_json(args.config)
    method_definition = config.get("scientific_environment", {}).get("methods", {}).get(args.method)
    if not isinstance(method_definition, Mapping):
        raise ValueError(f"Campaign method is not registered in the five-round configuration: {args.method}")
    if not str(method_definition.get("status", "")).startswith("READY"):
        raise ValueError(f"Campaign method is not admitted: {args.method}/{method_definition.get('status')}")
    if bool(method_definition.get("foundation", False)):
        if args.foundation_model is None or not args.foundation_model.is_file():
            raise ValueError(f"Campaign method {args.method} requires a readable --foundation-model")
    elif args.foundation_model is not None:
        raise ValueError("scratch_micro campaign may not receive a foundation model")
    if bool(method_definition.get("multiheads_finetuning", False)):
        if args.replay_file is None or not args.replay_file.is_file():
            raise ValueError(f"Campaign method {args.method} requires a readable --replay-file")
        if bool(method_definition.get("replay_validation_required", False)):
            if args.replay_valid_file is None or not args.replay_valid_file.is_file():
                raise ValueError(f"Campaign method {args.method} requires a readable --replay-valid-file")
    elif args.replay_file is not None:
        raise ValueError("Only a registered multihead method may receive a replay file")
    elif args.replay_valid_file is not None:
        raise ValueError("Only a registered multihead method may receive a replay validation file")
    unknown_molecules = set(args.molecules).difference(config["molecule_roles"])
    if unknown_molecules:
        raise ValueError(f"Unregistered molecules: {sorted(unknown_molecules)}")
    unknown_seeds = set(args.seeds).difference(config["scientific_environment"]["training"]["seeds"])
    if unknown_seeds:
        raise ValueError(f"Unregistered seeds: {sorted(unknown_seeds)}")
    if not args.mace_runner.is_file():
        raise FileNotFoundError(f"Missing MACE runner: {args.mace_runner}")
    if not args.mace_site.is_dir():
        raise FileNotFoundError(f"Missing MACE site directory: {args.mace_site}")
    policies = validated_policies(args.policies)
    assert_policy_method(policies, args.method)
    input_hashes = assert_inputs(config, args.dataset_dir)
    binding_hash = assert_policy_binding(
        args.policy_binding, args.config, args.dataset_dir, args.policies, policies, args.code_root
    )
    resource_gate_hash = assert_resource_gate(
        args.resource_gate, args.resource_gate_max_age_seconds
    )
    manifest_path = args.campaign_root / "campaign_manifest.json"
    if manifest_path.exists():
        campaign = load_json(manifest_path)
        if campaign.get("campaign_id") != args.campaign_id:
            raise ValueError("Campaign root belongs to a different campaign ID")
        if campaign.get("source_sha256", {}).get("config") != sha256(args.config):
            raise ValueError("Cannot resume with a changed configuration")
        if campaign.get("source_sha256", {}).get("policies") != sha256(args.policies):
            raise ValueError("Cannot resume with changed frozen policies")
        if campaign.get("source_sha256", {}).get("policy_binding") != binding_hash:
            raise ValueError("Cannot resume with a changed policy-environment binding")
        if campaign.get("source_sha256", {}).get("resource_gate") != resource_gate_hash:
            raise ValueError("Cannot resume with a changed shared-resource gate")
        if campaign.get("method") != args.method:
            raise ValueError("Cannot resume a campaign with a different training method")
        if campaign.get("source_sha256", {}).get("foundation_model") != (
            sha256(args.foundation_model) if args.foundation_model is not None else None
        ):
            raise ValueError("Cannot resume with a changed foundation model artifact")
        if campaign.get("source_sha256", {}).get("replay_file") != (
            sha256(args.replay_file) if args.replay_file is not None else None
        ):
            raise ValueError("Cannot resume with a changed replay artifact")
        if campaign.get("source_sha256", {}).get("replay_validation_file") != (
            sha256(args.replay_valid_file) if args.replay_valid_file is not None else None
        ):
            raise ValueError("Cannot resume with a changed replay validation artifact")
        if campaign.get("status") == "FAILED_NO_AUTO_RETRY":
            raise RuntimeError("A failed campaign is immutable; create a new versioned campaign instead")
    else:
        if args.campaign_root.exists():
            raise FileExistsError("Campaign root exists without a campaign manifest")
        args.campaign_root.mkdir(parents=True)
        campaign = initial_campaign_manifest(
            args, config, policies, input_hashes, binding_hash, resource_gate_hash
        )
        write_json(manifest_path, campaign)

    try:
        campaign["status"] = "RUNNING"
        write_json(manifest_path, campaign)
        for molecule in args.molecules:
            for seed in args.seeds:
                states = {
                    arm: load_or_prepare_episode(args, policies, molecule, seed, arm)
                    for arm in ARMS
                }
                if any(state.get("status") == "FAILED_NO_AUTO_RETRY" for state in states.values()):
                    raise RuntimeError("A failed episode is immutable; create a new versioned campaign instead")
                for round_index in range(1, ROUND_COUNT + 1):
                    for arm in ARMS:
                        state = states[arm]
                        if state.get("status") == "COMPLETED":
                            continue
                        if len(state["rounds"]) >= round_index:
                            continue
                        row = run_stage(
                            args,
                            args.campaign_root / "episodes" / episode_key(molecule, seed, arm),
                            state,
                            round_index,
                        )
                        state_after_round(config, state, row)
                        write_state(args.campaign_root / "episodes" / episode_key(molecule, seed, arm) / "state.json", state)
                        if state.get("status") == "FAILED_NO_AUTO_RETRY":
                            raise RuntimeError(f"Training or validation failed for {molecule}/seed{seed}/{arm}")
                write_validation_only_evidence(args, campaign, molecule, seed, states)
                run_independent_evaluations(args, molecule, seed, states)
        campaign["status"] = "COMPLETED_PENDING_ANALYSIS"
        campaign["completed_at"] = utc_now()
        campaign["completed_training_runs"] = campaign["expected_training_runs"]
        write_json(manifest_path, campaign)
        print(json.dumps({"status": campaign["status"], "campaign_root": str(args.campaign_root)}, indent=2))
        return 0
    except Exception as exc:
        campaign["status"] = "FAILED_NO_AUTO_RETRY"
        campaign["failed_at"] = utc_now()
        campaign["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, campaign)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

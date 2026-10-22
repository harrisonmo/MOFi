#!/usr/bin/env python3
"""Freeze one five-axis factorial cell's matched four-arm policy manifest.

Only development evidence may reach the outer-policy call.  The cell compiler
turns an LLM response into a finite, hashable sampler table and rejects any
policy that crosses the cell's Prompt/Data/Tool/Workflow contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_axis import (
    RAW_CONDITION_IDS,
    allowed_samplers,
    assert_policy_matches_cell,
    manual_cell_policy,
    raw_policy_from_response,
    static_policy_from_actions,
)
from rsi.micro_mlip.five_round import ADAPTIVE_ROUNDS, RESIDUAL_CLASSES, conditional_policy, policy_hash
from scripts.propose_micro_mlip_five_round_policies import invoke, safe_development_evidence, write_json


DEFAULT_SCREEN_CONFIG = ROOT / "configs" / "micro-mlip-five-axis-screen-v1.json"
DEFAULT_FIVE_ROUND_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def action_schema(actions: tuple[str, ...]) -> Dict[str, Any]:
    return {"type": "string", "enum": list(actions)}


def static_schema(actions: tuple[str, ...]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": {str(round_index): action_schema(actions) for round_index in ADAPTIVE_ROUNDS},
                "required": [str(round_index) for round_index in ADAPTIVE_ROUNDS],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1800},
        },
        "required": ["rounds", "rationale"],
    }


def conditional_schema(actions: tuple[str, ...]) -> Dict[str, Any]:
    by_class = {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: action_schema(actions) for name in RESIDUAL_CLASSES},
        "required": list(RESIDUAL_CLASSES),
    }
    rule = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"default_action": action_schema(actions), "by_residual_class": by_class},
        "required": ["default_action", "by_residual_class"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": {str(round_index): rule for round_index in ADAPTIVE_ROUNDS},
                "required": [str(round_index) for round_index in ADAPTIVE_ROUNDS],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1800},
        },
        "required": ["rounds", "rationale"],
    }


def raw_schema(actions: tuple[str, ...]) -> Dict[str, Any]:
    rule = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "condition_id": {"type": "string", "enum": list(RAW_CONDITION_IDS)},
            "if_true": action_schema(actions),
            "if_false": action_schema(actions),
            "if_missing": action_schema(actions),
        },
        "required": ["condition_id", "if_true", "if_false", "if_missing"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": {str(round_index): rule for round_index in ADAPTIVE_ROUNDS},
                "required": [str(round_index) for round_index in ADAPTIVE_ROUNDS],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1800},
        },
        "required": ["rounds", "rationale"],
    }


def cell_entry(registry: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    if registry.get("schema") not in {
        "micro-mlip-five-axis-resolution-v-registry-v2",
        "micro-mlip-pdtw-full-scratch-registry-v1",
    }:
        raise ValueError("Unexpected five-axis registry schema")
    matches = [entry for entry in registry.get("cells", []) if entry.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Registry has no unique cell {cell_id}")
    cell = matches[0]
    if cell.get("status") != "READY":
        raise ValueError(f"Cell {cell_id} is not admitted: {cell.get('blockers')}")
    levels = cell.get("levels")
    if not isinstance(levels, Mapping) or levels.get("model") != "low":
        raise ValueError("This compiler only admits model-low scratch cells")
    return cell


def response_policy(
    response: Mapping[str, Any], levels: Mapping[str, str], diagnostic: Mapping[str, Any]
) -> Dict[str, Any]:
    rounds = response.get("rounds")
    rationale = response.get("rationale")
    if not isinstance(rounds, Mapping) or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Outer policy response lacks rounds or rationale")
    workflow = levels["workflow"]
    tool = levels["tool"]
    if workflow == "low":
        policy = static_policy_from_actions(
            {int(round_index): str(action) for round_index, action in rounds.items()}, levels
        )
    elif tool == "high":
        normalized = {
            int(round_index): dict(value["by_residual_class"])
            for round_index, value in rounds.items()
            if isinstance(value, Mapping)
        }
        if set(normalized) != set(ADAPTIVE_ROUNDS):
            raise ValueError("Typed response has malformed conditional rounds")
        default_actions = {str(value["default_action"]) for value in rounds.values() if isinstance(value, Mapping)}
        if len(default_actions) != 1:
            raise ValueError("Conditional policy must use one frozen default action")
        policy = conditional_policy(normalized, default_actions.pop())
    elif tool == "low":
        policy = raw_policy_from_response(rounds, diagnostic)
    else:
        raise ValueError("Invalid Tool/Workflow levels")
    policy["rationale"] = rationale.strip()
    assert_policy_matches_cell(policy, levels)
    return policy


def outer_prompt(
    levels: Mapping[str, str], evidence: Mapping[str, Any], actions: tuple[str, ...]
) -> str:
    prompt = (
        "You are a bounded Evolution Loop selecting a finite five-round sampler table. "
        "Rounds 2 through 5 add exactly 64 labels after a common 128-label start. "
        f"Your only available sampler actions are {list(actions)}. Do not change the model recipe, split, "
        "evaluator, label budget, test access, or generate code. "
    )
    if levels["prompt"] == "high":
        prompt += (
            "Use the supplied development evidence and state a concise evidence-grounded rationale. "
            "Any validation pattern is an operational sampling hypothesis, never a demonstrated physical cause. "
        )
    else:
        prompt += "Use the supplied development evidence to select the finite table. "
    if levels["workflow"] == "low":
        prompt += "This cell has a static workflow: choose one action for each round before training begins. "
    elif levels["tool"] == "high":
        prompt += (
            "This cell exposes the typed validation_residual_v1 tool. Map every registered residual class "
            "to an action for each round. "
        )
    else:
        prompt += (
            "This cell exposes raw validation metrics only. For every round choose one registered numeric "
            "condition ID and the action for true, false, and missing values. "
        )
    return prompt + "Development-only evidence follows:\n\n" + json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def removed_prompt(actions: tuple[str, ...]) -> str:
    return (
        "You are the information-removed control for a five-round molecular-potential benchmark. "
        "You receive no development metrics, structures, labels, validation values, residuals, or holdout information. "
        f"Choose a static action from {list(actions)} for each of rounds 2 through 5. "
        "Do not propose any model, split, evaluator, budget, test-access, or code change."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-config", type=Path, default=DEFAULT_SCREEN_CONFIG)
    parser.add_argument("--five-round-config", type=Path, default=DEFAULT_FIVE_ROUND_CONFIG)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite cell-policy output root: {args.output_root}")
    config = load_json(args.screen_config)
    five_round_config = load_json(args.five_round_config)
    if config.get("parent_benchmark_id") != five_round_config.get("benchmark_id"):
        raise ValueError("Five-axis configuration points to a different five-round benchmark")
    if config.get("parent_five_round_config_sha256") != sha256(args.five_round_config):
        raise ValueError("Five-axis configuration does not match the supplied five-round configuration")
    registry = load_json(args.registry)
    if registry.get("config_sha256") != sha256(args.screen_config):
        raise ValueError("Registry was not built from the supplied five-axis configuration")
    cell = cell_entry(registry, args.cell_id)
    levels = cell["levels"]
    evidence = safe_development_evidence(args.development_evidence)
    actions = allowed_samplers(levels)
    diagnostic = five_round_config["validation_residual_diagnostic"]
    schema = (
        static_schema(actions)
        if levels["workflow"] == "low"
        else conditional_schema(actions)
        if levels["tool"] == "high"
        else raw_schema(actions)
    )

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "cell_contract.json",
        {
            "schema": "micro-mlip-five-axis-cell-contract-v1",
            "created_at": utc_now(),
            "generation": args.generation,
            "screen_config": str(args.screen_config),
            "screen_config_sha256": sha256(args.screen_config),
            "registry": str(args.registry),
            "registry_sha256": sha256(args.registry),
            "cell": cell,
            "allowed_samplers": list(actions),
            "model_boundary": "scratch_micro only; model-high methods require their separate admission.",
            "test_boundary": "Neither outer call receives policy-holdout or independent-test information.",
        },
    )
    write_json(args.output_root / "development_evidence.json", evidence)
    outer_response, outer_receipt = invoke(
        args.codex,
        args.model,
        args.reasoning_effort,
        args.timeout_seconds,
        "outer_generated",
        schema,
        outer_prompt(levels, evidence, actions),
        args.output_root,
    )
    outer = response_policy(outer_response, levels, diagnostic)
    write_json(args.output_root / "outer_generated" / "policy.json", outer)
    removed_response, removed_receipt = invoke(
        args.codex,
        args.model,
        args.reasoning_effort,
        args.timeout_seconds,
        "removed",
        static_schema(actions),
        removed_prompt(actions),
        args.output_root,
    )
    removed_rounds = removed_response.get("rounds")
    if not isinstance(removed_rounds, Mapping):
        raise ValueError("Information-removed response lacks static rounds")
    removed = static_policy_from_actions(
        {int(round_index): str(action) for round_index, action in removed_rounds.items()},
        levels,
        removed=True,
    )
    removed["rationale"] = str(removed_response.get("rationale", "")).strip()
    write_json(args.output_root / "removed" / "policy.json", removed)
    fixed = static_policy_from_actions(
        {round_index: "random" for round_index in ADAPTIVE_ROUNDS},
        {**levels, "tool": "low"},
    )
    fixed["rationale"] = "Pre-registered static random baseline for this factorial cell."
    manual = manual_cell_policy(levels, diagnostic)
    policies = {
        "schema": "micro-mlip-five-round-policies-v1",
        "status": "FROZEN_AFTER_DEVELOPMENT_BEFORE_NEW_FIVE_ROUND_EPISODES",
        "created_at": utc_now(),
        "generation": args.generation,
        "factorial_cell_id": args.cell_id,
        "factorial_levels": levels,
        "development_evidence": str(args.development_evidence),
        "development_evidence_sha256": sha256(args.development_evidence),
        "policy_generation_boundary": (
            "The outer-generated call received only the cleaned development evidence. The removed call received "
            "the same model interface but no metrics or validation context. Neither call received holdout or test data."
        ),
        "arms": {
            "fixed": {"origin": "pre_registered_static", "policy": fixed, "policy_sha256": policy_hash(fixed)},
            "outer_generated": {
                "origin": "one_gpt-6-astra_cell-constrained_development_evidence_call",
                "policy": outer,
                "policy_sha256": policy_hash(outer),
                "proposal_receipt": "outer_generated/result.json",
            },
            "manual_control": {
                "origin": "pre_registered_cell_matched_manual_control",
                "policy": manual,
                "policy_sha256": policy_hash(manual),
            },
            "removed": {
                "origin": "one_gpt-6-astra_cell-constrained_information_removed_call",
                "policy": removed,
                "policy_sha256": policy_hash(removed),
                "proposal_receipt": "removed/result.json",
            },
        },
        "outer_call_receipts": {"outer_generated": outer_receipt, "removed": removed_receipt},
    }
    write_json(args.output_root / "policies.json", policies)
    print(
        json.dumps(
            {
                "status": policies["status"],
                "cell_id": args.cell_id,
                "policies_sha256": sha256(args.output_root / "policies.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

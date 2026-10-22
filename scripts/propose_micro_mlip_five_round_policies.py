#!/usr/bin/env python3
"""Freeze five-round policy tables from development-only evidence.

Two real outer-model calls are retained: an evidence-aware conditional policy
and an information-removed static policy.  The resulting policy language is a
finite sampler table, never executable code or an evaluator modification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.generation_evidence import GENERATION_SCHEMA, sanitize_generation_evidence
from rsi.micro_mlip.five_round import (
    ADAPTIVE_ROUNDS,
    ALLOWED_SAMPLERS,
    RESIDUAL_CLASSES,
    conditional_policy,
    policy_core,
    policy_hash,
    static_policy,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def outer_schema() -> Dict[str, Any]:
    classes = {
        residual: {"type": "string", "enum": list(ALLOWED_SAMPLERS)} for residual in RESIDUAL_CLASSES
    }
    rule = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "default_action": {"type": "string", "enum": list(ALLOWED_SAMPLERS)},
            "by_residual_class": {
                "type": "object",
                "additionalProperties": False,
                "properties": classes,
                "required": list(RESIDUAL_CLASSES),
            },
        },
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


def removed_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    str(round_index): {"type": "string", "enum": list(ALLOWED_SAMPLERS)}
                    for round_index in ADAPTIVE_ROUNDS
                },
                "required": [str(round_index) for round_index in ADAPTIVE_ROUNDS],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1800},
        },
        "required": ["rounds", "rationale"],
    }


def safe_development_evidence(path: Path) -> Dict[str, Any]:
    """Extract only named development rows; do not forward holdout metadata."""

    source = load_json(path)
    if source.get("schema") == GENERATION_SCHEMA:
        safe = sanitize_generation_evidence(source)
        safe["source_schema"] = source.get("schema")
        return safe
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Development evidence has no rows")
    allowed_molecules = {"ethanol", "malonaldehyde"}
    cleaned = []
    required = {
        "molecule",
        "selection",
        "label_budget",
        "energy_mae_meV_per_atom",
        "force_component_mae_meV_per_A",
        "training_duration_seconds",
        "evaluation_duration_seconds",
    }
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("Development evidence has an incomplete row")
        if row["molecule"] not in allowed_molecules:
            raise ValueError("Development evidence includes a non-development molecule")
        if row["selection"] not in ALLOWED_SAMPLERS or int(row["label_budget"]) not in {128, 256}:
            raise ValueError("Development evidence contains an unregistered v1 action")
        cleaned.append({key: row[key] for key in sorted(required)})
    return {
        "source_schema": source.get("schema"),
        "development_molecules": sorted({row["molecule"] for row in cleaned}),
        "rows": cleaned,
        "information_boundary": (
            "Development-only evidence. No policy-holdout molecule, structure, label, test metric, "
            "or evaluator artifact is included."
        ),
    }


def invoke(
    executable: str,
    model: str,
    reasoning: str,
    timeout_seconds: int,
    label: str,
    schema: Mapping[str, Any],
    prompt: str,
    output_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Make exactly one structured, read-only outer-model call and retain receipt."""

    call_root = output_root / label
    call_root.mkdir(parents=True, exist_ok=False)
    workspace = call_root / "workspace"
    workspace.mkdir()
    schema_path = call_root / "schema.json"
    final_pending = call_root / ".final.pending.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    full_prompt = (
        "Act only as a bounded policy-table selector. Do not use tools, commands, web search, connectors, "
        "file access, or code generation. Return only one JSON object matching the supplied schema.\n\n"
        + prompt
    )
    (call_root / "prompt.txt").write_text(full_prompt, encoding="utf-8")
    command = [
        executable,
        "exec",
        "--model", model,
        "--json",
        "--output-schema", str(schema_path),
        "--output-last-message", str(final_pending),
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s", "read-only",
        "-C", str(workspace),
        "-c", f"model_reasoning_effort={json.dumps(reasoning)}",
        "-",
    ]
    write_json(call_root / "command.json", {"argv": command, "prompt_transport": "stdin"})
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().endswith("API_KEY")
        and key.upper() not in {"OPENAI_BASE_URL", "OPENAI_ACCESS_TOKEN"}
    }
    environment["PYTHONUTF8"] = "1"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=full_prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (call_root / "events.jsonl").write_text(exc.stdout or "", encoding="utf-8")
        (call_root / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
        raise RuntimeError(f"Outer policy call timed out after {timeout_seconds} seconds") from exc
    duration = round(time.monotonic() - started, 3)
    events_path = call_root / "events.jsonl"
    stderr_path = call_root / "stderr.log"
    events_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    final_path = call_root / "final.json"
    if final_pending.exists():
        final_pending.replace(final_path)
    else:
        final_path.write_text("null\n", encoding="utf-8")
    usage = None
    actual_model = None
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
        for candidate in (event, event.get("item"), event.get("response"), event.get("session")):
            if isinstance(candidate, dict) and isinstance(candidate.get("model"), str):
                actual_model = candidate["model"]
    receipt: Dict[str, Any] = {
        "label": label,
        "requested_model": model,
        "actual_model": actual_model,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "usage": usage,
        "events_sha256": sha256(events_path),
        "stderr_sha256": sha256(stderr_path),
        "final_sha256": sha256(final_path),
    }
    if completed.returncode:
        receipt["status"] = "FAILED"
        write_json(call_root / "result.json", receipt)
        raise RuntimeError(f"Outer policy call failed with exit code {completed.returncode}")
    try:
        response = json.loads(final_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise ValueError("final response is not a JSON object")
    except Exception as exc:
        receipt["status"] = "INVALID_OUTPUT"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_json(call_root / "result.json", receipt)
        raise RuntimeError("Outer policy response was invalid JSON") from exc
    receipt["status"] = "COMPLETED"
    write_json(call_root / "result.json", receipt)
    return response, receipt


def manual_policy() -> Dict[str, Any]:
    rules: Dict[int, Dict[str, str]] = {}
    for round_index in ADAPTIVE_ROUNDS:
        rules[round_index] = {
            "force_tail_or_undercoverage": "geometry_coverage",
            "force_dominant": "geometry_coverage",
            "energy_dominant": "random",
            "joint_high_error": "geometry_coverage",
            "plateau_or_regression": "random" if round_index % 2 == 0 else "geometry_coverage",
            "balanced_or_improving": "random",
        }
    policy = conditional_policy(rules, "random")
    policy["rationale"] = "Pre-registered validation-pattern control; it is not generated from the holdout."
    return policy


def outer_policy_from_response(response: Mapping[str, Any]) -> Dict[str, Any]:
    rationale = response.get("rationale")
    rounds = response.get("rounds")
    if not isinstance(rationale, str) or not rationale.strip() or not isinstance(rounds, Mapping):
        raise ValueError("Outer response has invalid rationale or rounds")
    policy = {
        "schema": "micro-mlip-five-round-policy-v1",
        "method": "scratch_micro",
        "diagnostic_route": "validation_residual_v1",
        "rounds": {
            key: {
                "mode": "conditional",
                "default_action": value["default_action"],
                "by_residual_class": value["by_residual_class"],
            }
            for key, value in rounds.items()
        },
        "rationale": rationale.strip(),
    }
    policy_core(policy)
    return policy


def removed_policy_from_response(response: Mapping[str, Any]) -> Dict[str, Any]:
    rationale = response.get("rationale")
    rounds = response.get("rounds")
    if not isinstance(rationale, str) or not rationale.strip() or not isinstance(rounds, Mapping):
        raise ValueError("Removed response has invalid rationale or rounds")
    policy = static_policy({int(key): value for key, value in rounds.items()}, route="removed")
    policy["rationale"] = rationale.strip()
    return policy


def outer_prompt(evidence: Mapping[str, Any]) -> str:
    return (
        "You are the Evolution Loop for a five-training-round molecular-potential benchmark. "
        "Freeze a sampler table for rounds 2 through 5. All arms start with the same 128 labels and add "
        "exactly 64 labels per later round; available samplers are only random and geometry_coverage. "
        "The Research Loop will classify validation-only residuals into the listed classes. Map every class "
        "in every round. The objective is lower final independent force error, using energy error and total "
        "training time only as secondary considerations. Do not claim a residual class is a physical cause; "
        "it is an operational sampling hypothesis. Do not propose changes to the model recipe, data split, "
        "evaluator, label budget, or test access. Development evidence follows:\n\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )


def removed_prompt() -> str:
    return (
        "You are the information-removed ablation for the same five-round molecular-potential benchmark. "
        "You receive no development metrics, structures, labels, validation residuals, or holdout information. "
        "Choose one static sampler, random or geometry_coverage, for each of rounds 2 through 5. "
        "All rounds have the same fixed 64-label increment. Do not propose a conditional rule or any change "
        "to model, data split, evaluator, budget, or test access."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--generation", default="g0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite policy output root: {args.output_root}")
    evidence = safe_development_evidence(args.development_evidence)
    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "development_evidence.json", evidence)
    outer_response, outer_receipt = invoke(
        args.codex,
        args.model,
        args.reasoning_effort,
        args.timeout_seconds,
        "outer_generated",
        outer_schema(),
        outer_prompt(evidence),
        args.output_root,
    )
    outer_policy = outer_policy_from_response(outer_response)
    write_json(args.output_root / "outer_generated" / "policy.json", outer_policy)
    removed_response, removed_receipt = invoke(
        args.codex,
        args.model,
        args.reasoning_effort,
        args.timeout_seconds,
        "removed",
        removed_schema(),
        removed_prompt(),
        args.output_root,
    )
    removed_policy = removed_policy_from_response(removed_response)
    write_json(args.output_root / "removed" / "policy.json", removed_policy)
    fixed_policy = static_policy({round_index: "random" for round_index in ADAPTIVE_ROUNDS})
    fixed_policy["rationale"] = "Pre-registered static random schedule."
    manual = manual_policy()
    policies = {
        "schema": "micro-mlip-five-round-policies-v1",
        "status": "FROZEN_AFTER_DEVELOPMENT_BEFORE_NEW_FIVE_ROUND_EPISODES",
        "created_at": utc_now(),
        "generation": args.generation,
        "development_evidence": str(args.development_evidence),
        "development_evidence_sha256": sha256(args.development_evidence),
        "policy_generation_boundary": (
            "The v1 development table was supplied only to outer_generated. The removed arm received the same "
            "model-call interface without metrics; neither call received policy-holdout information."
        ),
        "arms": {
            "fixed": {"origin": "pre_registered_static", "policy": fixed_policy, "policy_sha256": policy_hash(fixed_policy)},
            "outer_generated": {
                "origin": "one_gpt-6-astra_development_evidence_call",
                "policy": outer_policy,
                "policy_sha256": policy_hash(outer_policy),
                "proposal_receipt": "outer_generated/result.json",
            },
            "manual_control": {"origin": "pre_registered_conditional", "policy": manual, "policy_sha256": policy_hash(manual)},
            "removed": {
                "origin": "one_gpt-6-astra_information_removed_call",
                "policy": removed_policy,
                "policy_sha256": policy_hash(removed_policy),
                "proposal_receipt": "removed/result.json",
            },
        },
        "outer_call_receipts": {"outer_generated": outer_receipt, "removed": removed_receipt},
    }
    write_json(args.output_root / "policies.json", policies)
    print(json.dumps({"status": policies["status"], "policies_sha256": sha256(args.output_root / "policies.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

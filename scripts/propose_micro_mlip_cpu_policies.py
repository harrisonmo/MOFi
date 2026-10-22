#!/usr/bin/env python3
"""Freeze two bounded outer-loop policies from completed development evidence."""

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
from typing import Any, Dict, Mapping, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.pilot import policy_action, policy_hash, read_json


POLICY_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selection": {"type": "string", "enum": ["random", "geometry_coverage"]},
        "label_budget": {"type": "integer", "enum": [128, 256]},
        "method": {"type": "string", "enum": ["scratch_micro"]},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 1600},
    },
    "required": ["selection", "label_budget", "method", "rationale"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_response(value: Any) -> Tuple[Dict[str, Any], str]:
    if not isinstance(value, dict) or set(value) != {"selection", "label_budget", "method", "rationale"}:
        raise ValueError("Model response does not have the exact policy schema")
    action = policy_action({key: value[key] for key in ("selection", "label_budget", "method")})
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1600:
        raise ValueError("Model rationale is invalid")
    return action, rationale.strip()


def invoke_policy(
    executable: str,
    model: str,
    reasoning: str,
    timeout_seconds: int,
    label: str,
    prompt: str,
    output_root: Path,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Make exactly one structured, tool-free Codex CLI proposal and retain its receipt."""

    call_root = output_root / label
    call_root.mkdir(parents=True, exist_ok=False)
    workspace = call_root / "workspace"
    workspace.mkdir()
    schema_path = call_root / "schema.json"
    final_pending = call_root / ".final.pending.json"
    schema_path.write_text(json.dumps(POLICY_OUTPUT_SCHEMA, indent=2) + "\n", encoding="utf-8")
    full_prompt = (
        "Act only as a bounded policy selector. Do not use tools, commands, web search, connectors, "
        "or file access. Return only one JSON object matching the supplied schema.\n\n"
        + prompt
    )
    (call_root / "prompt.txt").write_text(full_prompt, encoding="utf-8")
    command = [
        executable,
        "exec",
        "--model",
        model,
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(final_pending),
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-C",
        str(workspace),
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning)}",
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
    (call_root / "events.jsonl").write_text(completed.stdout, encoding="utf-8")
    (call_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
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
        for source in (event, event.get("item"), event.get("session"), event.get("response")):
            if isinstance(source, dict) and isinstance(source.get("model"), str):
                actual_model = source["model"]
    if final_pending.exists():
        final_path = call_root / "final.json"
        final_pending.replace(final_path)
    else:
        final_path = call_root / "final.json"
        final_path.write_text("null\n", encoding="utf-8")
    receipt: Dict[str, Any] = {
        "label": label,
        "status": "FAILED" if completed.returncode else "COMPLETED",
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "requested_model": model,
        "actual_model": actual_model,
        "usage": usage,
        "events_sha256": sha256(call_root / "events.jsonl"),
        "stderr_sha256": sha256(call_root / "stderr.log"),
        "final_sha256": sha256(final_path),
    }
    if completed.returncode:
        write_json(call_root / "result.json", receipt)
        raise RuntimeError(f"Outer policy call failed with exit code {completed.returncode}")
    try:
        response = json.loads(final_path.read_text(encoding="utf-8"))
        action, rationale = validate_response(response)
    except Exception as exc:
        receipt["status"] = "INVALID_OUTPUT"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        write_json(call_root / "result.json", receipt)
        raise RuntimeError("Outer policy response was invalid") from exc
    receipt["status"] = "COMPLETED"
    receipt["action"] = action
    receipt["action_sha256"] = policy_hash(action)
    write_json(call_root / "result.json", receipt)
    return action, rationale, receipt


def development_prompt(summary: Mapping[str, Any]) -> str:
    return (
        "You are the Evolution Loop for a small molecular-potential benchmark. Select exactly one registered "
        "heldout training action. The allowed action menu is selection=random or geometry_coverage, "
        "label_budget=128 or 256, and method=scratch_micro. The objective is lower independent test force "
        "component MAE, with energy MAE and total train+evaluation time as tie breakers. Do not propose a new "
        "model, inspect any data, or infer anything about toluene. The following completed development receipts "
        "are the only evidence:\n\n"
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    )


def removed_prompt() -> str:
    return (
        "You are the diagnostic-route-removed ablation for the same small molecular-potential benchmark. "
        "You receive no development metrics, no structures, and no holdout information. Select exactly one "
        "registered heldout training action from selection=random or geometry_coverage, label_budget=128 or 256, "
        "and method=scratch_micro. Do not propose any other action."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-config", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing policy directory: {args.output_root}")
    admission = read_json(args.admission_config)
    development = read_json(args.development_summary)
    if development.get("phase") != "development" or len(development.get("rows", [])) != admission["budget"]["development_training_rounds"]:
        raise ValueError("Development summary is incomplete or does not match the frozen admission budget")
    args.output_root.mkdir(parents=True)
    policy_config = admission["policy_generation"]
    action_outer, rationale_outer, receipt_outer = invoke_policy(
        args.codex,
        policy_config["requested_model"],
        policy_config["reasoning_effort"],
        int(policy_config["timeout_seconds"]),
        "outer_generated",
        development_prompt(development),
        args.output_root,
    )
    action_removed, rationale_removed, receipt_removed = invoke_policy(
        args.codex,
        policy_config["requested_model"],
        policy_config["reasoning_effort"],
        int(policy_config["timeout_seconds"]),
        "removed",
        removed_prompt(),
        args.output_root,
    )
    controls = admission["pre_registered_controls"]
    fixed = policy_action(controls["fixed"]["action"])
    manual = policy_action(controls["manual_control"]["action"])
    policies = {
        "schema": "micro-mlip-rsi-cpu-admission-policies-v1",
        "status": "FROZEN_AFTER_DEVELOPMENT_BEFORE_HOLDOUT",
        "created_at": utc_now(),
        "admission_config_sha256": sha256(args.admission_config),
        "development_summary_sha256": sha256(args.development_summary),
        "arms": {
            "fixed": {
                "origin": "pre_registered",
                "rationale": controls["fixed"]["rationale"],
                "action": fixed,
                "action_sha256": policy_hash(fixed),
            },
            "outer_generated": {
                "origin": "one_outer_model_call_with_development_metrics",
                "rationale": rationale_outer,
                "action": action_outer,
                "action_sha256": policy_hash(action_outer),
                "proposal_receipt": "outer_generated/result.json",
            },
            "manual_control": {
                "origin": "pre_registered",
                "rationale": controls["manual_control"]["rationale"],
                "action": manual,
                "action_sha256": policy_hash(manual),
            },
            "removed": {
                "origin": "one_outer_model_call_without_development_metrics",
                "rationale": rationale_removed,
                "action": action_removed,
                "action_sha256": policy_hash(action_removed),
                "proposal_receipt": "removed/result.json",
            },
        },
        "outer_call_receipts": {
            "outer_generated": receipt_outer,
            "removed": receipt_removed,
        },
        "holdout_access": "NOT_STARTED",
    }
    write_json(args.output_root / "policies.json", policies)
    print(json.dumps({"status": policies["status"], "policies_sha256": sha256(args.output_root / "policies.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fail-closed verification of persisted structured CLI calls."""

import json
from jsonschema import validate
from rsi.backends.codex_cli import DECISION_INSTRUCTIONS, FORBIDDEN_ITEMS
from rsi.experiment import read_json


class AuditError(ValueError):
    """Saved evidence is incomplete, inconsistent or outside the frozen protocol."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def known_tokens(usage):
    require(isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
            for key in ("input_tokens", "output_tokens")), "Model usage is unknown or invalid")
    return usage["input_tokens"] + usage["output_tokens"]


def check_saved_call(directory, config, prompt=None, schema=None):
    """Cross-check CLI result, final JSON, command metadata and raw event usage."""
    result = read_json(directory / "result.json")
    saved_schema = read_json(directory / "schema.json")
    command = read_json(directory / "command.json")
    saved_prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
    if prompt is not None:
        require(saved_prompt == DECISION_INSTRUCTIONS + prompt, "Prompt replay differs")
    if schema is not None:
        require(saved_schema == schema, "Schema replay differs")
    require(saved_prompt.startswith(DECISION_INSTRUCTIONS), "Missing decision isolation instructions")
    require(result.get("status") == "ok" and result.get("exit_code") == 0, "Saved CLI call did not succeed")
    require(result.get("requested_model") == config["model"], "Requested model differs")
    require(command.get("requested_model") == config["model"] and command.get("reasoning") == config["reasoning"],
            "CLI model or reasoning differs")
    require(command.get("timeout_seconds") == config["cli_timeout_seconds"], "CLI timeout differs")
    require(command.get("authentication") == "existing_cli_login" and command.get("prompt_transport") == "stdin",
            "CLI authentication or prompt transport differs")
    require(read_json(directory / "final.json") == result.get("response"), "Final JSON differs from saved response")
    validate(result["response"], saved_schema)
    tokens = known_tokens(result.get("usage"))
    events, unexpected, completed, evidence = [], [], [], []
    for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise AuditError("Malformed raw CLI event") from exc
            require(isinstance(event, dict), "Raw CLI event is not an object")
            events.append(event)
    for event in events:
        require(event.get("type") not in {"error", "turn.failed"}, "Raw CLI failure event")
        item = event.get("item") or {}
        if isinstance(item, dict) and item.get("type") in FORBIDDEN_ITEMS:
            unexpected.append(item["type"])
        if event.get("type") == "turn.completed":
            completed.append(event.get("usage"))
        for source in (event, event.get("session"), event.get("response")):
            if isinstance(source, dict) and isinstance(source.get("model"), str):
                evidence.append({"event_type": event.get("type"), "model": source["model"]})
    require(not unexpected and result.get("unexpected_tool_types") == [], "Unexpected model tool use")
    require(len(completed) == 1 and completed[0] == result["usage"], "Raw CLI usage differs or contains extra turns")
    require(result.get("model_evidence") == evidence, "Model identity evidence differs")
    require(result.get("actual_model") == (evidence[-1]["model"] if evidence else None), "Actual model differs from evidence")
    require(result.get("actual_model") in {None, config["model"]}, "Explicit actual model differs from requested model")
    return result, tokens

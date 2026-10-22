#!/usr/bin/env python3
"""Record one non-scientific compatibility probe for the registered outer model."""

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
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"status": {"type": "string", "enum": ["ready"]}},
    "required": ["status"],
}
PROMPT = (
    "This is a compatibility probe for a bounded scientific-policy interface. "
    "Do not use tools, commands, files, web access, or reasoning about a scientific task. "
    "Return exactly the JSON object required by the supplied schema."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite a probe receipt: {args.output_root}")
    args.output_root.mkdir(parents=True)
    workspace = args.output_root / "workspace"
    workspace.mkdir()
    schema_path = args.output_root / "schema.json"
    final_pending = args.output_root / ".final.pending.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")
    command = [
        args.codex,
        "exec",
        "--model", args.model,
        "--json",
        "--output-schema", str(schema_path),
        "--output-last-message", str(final_pending),
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s", "read-only",
        "-C", str(workspace),
        "-c", f"model_reasoning_effort={json.dumps(args.reasoning_effort)}",
        "-",
    ]
    write_json(args.output_root / "command.json", {"argv": command, "prompt_transport": "stdin"})
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
            input=PROMPT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (args.output_root / "events.jsonl").write_text(exc.stdout or "", encoding="utf-8")
        (args.output_root / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
        raise RuntimeError(f"Outer CLI probe timed out after {args.timeout_seconds} seconds") from exc
    duration = round(time.monotonic() - started, 3)
    events_path = args.output_root / "events.jsonl"
    stderr_path = args.output_root / "stderr.log"
    events_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    final_path = args.output_root / "final.json"
    if final_pending.exists():
        final_pending.replace(final_path)
    else:
        final_path.write_text("null\n", encoding="utf-8")
    actual_model = None
    usage = None
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
    try:
        response = json.loads(final_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        response = None
    success = completed.returncode == 0 and response == {"status": "ready"}
    receipt: Dict[str, Any] = {
        "schema": "micro-mlip-outer-cli-probe-v1",
        "created_at": utc_now(),
        "status": "COMPLETED" if success else "FAILED",
        "purpose": "Versioned interface probe only; not an RSI policy or scientific result.",
        "requested_model": args.model,
        "actual_model": actual_model,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "usage": usage,
        "response": response,
        "events_sha256": sha256(events_path),
        "stderr_sha256": sha256(stderr_path),
        "final_sha256": sha256(final_path),
    }
    write_json(args.output_root / "result.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

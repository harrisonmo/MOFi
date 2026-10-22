"""One bounded, real structured-output call through a provider API backend.

Usage (from the repository root, in a shell where the key variable is set):
    python scripts/api_smoke.py --backend anthropic_api --model claude-opus-5-5
Writes a fresh runs/api-smoke-<backend>-<UTC time>/ directory and prints a
credential-free summary. Exits non-zero unless the call returns status ok.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.backends.factory import backend_identity, create_backend  # noqa: E402

SCHEMA = {"type": "object", "properties": {"ready": {"type": "boolean"}, "summary": {"type": "string"}},
          "required": ["ready", "summary"], "additionalProperties": False}
PROMPT = "Smoke test of the decision transport. Return ready=true and summary='API_SMOKE_OK'."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("anthropic_api", "openai_api"), default="anthropic_api")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    config = {"backend": args.backend, "model": args.model, "reasoning": args.reasoning,
              "cli_timeout_seconds": args.timeout, "api": {"max_output_tokens": 4000}}
    backend = create_backend(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runs" / f"api-smoke-{args.backend}-{stamp}"
    result = backend.run(PROMPT, SCHEMA, output)
    summary = {"output": output.relative_to(ROOT).as_posix(),"identity": backend_identity(backend),
               **{key: result[key] for key in ("status", "response", "usage", "requested_model",
                                               "actual_model", "stop_reason", "error", "duration_seconds")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "ok" and result["response"] == {"ready": True, "summary": "API_SMOKE_OK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

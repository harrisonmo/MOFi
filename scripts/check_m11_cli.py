"""Make one accounted Astra admission attempt; never retry or provision credentials."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.backends.codex_cli import CodexCLI
from rsi.experiment import cli_identity, git_state, write_json
from rsi.m11 import ADMISSION_LIMITS, ADMISSION_PROMPT, ADMISSION_SCHEMA


def run(output, protocol="original"):
    limit = ADMISSION_LIMITS[protocol]
    output.mkdir(parents=True, exist_ok=False)
    backend = CodexCLI(model="gpt-6-astra", reasoning="high", timeout=120)
    write_json(output / "manifest.json", {
        "root": git_state(ROOT), "cli": cli_identity(backend.executable),
        "admission_protocol": protocol,
        "max_calls": 1, "max_observed_tokens": limit, "science_units": 0,
        "authentication": "existing_cli_login", "automatic_retries": False,
    })
    write_json(output / "ledger.json", [{"label": "admission", "status": "reserved"}])
    result = backend.run(ADMISSION_PROMPT, ADMISSION_SCHEMA, output / "probe")
    usage = result.get("usage") or {}
    known = all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens"))
    tokens = sum(usage.get(key, 0) for key in ("input_tokens", "output_tokens")) if known else None
    write_json(output / "ledger.json", [{"label": "admission", "status": "known" if known else "unknown",
                                        "usage": usage, "result_status": result["status"]}])
    summary = {"status": result["status"], "usage_known": known, "tokens": tokens,
               "admitted": result["status"] == "ok" and known and tokens <= limit
                           and result["response"] == {"ready": True},
               "actual_model": result.get("actual_model"), "duration_seconds": result["duration_seconds"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=ADMISSION_LIMITS, default="original")
    args = parser.parse_args()
    result = run(args.output.resolve(), args.protocol)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["admitted"] else 1)

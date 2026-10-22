#!/usr/bin/env python3
"""Run and validate one model-derived adsorption decomposition evaluator.

The runner is deliberately one-shot: it records the command and model hashes,
does not retry a failed evaluator, and writes the success sentinel only after a
non-empty, schema-valid JSON result exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_numbers(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return True


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--scratch-model", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--exclude-source-name", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_prefix.with_name(args.output_prefix.name + ".state.json")
    log_path = args.output_prefix.with_suffix(".log")
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    done_path = args.output_prefix.with_suffix(".done")
    failed_path = args.output_prefix.with_suffix(".failed")
    for path in (args.evaluator, args.source_tar, args.reference_csv, args.scratch_model, args.foundation_model):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"required input missing or empty: {path}")

    command = [
        args.python,
        str(args.evaluator),
        "--source-tar",
        str(args.source_tar),
        "--exclude-source-name",
        *args.exclude_source_name,
        "--reference-csv",
        str(args.reference_csv),
        "--scratch-model",
        str(args.scratch_model),
        "--foundation-model",
        str(args.foundation_model),
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
        "--device",
        args.device,
    ]
    write_json(
        state_path,
        {
            "status": "RUNNING",
            "started_at": now(),
            "host": platform.node(),
            "python": args.python,
            "command": command,
            "inputs": {
                "source_tar": {"path": str(args.source_tar), "sha256": digest(args.source_tar)},
                "reference_csv": {"path": str(args.reference_csv), "sha256": digest(args.reference_csv)},
                "scratch_model": {"path": str(args.scratch_model), "sha256": digest(args.scratch_model)},
                "foundation_model": {"path": str(args.foundation_model), "sha256": digest(args.foundation_model)},
            },
            "excluded_source_names": args.exclude_source_name,
        },
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (args.output_prefix.with_suffix(".command.txt")).write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"STARTED_AT: {now()}\n")
        log.write(f"COMMAND: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, env=os.environ.copy())

    result = {
        "status": "FAILED",
        "finished_at": now(),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "log_sha256": digest(log_path),
    }
    if completed.returncode == 0 and json_path.is_file() and csv_path.is_file() and json_path.stat().st_size > 0 and csv_path.stat().st_size > 0:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        if payload.get("status") != "EXECUTED_EXACT_ID_MATCHES_ONLY":
            result["validation_error"] = "unexpected evaluator status"
        elif int(counts.get("evaluated_frames", 0)) <= 0:
            result["validation_error"] = "evaluator produced zero matched frames"
        elif not finite_numbers(payload):
            result["validation_error"] = "non-finite value in evaluator JSON"
        else:
            result.update(
                {
                    "status": "COMPLETED_VALIDATED",
                    "result_sha256": digest(json_path),
                    "csv_sha256": digest(csv_path),
                    "evaluated_frames": counts.get("evaluated_frames"),
                    "matched_mof_ids": counts.get("matched_mof_ids"),
                }
            )
    write_json(args.output_prefix.with_suffix(".result.json"), result)
    if result["status"] == "COMPLETED_VALIDATED":
        done_path.write_text(result["finished_at"] + "\n", encoding="utf-8")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(result)
        write_json(state_path, state)
        return 0

    failed_path.write_text(result["finished_at"] + "\n", encoding="utf-8")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(result)
    write_json(state_path, state)
    return int(completed.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())

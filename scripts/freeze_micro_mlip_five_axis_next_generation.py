#!/usr/bin/env python3
"""Freeze a g1 five-axis policy set from g0 validation-only trajectories.

This script is the outer-loop transition, not a training launcher.  It accepts
only the evidence documents emitted before independent release by the complete
g0 screen.  For each exact factorial cell it makes one bounded evidence-aware
proposal and one matched information-removed proposal, then audits the full
new policy block before any g1 MACE episode can start.
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
from rsi.micro_mlip.generation_evidence import GENERATION_SCHEMA, sanitize_generation_evidence
from scripts.run_micro_mlip_five_axis_screen import runnable_cells


DEFAULT_SCREEN_CONFIG = ROOT / "configs" / "micro-mlip-pdtw-full-scratch-screen-v1.json"
DEFAULT_FIVE_ROUND_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"


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


def prior_generation_evidence(screen_root: Path, cell_id: str, prior_generation: str) -> Path:
    path = screen_root / "cells" / cell_id / f"generation_{prior_generation}_evidence.json"
    source = load_json(path)
    safe = sanitize_generation_evidence(source)
    if safe.get("generation") != prior_generation:
        raise ValueError(f"Evidence generation mismatch for {cell_id}")
    if source.get("factorial_cell_id") != cell_id:
        raise ValueError(f"Evidence is not bound to the expected factorial cell: {cell_id}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-screen-root", type=Path, required=True)
    parser.add_argument("--prior-generation", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output-policy-root", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--screen-config", type=Path, default=DEFAULT_SCREEN_CONFIG)
    parser.add_argument("--five-round-config", type=Path, default=DEFAULT_FIVE_ROUND_CONFIG)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prior-policy-root", type=Path, required=True)
    parser.add_argument("--prior-policy-audit", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.output_policy_root.exists() or args.output_audit.exists():
        raise FileExistsError("New generation outputs must not overwrite an existing freeze")
    prior_manifest = load_json(args.prior_screen_root / "screen_manifest.json")
    if prior_manifest.get("status") != "COMPLETED_PENDING_G1_POLICY_FREEZE":
        raise ValueError("Prior screen has not completed its g0 training/evidence phase")
    if prior_manifest.get("generation") != args.prior_generation:
        raise ValueError("Prior screen generation does not match --prior-generation")
    if prior_manifest.get("source_sha256", {}).get("screen_config") != sha256(args.screen_config):
        raise ValueError("Prior screen does not match the supplied screen configuration")
    if prior_manifest.get("source_sha256", {}).get("registry") != sha256(args.registry):
        raise ValueError("Prior screen does not match the supplied registry")
    cells = runnable_cells(args.screen_config, args.registry, args.prior_policy_root, args.prior_policy_audit)
    if len(prior_manifest.get("cells", [])) != len(cells):
        raise ValueError("Prior screen does not contain the complete registered cell set")
    for row in prior_manifest["cells"]:
        if row.get("status") != "COMPLETED":
            raise ValueError("A prior factorial cell is incomplete")

    args.output_policy_root.mkdir(parents=True)
    manifest_path = args.output_policy_root / "generation_freeze_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-five-axis-generation-freeze-v1",
        "status": "RUNNING",
        "created_at": utc_now(),
        "prior_screen_root": str(args.prior_screen_root),
        "prior_generation": args.prior_generation,
        "generation": args.generation,
        "source_sha256": {
            "screen_config": sha256(args.screen_config),
            "five_round_config": sha256(args.five_round_config),
            "registry": sha256(args.registry),
            "prior_policy_audit": sha256(args.prior_policy_audit),
        },
        "cells": [],
        "information_boundary": "Only validated development training/validation evidence is admitted to the outer calls; independent outcomes are excluded.",
    }
    write_json(manifest_path, manifest)
    try:
        for cell in cells:
            cell_id = cell["cell_id"]
            evidence_path = prior_generation_evidence(args.prior_screen_root, cell_id, args.prior_generation)
            cell_record: Dict[str, Any] = {
                "cell_id": cell_id,
                "evidence": str(evidence_path),
                "evidence_sha256": sha256(evidence_path),
                "status": "RUNNING",
            }
            manifest["cells"].append(cell_record)
            write_json(manifest_path, manifest)
            command = [
                args.python,
                str(args.code_root / "scripts" / "propose_micro_mlip_five_axis_cell_policies.py"),
                "--screen-config", str(args.screen_config),
                "--five-round-config", str(args.five_round_config),
                "--registry", str(args.registry),
                "--cell-id", cell_id,
                "--development-evidence", str(evidence_path),
                "--output-root", str(args.output_policy_root / cell_id),
                "--codex", args.codex,
                "--model", args.model,
                "--reasoning-effort", args.reasoning_effort,
                "--timeout-seconds", str(args.timeout_seconds),
                "--generation", args.generation,
            ]
            result = invoke(command, args.output_policy_root / cell_id / "generation_freeze_controller.log")
            cell_record["controller"] = result
            if result["exit_code"] != 0:
                raise RuntimeError(f"Outer policy freeze failed for {cell_id}")
            cell_record["status"] = "COMPLETED"
            write_json(manifest_path, manifest)

        audit = invoke(
            [
                sys.executable,
                str(args.code_root / "scripts" / "audit_micro_mlip_five_axis_policies.py"),
                "--screen-config", str(args.screen_config),
                "--registry", str(args.registry),
                "--policy-root", str(args.output_policy_root),
                "--output", str(args.output_audit),
            ],
            args.output_policy_root / "policy_audit_controller.log",
        )
        manifest["audit"] = audit
        if audit["exit_code"] != 0:
            raise RuntimeError("New generation policy audit failed")
        manifest["status"] = "COMPLETED_BEFORE_G1_TRAINING"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "policy_root": str(args.output_policy_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

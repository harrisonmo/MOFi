#!/usr/bin/env python3
"""Run the complete g0 → g1 → toluene policy-holdout P/D/T/W program.

This is a serial orchestrator for the already frozen experiment contracts.  It
does not invent policies, retries, or scientific settings.  Every training
entry point retains its own fresh resource gate; therefore starting this
program while the formal MOF campaign is active fails closed before a MACE run.
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


def require_success(label: str, receipt: Mapping[str, Any]) -> None:
    if receipt.get("exit_code") != 0:
        raise RuntimeError(f"{label} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--program-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--five-round-config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--g0-policy-root", type=Path, required=True)
    parser.add_argument("--g0-policy-audit", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--g0-seed", type=int, default=20261020)
    parser.add_argument("--g1-seed", type=int, default=20261021)
    parser.add_argument("--holdout-seed", type=int, default=20261022)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.program_root.exists():
        raise FileExistsError("Program root exists; completed and failed programs are immutable")
    args.program_root.mkdir(parents=True)
    manifest_path = args.program_root / "program_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-full-pdtw-program-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "program_id": args.program_id,
        "source_sha256": {
            "screen_config": sha256(args.screen_config),
            "five_round_config": sha256(args.five_round_config),
            "registry": sha256(args.registry),
            "g0_policy_audit": sha256(args.g0_policy_audit),
        },
        "seeds": {"g0": args.g0_seed, "g1": args.g1_seed, "holdout": args.holdout_seed},
        "stages": {},
        "execution_boundary": "All scientific training is serial and individually resource-gated; no automatic retry or policy substitution.",
    }
    write_json(manifest_path, manifest)

    common_gate = [
        "--formal-launcher-pid", str(args.formal_launcher_pid),
        "--formal-state", str(args.formal_state),
        "--formal-done", str(args.formal_done),
        "--disk-path", str(args.disk_path),
        "--minimum-free-gib", str(args.minimum_free_gib),
    ]
    try:
        manifest["status"] = "RUNNING_G0"
        write_json(manifest_path, manifest)
        g0_root = args.program_root / "g0"
        g0 = invoke(
            [
                args.python,
                str(args.code_root / "scripts" / "run_micro_mlip_five_axis_screen.py"),
                "--screen-id", f"{args.program_id}__g0",
                "--screen-root", str(g0_root),
                "--generation", "g0",
                "--screen-config", str(args.screen_config),
                "--five-round-config", str(args.five_round_config),
                "--registry", str(args.registry),
                "--policy-root", str(args.g0_policy_root),
                "--policy-audit", str(args.g0_policy_audit),
                "--dataset-dir", str(args.dataset_dir),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--molecule", "ethanol",
                "--molecule", "malonaldehyde",
                "--seed", str(args.g0_seed),
                *common_gate,
            ],
            args.program_root / "g0_controller.log",
        )
        manifest["stages"]["g0"] = g0
        write_json(manifest_path, manifest)
        require_success("g0 full P/D/T/W screen", g0)

        manifest["status"] = "FREEZING_G1_POLICIES"
        write_json(manifest_path, manifest)
        g1_policy_root = args.program_root / "g1_policies"
        g1_audit = args.program_root / "g1_policy_audit.json"
        g1_freeze = invoke(
            [
                args.python,
                str(args.code_root / "scripts" / "freeze_micro_mlip_five_axis_next_generation.py"),
                "--prior-screen-root", str(g0_root),
                "--prior-generation", "g0",
                "--generation", "g1",
                "--output-policy-root", str(g1_policy_root),
                "--output-audit", str(g1_audit),
                "--screen-config", str(args.screen_config),
                "--five-round-config", str(args.five_round_config),
                "--registry", str(args.registry),
                "--prior-policy-root", str(args.g0_policy_root),
                "--prior-policy-audit", str(args.g0_policy_audit),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--codex", args.codex,
                "--model", args.model,
                "--reasoning-effort", args.reasoning_effort,
                "--timeout-seconds", str(args.timeout_seconds),
            ],
            args.program_root / "g1_policy_freeze_controller.log",
        )
        manifest["stages"]["g1_policy_freeze"] = g1_freeze
        write_json(manifest_path, manifest)
        require_success("g1 policy freeze", g1_freeze)

        manifest["status"] = "RUNNING_G1"
        write_json(manifest_path, manifest)
        g1_root = args.program_root / "g1"
        g1 = invoke(
            [
                args.python,
                str(args.code_root / "scripts" / "run_micro_mlip_five_axis_screen.py"),
                "--screen-id", f"{args.program_id}__g1",
                "--screen-root", str(g1_root),
                "--generation", "g1",
                "--screen-config", str(args.screen_config),
                "--five-round-config", str(args.five_round_config),
                "--registry", str(args.registry),
                "--policy-root", str(g1_policy_root),
                "--policy-audit", str(g1_audit),
                "--dataset-dir", str(args.dataset_dir),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--molecule", "ethanol",
                "--molecule", "malonaldehyde",
                "--seed", str(args.g1_seed),
                *common_gate,
            ],
            args.program_root / "g1_controller.log",
        )
        manifest["stages"]["g1"] = g1
        write_json(manifest_path, manifest)
        require_success("g1 full P/D/T/W screen", g1)

        manifest["status"] = "SELECTING_PROMOTION"
        write_json(manifest_path, manifest)
        registry = load_json(args.registry)
        cells = [cell["cell_id"] for cell in registry.get("cells", []) if cell.get("status") == "READY"]
        promotion_path = args.program_root / "promotion.json"
        promotion_command = [
            args.python,
            str(args.code_root / "scripts" / "select_micro_mlip_pdtw_promotion.py"),
            "--five-round-config", str(args.five_round_config),
        ]
        for cell_id in sorted(cells):
            promotion_command.extend(
                [
                    "--evidence", str(g0_root / "cells" / cell_id / "generation_g0_evidence.json"),
                    "--policy-manifest", str(args.g0_policy_root / cell_id / "policies.json"),
                    "--evidence", str(g1_root / "cells" / cell_id / "generation_g1_evidence.json"),
                    "--policy-manifest", str(g1_policy_root / cell_id / "policies.json"),
                ]
            )
        promotion_command.extend(["--output", str(promotion_path)])
        promotion = invoke(promotion_command, args.program_root / "promotion_controller.log")
        manifest["stages"]["promotion"] = promotion
        write_json(manifest_path, manifest)
        require_success("validation-only promotion selection", promotion)
        promotion_receipt = load_json(promotion_path)
        selected = promotion_receipt["selected"]

        manifest["status"] = "RUNNING_TOLUENE_POLICY_HOLDOUT"
        write_json(manifest_path, manifest)
        holdout = invoke(
            [
                args.python,
                str(args.code_root / "scripts" / "run_micro_mlip_policy_holdout.py"),
                "--holdout-id", f"{args.program_id}__g2_toluene",
                "--holdout-root", str(args.program_root / "g2_toluene"),
                "--promotion", str(promotion_path),
                "--policy-manifest", str(selected["policy_manifest"]),
                "--five-round-config", str(args.five_round_config),
                "--dataset-dir", str(args.dataset_dir),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--seed", str(args.holdout_seed),
                *common_gate,
            ],
            args.program_root / "g2_holdout_controller.log",
        )
        manifest["stages"]["g2_toluene"] = holdout
        write_json(manifest_path, manifest)
        require_success("toluene policy holdout", holdout)

        manifest["status"] = "COMPLETED_PENDING_CROSS_CELL_ANALYSIS"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "program_root": str(args.program_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

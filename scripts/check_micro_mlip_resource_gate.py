#!/usr/bin/env python3
"""Audit whether the five-round micro campaign may use the shared remote host.

The check is deliberately a receipt generator, not a launcher.  A blocked
receipt is useful evidence: the micro campaign controller will reject it and
therefore cannot accidentally compete with the formal MOF campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Mapping


TERMINAL_SUCCESS = {"COMPLETED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def process_state(pid: int) -> str:
    """Return launcher state; unreadable process state must stop admission."""

    if os.name == "nt":
        import psutil

        try:
            state = psutil.Process(pid).status()
        except psutil.NoSuchProcess:
            return "ABSENT"
        # AccessDenied and other inspection failures propagate, so they cannot
        # authorize a second campaign on an unverified machine.
        return "ZOMBIE" if state == psutil.STATUS_ZOMBIE else "ACTIVE"

    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    state = completed.stdout.strip()
    if completed.returncode != 0 or not state:
        return "ABSENT"
    return "ZOMBIE" if "Z" in state else "ACTIVE"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite resource-gate receipt: {args.output}")
    if not args.formal_state.is_file():
        raise FileNotFoundError(f"Formal campaign state is missing: {args.formal_state}")
    if not args.disk_path.exists():
        raise FileNotFoundError(f"Disk path does not exist: {args.disk_path}")

    formal = load_json(args.formal_state)
    launcher_state = process_state(args.formal_launcher_pid)
    disk = shutil.disk_usage(args.disk_path)
    free_gib = disk.free / (1024**3)
    formal_status = formal.get("status")
    blockers = []
    if launcher_state == "ACTIVE":
        blockers.append("formal_launcher_process_is_active")
    if formal_status not in TERMINAL_SUCCESS:
        blockers.append(f"formal_campaign_status_is_not_completed:{formal_status}")
    if not args.formal_done.is_file():
        blockers.append("formal_completion_sentinel_is_missing")
    if free_gib < args.minimum_free_gib:
        blockers.append(
            f"free_disk_below_minimum:{free_gib:.2f}GiB<{args.minimum_free_gib:.2f}GiB"
        )
    receipt = {
        "schema": "micro-mlip-shared-resource-gate-v1",
        "checked_at": utc_now(),
        "status": "ADMITTED" if not blockers else "BLOCKED",
        "formal_launcher_pid": args.formal_launcher_pid,
        "formal_launcher_state": launcher_state,
        "formal_state": str(args.formal_state),
        "formal_campaign_status": formal_status,
        "formal_done": str(args.formal_done),
        "formal_done_present": args.formal_done.is_file(),
        "disk_path": str(args.disk_path),
        "disk_total_gib": round(disk.total / (1024**3), 3),
        "disk_used_gib": round(disk.used / (1024**3), 3),
        "disk_free_gib": round(free_gib, 3),
        "minimum_free_gib": args.minimum_free_gib,
        "blockers": blockers,
        "admission_rule": "Formal campaign must be completed with its sentinel present, its launcher inactive, and free disk above the fixed floor.",
    }
    write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "blockers": blockers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

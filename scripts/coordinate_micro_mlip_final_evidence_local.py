#!/usr/bin/env python3
"""Wait for the primary P/D/T/W program, then run its matched final extension.

The active primary coordinator owns g0, g1, promotion, and scratch g2.  This
second local coordinator only observes its immutable status.  After the
primary program has closed, it deploys a newer immutable code snapshot to the
remote host, runs the serial model-method extension there, and downloads the
closed analysis/figure/report package.  It never sends a local Codex token or
outer-policy request to the remote host.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256


SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
RSYNC_SSH_COMMAND = "ssh -o BatchMode=yes -o ConnectTimeout=30"
PRIMARY_COMPLETION = "COMPLETED_PENDING_CROSS_CELL_ANALYSIS"


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
        raise RuntimeError(f"{label} failed; inspect {receipt.get('log_sha256')}")


def remote_command(host: str, command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    return invoke(["ssh", *SSH_OPTIONS, host, shlex.join(command)], log_path)


def ensure_remote_absent(host: str, path: Path, log_path: Path) -> None:
    receipt = remote_command(host, ["test", "!", "-e", str(path)], log_path)
    require_success(f"remote path absence check for {path}", receipt)


def sync_local_to_remote(host: str, local: Path, remote: Path, log_path: Path) -> None:
    ensure_remote_absent(host, remote, log_path.with_name(log_path.stem + "_absence.log"))
    receipt = invoke(["rsync", "-e", RSYNC_SSH_COMMAND, "-a", str(local), f"{host}:{remote}"], log_path)
    require_success(f"upload {local}", receipt)


def sync_remote_to_local(host: str, remote: Path, local: Path, log_path: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    receipt = invoke(["rsync", "-e", RSYNC_SSH_COMMAND, "-a", f"{host}:{remote}", str(local)], log_path)
    require_success(f"download {remote}", receipt)


def wait_for_primary(
    *,
    primary_manifest: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
    poll_seconds: int,
) -> None:
    attempt = 0
    while True:
        attempt += 1
        if not primary_manifest.is_file():
            snapshot: Dict[str, Any] = {"status": "MISSING", "path": str(primary_manifest)}
        else:
            primary = load_json(primary_manifest)
            snapshot = {
                "status": primary.get("status"),
                "updated_at": primary.get("updated_at"),
                "completed_at": primary.get("completed_at"),
                "failure": primary.get("failure"),
            }
        manifest["primary_poll"] = {"attempt": attempt, "snapshot": snapshot}
        write_json(manifest_path, manifest)
        status = snapshot.get("status")
        if status == PRIMARY_COMPLETION:
            return
        if isinstance(status, str) and status.startswith("FAILED"):
            raise RuntimeError("Primary P/D/T/W coordinator failed; final extension will not start")
        time.sleep(max(1, min(poll_seconds, 60)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--primary-coordinator-root", type=Path, required=True)
    parser.add_argument("--host", default="kozo-dev-yi")
    parser.add_argument("--local-code-root", type=Path, default=ROOT)
    parser.add_argument("--remote-code-root", type=Path, required=True)
    parser.add_argument("--primary-remote-code-root", type=Path, required=True)
    parser.add_argument("--primary-remote-program-root", type=Path, required=True)
    parser.add_argument("--remote-extension-root", type=Path, required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--remote-dataset-dir", type=Path, required=True)
    parser.add_argument("--remote-mace-runner", type=Path, required=True)
    parser.add_argument("--remote-mace-site", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, required=True)
    parser.add_argument("--replay-valid-file", type=Path, required=True)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.local_root.exists():
        raise FileExistsError("Final-evidence coordinator root already exists; programs are immutable")
    primary_manifest = args.primary_coordinator_root / "coordinator_manifest.json"
    args.local_root.mkdir(parents=True)
    manifest_path = args.local_root / "coordinator_manifest.json"
    log_root = args.local_root / "logs"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-final-evidence-local-coordinator-v1",
        "status": "WAITING_FOR_PRIMARY_PDTW_COMPLETION",
        "created_at": utc_now(),
        "program_id": args.program_id,
        "primary_coordinator_manifest": str(primary_manifest),
        "source_sha256": {"local_code_root": sha256(args.local_code_root / "scripts/run_micro_mlip_model_method_extension.py")},
        "remote": {
            "host": args.host,
            "code_root": str(args.remote_code_root),
            "primary_code_root": str(args.primary_remote_code_root),
            "primary_program_root": str(args.primary_remote_program_root),
            "extension_root": str(args.remote_extension_root),
        },
        "stages": {},
        "boundary": "This observer waits for the primary program's immutable completion status. It runs no training before that event and uses only a new remote code snapshot afterward.",
    }
    write_json(manifest_path, manifest)
    try:
        wait_for_primary(
            primary_manifest=primary_manifest,
            manifest=manifest,
            manifest_path=manifest_path,
            poll_seconds=args.poll_seconds,
        )
        manifest["status"] = "DEPLOYING_IMMUTABLE_FINAL_EVIDENCE_CODE"
        write_json(manifest_path, manifest)
        sync_local_to_remote(args.host, args.local_code_root, args.remote_code_root, log_root / "upload_code.log")
        ensure_remote_absent(args.host, args.remote_extension_root, log_root / "extension_root_absence.log")

        manifest["status"] = "RUNNING_REMOTE_MATCHED_MODEL_EXTENSION"
        write_json(manifest_path, manifest)
        remote = remote_command(
            args.host,
            [
                args.remote_python,
                str(args.remote_code_root / "scripts/run_micro_mlip_model_method_extension.py"),
                "--extension-id", args.program_id,
                "--extension-root", str(args.remote_extension_root),
                "--primary-program-root", str(args.primary_remote_program_root),
                "--primary-code-root", str(args.primary_remote_code_root),
                "--code-root", str(args.remote_code_root),
                "--dataset-dir", str(args.remote_dataset_dir),
                "--foundation-model", str(args.foundation_model),
                "--replay-file", str(args.replay_file),
                "--replay-valid-file", str(args.replay_valid_file),
                "--python", args.remote_python,
                "--mace-runner", str(args.remote_mace_runner),
                "--mace-site", str(args.remote_mace_site),
                "--device", "cpu",
                "--formal-launcher-pid", str(args.formal_launcher_pid),
                "--formal-state", str(args.formal_state),
                "--formal-done", str(args.formal_done),
                "--disk-path", str(args.disk_path),
                "--minimum-free-gib", str(args.minimum_free_gib),
            ],
            log_root / "remote_extension.log",
        )
        manifest["stages"]["remote_extension"] = remote
        write_json(manifest_path, manifest)
        require_success("remote matched model-method extension", remote)

        local_evidence = args.local_root / "evidence"
        sync_remote_to_local(args.host, args.remote_extension_root, local_evidence, log_root / "download_evidence.log")
        extension_manifest = local_evidence / "extension_manifest.json"
        completed = load_json(extension_manifest)
        if completed.get("status") != "COMPLETED_PENDING_MANUAL_FIGURE_QA":
            raise ValueError("Downloaded remote evidence package is not closed for manual figure QA")
        manifest["status"] = "COMPLETED_PENDING_MANUAL_FIGURE_QA"
        manifest["completed_at"] = utc_now()
        manifest["evidence_root"] = str(local_evidence)
        manifest["extension_manifest_sha256"] = sha256(extension_manifest)
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "evidence_root": str(local_evidence)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

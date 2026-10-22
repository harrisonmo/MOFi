#!/usr/bin/env python3
"""Coordinate the full P/D/T/W program across local outer calls and remote MACE.

The remote host deliberately runs only MACE and deterministic controllers.  The
authenticated local Codex CLI freezes g1 policies from the downloaded,
validation-only g0 receipts.  This avoids granting a remote process access to
the user's Codex configuration while preserving the test/holdout firewall.

The coordinator is designed to be launched as one durable local process.  It
waits quietly for the formal MOF sentinel, then executes immutable stages in
order.  Every remote MACE stage independently rechecks the shared-resource
gate, and failures stop the program without retries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256


SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
RSYNC_SSH_COMMAND = "ssh -o BatchMode=yes -o ConnectTimeout=30"


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


def invoke(command: Sequence[str], log_path: Path, *, input_text: str | None = None) -> Dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        input=input_text,
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


def sync_remote_to_local(host: str, remote: Path, local: Path, log_path: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    receipt = invoke(
        ["rsync", "-e", RSYNC_SSH_COMMAND, "-a", f"{host}:{remote}", str(local)],
        log_path,
    )
    require_success(f"download {remote}", receipt)


def sync_local_to_remote(host: str, local: Path, remote: Path, log_path: Path) -> None:
    ensure_remote_absent(host, remote, log_path.with_name(log_path.stem + "_absence.log"))
    receipt = invoke(
        ["rsync", "-e", RSYNC_SSH_COMMAND, "-a", str(local), f"{host}:{remote}"],
        log_path,
    )
    require_success(f"upload {local}", receipt)


def poll_formal_completion(
    *,
    host: str,
    remote_python: str,
    formal_launcher_pid: int,
    formal_state: Path,
    formal_done: Path,
    disk_path: Path,
    minimum_free_gib: float,
    poll_seconds: int,
    manifest: Dict[str, Any],
    manifest_path: Path,
    log_root: Path,
) -> None:
    """Wait only for the precise completion condition required by the resource gate."""

    check_source = (
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        f"state = json.loads(Path({str(formal_state)!r}).read_text())\n"
        f"pid = {formal_launcher_pid}\n"
        "try:\n"
        "    os.kill(pid, 0)\n"
        "    active = True\n"
        "except ProcessLookupError:\n"
        "    active = False\n"
        f"done = Path({str(formal_done)!r}).is_file()\n"
        f"free = shutil.disk_usage({str(disk_path)!r}).free / (1024 ** 3)\n"
        "result = {'formal_status': state.get('status'), 'launcher_active': active, 'done': done, 'free_gib': round(free, 3)}\n"
        "print(json.dumps(result, sort_keys=True))\n"
        "if str(state.get('status', '')).startswith('FAILED'):\n"
        "    raise SystemExit(2)\n"
        f"raise SystemExit(0 if state.get('status') == 'COMPLETED' and not active and done and free >= {minimum_free_gib!r} else 1)\n"
    )
    attempt = 0
    while True:
        attempt += 1
        receipt = remote_command(
            host,
            [remote_python, "-c", check_source],
            log_root / f"formal_poll_{attempt:05d}.log",
        )
        text = (log_root / f"formal_poll_{attempt:05d}.log").read_text(encoding="utf-8", errors="replace")
        snapshot: Any
        try:
            snapshot = json.loads(text.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            snapshot = {"unparsed": text[-1000:]}
        manifest["formal_poll"] = {"attempt": attempt, "receipt": receipt, "snapshot": snapshot}
        write_json(manifest_path, manifest)
        if receipt["exit_code"] == 0:
            return
        if receipt["exit_code"] == 2:
            raise RuntimeError("Formal MOF campaign failed; coordinator will not launch the micro program")
        # A bounded one-minute polling interval keeps the observer quiet and
        # never performs scientific work while the formal campaign is active.
        time.sleep(max(1, min(poll_seconds, 60)))


def cell_ids(registry_path: Path) -> list[str]:
    registry = load_json(registry_path)
    cells = [str(cell["cell_id"]) for cell in registry.get("cells", []) if cell.get("status") == "READY"]
    expected = registry.get("ready_cell_count")
    if not cells or len(cells) != expected:
        raise ValueError("Registry has an incomplete runnable cell set")
    return sorted(cells)


def remote_screen_command(
    *,
    remote_python: str,
    remote_code_root: Path,
    screen_id: str,
    screen_root: Path,
    generation: str,
    policy_root: Path,
    policy_audit: Path,
    dataset_dir: Path,
    mace_runner: Path,
    mace_site: Path,
    formal_launcher_pid: int,
    formal_state: Path,
    formal_done: Path,
    disk_path: Path,
    minimum_free_gib: float,
    seed: int,
) -> list[str]:
    return [
        remote_python,
        str(remote_code_root / "scripts" / "run_micro_mlip_five_axis_screen.py"),
        "--screen-id", screen_id,
        "--screen-root", str(screen_root),
        "--generation", generation,
        "--screen-config", str(remote_code_root / "configs" / "micro-mlip-pdtw-full-scratch-screen-v1.json"),
        "--five-round-config", str(remote_code_root / "configs" / "micro-mlip-rsi-five-round-v1.json"),
        "--registry", str(remote_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_scratch_registry_20261021.json"),
        "--policy-root", str(policy_root),
        "--policy-audit", str(policy_audit),
        "--dataset-dir", str(dataset_dir),
        "--code-root", str(remote_code_root),
        "--python", remote_python,
        "--mace-runner", str(mace_runner),
        "--mace-site", str(mace_site),
        "--device", "cpu",
        "--molecule", "ethanol",
        "--molecule", "malonaldehyde",
        "--seed", str(seed),
        "--formal-launcher-pid", str(formal_launcher_pid),
        "--formal-state", str(formal_state),
        "--formal-done", str(formal_done),
        "--disk-path", str(disk_path),
        "--minimum-free-gib", str(minimum_free_gib),
    ]


def fetch_generation_evidence(
    *,
    host: str,
    remote_screen_root: Path,
    local_screen_root: Path,
    cells: Iterable[str],
    generation: str,
    log_root: Path,
) -> None:
    local_screen_root.mkdir(parents=True, exist_ok=True)
    sync_remote_to_local(
        host,
        remote_screen_root / "screen_manifest.json",
        local_screen_root / "screen_manifest.json",
        log_root / f"fetch_{generation}_manifest.log",
    )
    for cell_id in cells:
        local_cell = local_screen_root / "cells" / cell_id
        local_cell.mkdir(parents=True, exist_ok=True)
        sync_remote_to_local(
            host,
            remote_screen_root / "cells" / cell_id / f"generation_{generation}_evidence.json",
            local_cell,
            log_root / f"fetch_{generation}_{cell_id}.log",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--host", default="kozo-dev-yi")
    parser.add_argument("--local-code-root", type=Path, default=ROOT)
    parser.add_argument("--remote-code-root", type=Path, required=True)
    parser.add_argument("--remote-program-root", type=Path, required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--remote-dataset-dir", type=Path, required=True)
    parser.add_argument("--remote-mace-runner", type=Path, required=True)
    parser.add_argument("--remote-mace-site", type=Path, required=True)
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--g0-seed", type=int, default=20261020)
    parser.add_argument("--g1-seed", type=int, default=20261021)
    parser.add_argument("--holdout-seed", type=int, default=20261022)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.local_root.exists():
        raise FileExistsError("Local coordinator root already exists; programs are immutable")
    ensure_remote_absent(args.host, args.remote_program_root, args.local_root.parent / "remote_root_absence.log")
    args.local_root.mkdir(parents=True)
    log_root = args.local_root / "logs"
    manifest_path = args.local_root / "coordinator_manifest.json"
    local_registry = args.local_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_scratch_registry_20261021.json"
    local_g0_policy_root = args.local_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_policies_g0_20261021"
    local_g0_audit = args.local_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_policy_audit_v2_20261021.json"
    remote_g0_policy_root = args.remote_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_policies_g0_20261021"
    remote_g0_audit = args.remote_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_policy_audit_v2_20261021.json"
    cells = cell_ids(local_registry)
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-full-pdtw-local-coordinator-v1",
        "status": "WAITING_FOR_FORMAL_MOF_COMPLETION",
        "created_at": utc_now(),
        "program_id": args.program_id,
        "remote": {"host": args.host, "code_root": str(args.remote_code_root), "program_root": str(args.remote_program_root)},
        "local_code_root": str(args.local_code_root),
        "cell_count": len(cells),
        "source_sha256": {
            "registry": sha256(local_registry),
            "g0_policy_audit": sha256(local_g0_audit),
        },
        "stages": {},
        "boundary": "The local Codex CLI sees only downloaded validation-only evidence. Remote MACE controllers independently gate resources and never receive local Codex credentials.",
    }
    write_json(manifest_path, manifest)

    try:
        poll_formal_completion(
            host=args.host,
            remote_python=args.remote_python,
            formal_launcher_pid=args.formal_launcher_pid,
            formal_state=args.formal_state,
            formal_done=args.formal_done,
            disk_path=args.disk_path,
            minimum_free_gib=args.minimum_free_gib,
            poll_seconds=args.poll_seconds,
            manifest=manifest,
            manifest_path=manifest_path,
            log_root=log_root,
        )
        manifest["status"] = "RUNNING_G0_REMOTE"
        write_json(manifest_path, manifest)
        g0_remote_root = args.remote_program_root / "g0"
        g0 = remote_command(
            args.host,
            remote_screen_command(
                remote_python=args.remote_python,
                remote_code_root=args.remote_code_root,
                screen_id=f"{args.program_id}__g0",
                screen_root=g0_remote_root,
                generation="g0",
                policy_root=remote_g0_policy_root,
                policy_audit=remote_g0_audit,
                dataset_dir=args.remote_dataset_dir,
                mace_runner=args.remote_mace_runner,
                mace_site=args.remote_mace_site,
                formal_launcher_pid=args.formal_launcher_pid,
                formal_state=args.formal_state,
                formal_done=args.formal_done,
                disk_path=args.disk_path,
                minimum_free_gib=args.minimum_free_gib,
                seed=args.g0_seed,
            ),
            log_root / "g0_remote.log",
        )
        manifest["stages"]["g0"] = g0
        write_json(manifest_path, manifest)
        require_success("remote g0 screen", g0)

        local_g0_root = args.local_root / "g0_evidence"
        fetch_generation_evidence(
            host=args.host,
            remote_screen_root=g0_remote_root,
            local_screen_root=local_g0_root,
            cells=cells,
            generation="g0",
            log_root=log_root,
        )
        manifest["status"] = "FREEZING_G1_POLICIES_LOCALLY"
        write_json(manifest_path, manifest)
        local_g1_policy_root = args.local_root / "g1_policies"
        local_g1_audit = args.local_root / "g1_policy_audit.json"
        g1_freeze = invoke(
            [
                sys.executable,
                str(args.local_code_root / "scripts" / "freeze_micro_mlip_five_axis_next_generation.py"),
                "--prior-screen-root", str(local_g0_root),
                "--prior-generation", "g0",
                "--generation", "g1",
                "--output-policy-root", str(local_g1_policy_root),
                "--output-audit", str(local_g1_audit),
                "--screen-config", str(args.local_code_root / "configs/micro-mlip-pdtw-full-scratch-screen-v1.json"),
                "--five-round-config", str(args.local_code_root / "configs/micro-mlip-rsi-five-round-v1.json"),
                "--registry", str(local_registry),
                "--prior-policy-root", str(local_g0_policy_root),
                "--prior-policy-audit", str(local_g0_audit),
                "--code-root", str(args.local_code_root),
                "--codex", args.codex,
                "--model", args.model,
                "--reasoning-effort", args.reasoning_effort,
                "--timeout-seconds", str(args.timeout_seconds),
            ],
            log_root / "g1_policy_freeze_local.log",
        )
        manifest["stages"]["g1_policy_freeze"] = g1_freeze
        write_json(manifest_path, manifest)
        require_success("local g1 policy freeze", g1_freeze)

        remote_g1_policy_root = args.remote_program_root / "g1_policies"
        remote_g1_audit = args.remote_program_root / "g1_policy_audit.json"
        sync_local_to_remote(args.host, local_g1_policy_root, remote_g1_policy_root, log_root / "upload_g1_policies.log")
        sync_local_to_remote(args.host, local_g1_audit, remote_g1_audit, log_root / "upload_g1_audit.log")
        manifest["status"] = "RUNNING_G1_REMOTE"
        write_json(manifest_path, manifest)
        g1_remote_root = args.remote_program_root / "g1"
        g1 = remote_command(
            args.host,
            remote_screen_command(
                remote_python=args.remote_python,
                remote_code_root=args.remote_code_root,
                screen_id=f"{args.program_id}__g1",
                screen_root=g1_remote_root,
                generation="g1",
                policy_root=remote_g1_policy_root,
                policy_audit=remote_g1_audit,
                dataset_dir=args.remote_dataset_dir,
                mace_runner=args.remote_mace_runner,
                mace_site=args.remote_mace_site,
                formal_launcher_pid=args.formal_launcher_pid,
                formal_state=args.formal_state,
                formal_done=args.formal_done,
                disk_path=args.disk_path,
                minimum_free_gib=args.minimum_free_gib,
                seed=args.g1_seed,
            ),
            log_root / "g1_remote.log",
        )
        manifest["stages"]["g1"] = g1
        write_json(manifest_path, manifest)
        require_success("remote g1 screen", g1)

        local_g1_root = args.local_root / "g1_evidence"
        fetch_generation_evidence(
            host=args.host,
            remote_screen_root=g1_remote_root,
            local_screen_root=local_g1_root,
            cells=cells,
            generation="g1",
            log_root=log_root,
        )
        manifest["status"] = "SELECTING_PROMOTION_LOCALLY"
        write_json(manifest_path, manifest)
        promotion = args.local_root / "promotion.json"
        selection_command = [
            sys.executable,
            str(args.local_code_root / "scripts" / "select_micro_mlip_pdtw_promotion.py"),
            "--five-round-config", str(args.local_code_root / "configs/micro-mlip-rsi-five-round-v1.json"),
        ]
        for cell_id in cells:
            selection_command.extend(
                [
                    "--evidence", str(local_g0_root / "cells" / cell_id / "generation_g0_evidence.json"),
                    "--policy-manifest", str(local_g0_policy_root / cell_id / "policies.json"),
                    "--evidence", str(local_g1_root / "cells" / cell_id / "generation_g1_evidence.json"),
                    "--policy-manifest", str(local_g1_policy_root / cell_id / "policies.json"),
                ]
            )
        selection_command.extend(["--output", str(promotion)])
        selection = invoke(selection_command, log_root / "promotion_local.log")
        manifest["stages"]["promotion"] = selection
        write_json(manifest_path, manifest)
        require_success("local validation-only promotion", selection)
        promotion_doc = load_json(promotion)
        selected = promotion_doc["selected"]
        generation, cell_id = selected["generation"], selected["factorial_cell_id"]
        if generation == "g0":
            remote_selected_policy = remote_g0_policy_root / cell_id / "policies.json"
        elif generation == "g1":
            remote_selected_policy = remote_g1_policy_root / cell_id / "policies.json"
        else:
            raise ValueError(f"Promotion selected an unknown generation: {generation}")
        remote_promotion = args.remote_program_root / "promotion.json"
        sync_local_to_remote(args.host, promotion, remote_promotion, log_root / "upload_promotion.log")

        manifest["status"] = "RUNNING_TOLUENE_POLICY_HOLDOUT_REMOTE"
        write_json(manifest_path, manifest)
        holdout = remote_command(
            args.host,
            [
                args.remote_python,
                str(args.remote_code_root / "scripts" / "run_micro_mlip_policy_holdout.py"),
                "--holdout-id", f"{args.program_id}__g2_toluene",
                "--holdout-root", str(args.remote_program_root / "g2_toluene"),
                "--promotion", str(remote_promotion),
                "--policy-manifest", str(remote_selected_policy),
                "--five-round-config", str(args.remote_code_root / "configs/micro-mlip-rsi-five-round-v1.json"),
                "--dataset-dir", str(args.remote_dataset_dir),
                "--code-root", str(args.remote_code_root),
                "--python", args.remote_python,
                "--mace-runner", str(args.remote_mace_runner),
                "--mace-site", str(args.remote_mace_site),
                "--device", "cpu",
                "--seed", str(args.holdout_seed),
                "--formal-launcher-pid", str(args.formal_launcher_pid),
                "--formal-state", str(args.formal_state),
                "--formal-done", str(args.formal_done),
                "--disk-path", str(args.disk_path),
                "--minimum-free-gib", str(args.minimum_free_gib),
            ],
            log_root / "g2_holdout_remote.log",
        )
        manifest["stages"]["g2_toluene"] = holdout
        write_json(manifest_path, manifest)
        require_success("remote toluene policy holdout", holdout)

        manifest["status"] = "COMPLETED_PENDING_CROSS_CELL_ANALYSIS"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "local_root": str(args.local_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Install and verify an isolated, pinned MACE feature sidecar.

The script never modifies the formal campaign's Python environment.  Instead,
it installs only the pinned ``mace-torch`` wheel, without dependencies, into a
fresh target directory and overlays that directory through ``PYTHONPATH`` for
an actual runner ``--help`` capability probe.  The caller must run this only
after its shared-resource gate has released the formal campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command: Sequence[str], log_path: Path, *, env: Mapping[str, str] | None = None) -> Dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=dict(env) if env is not None else None,
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
        raise RuntimeError(f"{label} failed; inspect log {receipt.get('log_sha256')}")


def overlay_environment(site_packages: Path) -> Dict[str, str]:
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(site_packages) if not inherited else str(site_packages) + os.pathsep + inherited
    return env


def contains_flag(help_text: str, flag: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_-])" + re.escape(flag) + r"(?![A-Za-z0-9_-])", help_text) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--mace-version", default="0.3.15")
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--required-flag", action="append", required=True)
    args = parser.parse_args()
    if args.sidecar_root.exists():
        raise FileExistsError("MACE sidecar root already exists; provisioning is immutable")
    if not args.runner.is_file():
        raise FileNotFoundError(f"MACE runner is missing: {args.runner}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.wheel_sha256):
        raise ValueError("The pinned MACE wheel SHA256 must be a 64-character lowercase hexadecimal digest")
    if len(set(args.required_flag)) != len(args.required_flag):
        raise ValueError("Each required MACE capability flag must be listed at most once")

    args.sidecar_root.mkdir(parents=True)
    manifest_path = args.sidecar_root / "sidecar_manifest.json"
    requirement_path = args.sidecar_root / "requirements.lock"
    site_packages = args.sidecar_root / "site-packages"
    requirement_path.write_text(
        f"mace-torch=={args.mace_version} --hash=sha256:{args.wheel_sha256}\n",
        encoding="utf-8",
    )
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-mace-sidecar-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "python": args.python,
        "runner": str(args.runner),
        "runner_sha256": sha256(args.runner),
        "mace_version": args.mace_version,
        "wheel_sha256": args.wheel_sha256,
        "requirements_lock": str(requirement_path),
        "requirements_lock_sha256": sha256(requirement_path),
        "site_packages": str(site_packages),
        "required_flags": list(args.required_flag),
        "boundary": "The sidecar is a new target directory. It does not modify the formal campaign virtual environment.",
    }
    write_json(manifest_path, manifest)
    try:
        install = invoke(
            [
                args.python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-compile",
                "--no-cache-dir",
                "--require-hashes",
                "--target",
                str(site_packages),
                "-r",
                str(requirement_path),
            ],
            args.sidecar_root / "pip_install.log",
        )
        manifest["install"] = install
        write_json(manifest_path, manifest)
        require_success("pinned MACE sidecar install", install)
        env = overlay_environment(site_packages)
        identity = invoke(
            [
                args.python,
                "-c",
                "import importlib.metadata as m, json; import mace; print(json.dumps({'version': m.version('mace-torch'), 'module_file': mace.__file__}))",
            ],
            args.sidecar_root / "identity.log",
            env=env,
        )
        manifest["identity"] = identity
        write_json(manifest_path, manifest)
        require_success("MACE sidecar import", identity)
        identity_lines = (args.sidecar_root / "identity.log").read_text(encoding="utf-8", errors="replace").splitlines()
        try:
            identity_doc = json.loads(identity_lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ValueError("MACE sidecar import did not emit its version identity record") from exc
        if identity_doc.get("version") != args.mace_version:
            raise ValueError("Imported MACE distribution version does not match the pinned sidecar version")
        if str(site_packages) not in str(identity_doc.get("module_file")):
            raise ValueError("MACE import did not resolve from the isolated sidecar target directory")
        help_receipt = invoke([args.python, str(args.runner), "--help"], args.sidecar_root / "runner_help.log", env=env)
        manifest["runner_help"] = help_receipt
        write_json(manifest_path, manifest)
        require_success("MACE sidecar runner help probe", help_receipt)
        help_text = (args.sidecar_root / "runner_help.log").read_text(encoding="utf-8", errors="replace")
        capabilities = {flag: contains_flag(help_text, flag) for flag in args.required_flag}
        manifest["capabilities"] = capabilities
        missing = [flag for flag, available in capabilities.items() if not available]
        if missing:
            raise ValueError(f"Pinned MACE sidecar runner lacks required capabilities: {missing}")
        manifest["status"] = "READY"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "sidecar_root": str(args.sidecar_root), "mace_version": args.mace_version}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

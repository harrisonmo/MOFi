"""Serve one science hub request, read as JSON from stdin, in the environment that hosts the engines.

The Windows controller (``rsi.science.wsl_hub.WslHub``) starts this module inside WSL through
``scripts/science-hub-wsl.sh`` for every catalog, budget or tool request. The hub session persists on
disk between requests and its binding check refuses a session whose registration changed. Engine
output is redirected to stderr so stdout carries exactly one JSON reply.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys

from rsi.science.hub import ScienceToolHub


COMMANDS = {"catalog", "budget", "call"}
PACKAGES = ("ase", "numpy", "gpaw", "raspalib", "psutil")


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "cp2k": shutil.which("cp2k.psmp"), "cp2k_data_dir": os.environ.get("CP2K_DATA_DIR"),
            "zeopp": os.environ.get("XTAL_ZEOPP"), "orca": os.environ.get("RSI_ORCA")}


def serve(request):
    if not isinstance(request, dict) or request.get("command") not in COMMANDS:
        raise ValueError(f"Request command must be one of {sorted(COMMANDS)}")
    hub = ScienceToolHub(request["config"], request["session"])
    if request["command"] == "catalog":
        result = hub.catalog()
    elif request["command"] == "budget":
        result = hub.budget_status()
    else:
        result = hub.call(request["tool"], request["arguments"])
    return {"ok": True, "result": result, "binding": hub.binding, "session_root": str(hub.session_root),
            "environment": environment()}


def main():
    request = json.loads(sys.stdin.read())
    sys.stdout.flush()
    reply_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        reply = serve(request)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        reply = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    finally:
        sys.stdout.flush()
        os.dup2(reply_fd, 1)
        os.close(reply_fd)
    sys.stdout.write(json.dumps(reply, allow_nan=False, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    main()

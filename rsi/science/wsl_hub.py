"""A Windows-side stand-in for ``ScienceToolHub`` that executes every request inside WSL.

GPAW, CP2K and the RASPA3 bindings run only in WSL, while the authenticated Codex CLI runs on Windows.
``WslHub`` keeps the agent loop and model calls on Windows and forwards each catalog, budget and tool
request to ``rsi.science.hub_bridge`` through ``scripts/science-hub-wsl.sh``. It exposes the same
``catalog``, ``budget_status``, ``call``, ``session_root`` and ``binding`` members the agent uses. The
session lives under a Windows folder that WSL reaches through ``/mnt/<drive>``, so the run directory
holds the hub ledger and evidence as usual; paths inside envelopes are WSL paths.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import re
import subprocess

from rsi.science.hub import ROOT


LAUNCHER = ROOT / "scripts/science-hub-wsl.sh"


def wsl_path(path):
    """Map an absolute Windows drive path to its /mnt/<drive> form inside WSL."""
    windows = PureWindowsPath(Path(path).resolve())
    if not re.fullmatch(r"[A-Za-z]:", windows.drive):
        raise ValueError(f"Only drive-letter paths are reachable from WSL: {windows}")
    return "/".join(["/mnt", windows.drive[0].lower(), *windows.parts[1:]])


class WslHub:
    """Owned by a controller: the distro, launcher and session are never model inputs."""

    def __init__(self, config_path, session_root, distro="Ubuntu", launcher=LAUNCHER, wsl="wsl.exe",
                 runner=subprocess.run):
        self.session_root = Path(session_root).resolve()
        self.distro = distro
        self.argv = [wsl, "-d", distro, "--exec", "bash", wsl_path(launcher), "-m", "rsi.science.hub_bridge"]
        self.request = {"config": wsl_path(config_path), "session": wsl_path(self.session_root)}
        self.runner = runner
        self.binding = None
        reply = self._request("catalog")  # Opens or creates the session now, as ScienceToolHub does.
        self.binding = reply["binding"]
        self.environment = reply["environment"]

    def _request(self, command, **fields):
        process = self.runner(self.argv, input=json.dumps({**self.request, "command": command, **fields}),
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            reply = json.loads(process.stdout)
        except ValueError:
            raise OSError(f"WSL hub bridge gave no reply (exit {process.returncode}): "
                          f"{(process.stderr or '')[-2000:]}") from None
        if not reply.get("ok"):
            error = reply.get("error") or {}
            raise ValueError(f"WSL hub refused the request: {error.get('type')}: {error.get('message')}")
        if self.binding is not None and reply["binding"] != self.binding:
            raise ValueError("WSL hub binding changed during the run")
        return reply

    def catalog(self):
        return self._request("catalog")["result"]

    def budget_status(self):
        return self._request("budget")["result"]

    def call(self, name, arguments):
        return self._request("call", tool=name, arguments=arguments)["result"]

"""WSL hub bridge: path mapping, request forwarding, refusals, and the bridge's single JSON reply."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("ase")

from rsi.science.hub_bridge import serve  # noqa: E402
from rsi.science.wsl_hub import WslHub, wsl_path  # noqa: E402
from test_science_hub import emt_relax, make_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeWsl:
    """Answer bridge requests in-process with the real hub, recording what the controller sent."""

    def __init__(self, translate, stdout=None, binding_override=None):
        self.translate = translate
        self.stdout = stdout
        self.binding_override = binding_override
        self.calls = []

    def __call__(self, argv, input, **kwargs):
        request = json.loads(input)
        self.calls.append((argv, request))
        if self.stdout is not None:
            return subprocess.CompletedProcess(argv, 1, self.stdout, "Traceback: boom")
        reply = serve({**request, "config": self.translate[request["config"]],
                       "session": self.translate[request["session"]]})
        if self.binding_override and len(self.calls) > 1:
            reply["binding"] = self.binding_override
        return subprocess.CompletedProcess(argv, 0, json.dumps(reply), "")


@pytest.mark.skipif(os.name != "nt", reason="Windows drive paths")
def test_wsl_path_maps_drive_paths():
    assert wsl_path(r"C:\Users\x y\LLM Materials\runs") == "/mnt/c/Users/x y/LLM Materials/runs"
    assert wsl_path(r"D:\a") == "/mnt/d/a"
    with pytest.raises(ValueError):
        wsl_path(r"\\server\share\x")


def fake_hub(tmp_path, **kwargs):
    config = make_config(tmp_path)
    session = tmp_path / "session"
    fake = FakeWsl({wsl_path(config): str(config), wsl_path(session): str(session)}, **kwargs)
    return WslHub(config, session, distro="Ubuntu", runner=fake), fake


@pytest.mark.skipif(os.name != "nt", reason="Windows drive paths")
def test_requests_are_forwarded_through_the_launcher(tmp_path):
    hub, fake = fake_hub(tmp_path)
    argv, request = fake.calls[0]
    assert argv[:5] == ["wsl.exe", "-d", "Ubuntu", "--exec", "bash"]
    assert argv[5].endswith("/scripts/science-hub-wsl.sh") and argv[-2:] == ["-m", "rsi.science.hub_bridge"]
    assert request["command"] == "catalog" and request["session"].startswith("/mnt/")
    assert "dft_calculation" in [t["function"]["name"] for t in hub.catalog()["tools"]]
    result = hub.call("dft_calculation", emt_relax())
    assert result["ok"] and result["call_id"] == "call-0001"
    sent = fake.calls[-1][1]
    assert sent["command"] == "call" and sent["tool"] == "dft_calculation" and sent["arguments"] == emt_relax()
    assert hub.budget_status()["calls_used"] == 1
    assert hub.environment["packages"]["ase"]


@pytest.mark.skipif(os.name != "nt", reason="Windows drive paths")
def test_bridge_failures_and_binding_changes_are_refused(tmp_path):
    with pytest.raises(OSError, match="no reply"):
        fake_hub(tmp_path, stdout="not json")
    hub, _ = fake_hub(tmp_path, binding_override={"changed": True})
    with pytest.raises(ValueError, match="binding changed"):
        hub.catalog()


def test_bridge_refuses_unknown_command(tmp_path):
    with pytest.raises(ValueError, match="command"):
        serve({"command": "delete", "config": str(make_config(tmp_path)), "session": str(tmp_path / "s")})


def test_bridge_process_writes_one_json_reply(tmp_path):
    config = make_config(tmp_path)
    request = {"command": "call", "config": str(config), "session": str(tmp_path / "session"),
               "tool": "dft_calculation", "arguments": emt_relax()}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "Crystal-Builder")])}
    process = subprocess.run([sys.executable, "-m", "rsi.science.hub_bridge"], input=json.dumps(request),
                             capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    reply = json.loads(process.stdout)
    assert process.returncode == 0 and reply["ok"] and reply["result"]["ok"]
    assert reply["result"]["data"]["converged"] is True
    bad = subprocess.run([sys.executable, "-m", "rsi.science.hub_bridge"],
                         input=json.dumps({**request, "config": str(tmp_path / "missing.json")}),
                         capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    assert json.loads(bad.stdout)["ok"] is False

"""Portable MCP contracts, argument validation and workspace-binding checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mofi_tools.providers import CatalogProvider, WorkspaceProvider


SOURCE = Path(__file__).resolve().parents[1] / "src"
FILTERS = dict(name_contains=None, method=None, data_type_contains=None,
               identifier=None, required_fields=None, usable_link_only=None)


def test_catalog_assets_are_pinned():
    provider = CatalogProvider()
    result = provider.call("mof_database_catalog", FILTERS)
    assert result["matches"] == 26
    assert "Unverified" in result["source"]["caveat"]


def test_invalid_filters_are_rejected_before_query():
    import jsonschema
    provider = CatalogProvider()
    with pytest.raises(jsonschema.ValidationError):
        provider.call("mof_database_catalog", {**FILTERS, "required_fields": ["unknown"]})
    with pytest.raises(jsonschema.ValidationError):
        provider.call("mof_database_catalog", {**FILTERS, "path": "/unexpected"})
    with pytest.raises(ValueError, match="Unknown tool"):
        provider.call("run_shell", {})


def test_cli_query_filters_without_science_dependencies():
    result = subprocess.run([sys.executable, "-m", "mofi_tools", "query", "--arguments",
                             json.dumps({"name_contains": "QMOF"})], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["matches"] == 1
    invalid = subprocess.run([sys.executable, "-m", "mofi_tools", "query", "--arguments", "[]"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0


def test_workspace_reply_binding_is_required_and_frozen(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    bridge = workspace / "rsi/science/hub_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("# Synthetic bridge path.\n", encoding="utf-8")
    (workspace / "hub.json").write_text("{}", encoding="utf-8")
    binding = {"files": {"config": "fixed"}}
    calls = []

    def run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        request = calls[-1][1]
        result = {"tools": [{"function": {"name": "bounded_tool", "description": "Synthetic tool",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]} if request["command"] == "catalog" else {"ok": True}
        return subprocess.CompletedProcess(command, 0, json.dumps({"ok": True, "result": result,
                                                                   "binding": binding}))

    monkeypatch.setattr(subprocess, "run", run)
    provider = WorkspaceProvider(workspace, sys.executable, "hub.json", "session")
    assert provider.call("bounded_tool", {})["ok"]
    assert all(command == [str(Path(sys.executable).resolve()), "-m", "rsi.science.hub_bridge"] for command, _ in calls)
    with pytest.raises(ValueError):
        provider.call("unknown", {})
    binding["files"]["config"] = "changed"
    with pytest.raises(RuntimeError, match="binding changed"):
        provider.call("bounded_tool", {})


def test_real_stdio_discovery_call_errors_and_resource():
    import anyio

    async def exercise():
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "mofi_tools", "serve"],
                                          env={"PYTHONPATH": str(SOURCE)})
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "mofi"
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {"mof_database_catalog", "mofi_tool_catalog"}
                result = await session.call_tool("mof_database_catalog", {**FILTERS, "name_contains": "QMOF"})
                assert not result.isError and result.structuredContent["matches"] == 1
                assert json.loads(result.content[0].text) == result.structuredContent
                invalid = await session.call_tool("mof_database_catalog", {"unknown": 1})
                assert invalid.isError
                unknown = await session.call_tool("run_shell", {})
                assert unknown.isError
                overview = await session.call_tool("mofi_tool_catalog", {})
                assert overview.structuredContent["profile"] == "catalog-only"
                resources = await session.list_resources()
                assert str(resources.resources[0].uri) == "mofi://catalog"
                resource = await session.read_resource("mofi://catalog")
                assert json.loads(resource.contents[0].text)["profile"] == "catalog-only"

    anyio.run(exercise)


@pytest.mark.parametrize("stdout,code", [
    ("not JSON", 0),
    ('{"ok":false,"error":{"message":"refused"}}', 0),
    ('{"ok":true,"result":{},"binding":null}', 0),
    ('{"ok":true,"result":null,"binding":{}}', 0),
    ("", 1),
])
def test_workspace_invalid_reply_fails_closed(tmp_path, monkeypatch, stdout, code):
    bridge = tmp_path / "rsi/science/hub_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("# Synthetic path.\n", encoding="utf-8")
    config = tmp_path / "hub.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, code, stdout))
    with pytest.raises(RuntimeError):
        WorkspaceProvider(tmp_path, sys.executable, config, tmp_path / "session")

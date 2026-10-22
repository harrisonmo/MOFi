"""No-Ollama ARIA MCP contracts; model responses and installation are synthetic."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import jsonschema

pytest.importorskip("mcp")
from rsi.aria_mcp import client, core
from rsi.hpc.core import read_json, sha256, write_json
from rsi.literature.grounding import ResearchToolHub

NULLABLE = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
MAP = {"anyOf": [{"type": "object", "additionalProperties": {"type": "string"}}, {"type": "null"}], "default": None}


def remote_tool(name):
    props = {"kg": copy.deepcopy(NULLABLE)}
    required = []
    if name != "diagnose":
        props["model"] = {"type": "string", "default": "qwen2:7b"}
        props["target_property"] = {"type": "string"}
        required.append("target_property")
    if name in {"predict", "design"}:
        props["mode"] = {"type": "string", "default": "aria"}
        field = "material" if name == "predict" else "target_material"
        props[field] = {"type": "string"}
        required.append(field)
        props["processing" if name == "predict" else "constraints"] = copy.deepcopy(MAP)
    if name == "explain":
        props.update(direction={"type": "string"}, material=copy.deepcopy(NULLABLE),
                     target_material=copy.deepcopy(NULLABLE), processing=copy.deepcopy(MAP))
        required.append("direction")
    return {"name": name, "description": "Synthetic ARIA tool", "inputSchema": {
        "type": "object", "properties": props, "required": required}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    package = tmp_path / "installed-aria"
    (package / "data").mkdir(parents=True)
    kg = package / "data/aria_2d_kg_demo.json"
    write_json(kg, {"causal_relationships": [], "description": "Synthetic graph, no scientific meaning"})
    identity = {"repository": client.REPOSITORY, "revision": client.REVISION,
                "version": "synthetic", "source_files": {"fake.py": "fixed"}, "mcp_version": "1.30.0"}
    monkeypatch.setattr(core, "installation", lambda: (package, copy.deepcopy(identity)))
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"schema_version": 1, "provider": "aria", "transport": "stdio",
                         "installation": identity, "tools": [remote_tool(name) for name in sorted(client.TOOLS)]})
    config = tmp_path / "config.json"
    write_json(config, {"schema_version": 1, "catalog_path": "catalog.json", "kg_path": None,
                       "model": "controller-model", "mode": "aria", "allow_external_search": False,
                       "allowed_tools": ["predict", "design", "diagnose"], "max_calls": 2,
                       "timeout_seconds": 5, "max_reserved_wall_seconds": 10, "max_response_bytes": 4096})
    calls = []
    async def exchange(action, **kwargs):
        assert action == "call"
        calls.append(copy.deepcopy(kwargs))
        data = ({"structure": {"num_nodes": 2, "num_edges": 1}} if kwargs["tool"] == "diagnose" else
                {"answer": {"hypothesis": "synthetic"}, "tier": 1, "confidence": 0.8,
                 "reasoning_type": "synthetic"})
        return {"isError": False, "structuredContent": data, "content": []}
    monkeypatch.setattr(core, "exchange", exchange)
    tools = core.AriaTools(config, tmp_path / "session")
    return SimpleNamespace(root=tmp_path, config=config, catalog=catalog, tools=tools,
                           calls=calls, kg=kg, identity=identity, package=package)


def arguments():
    return {"material": "MoS2", "target_property": "carrier mobility", "processing": {"method": "CVD"}}


def test_catalog_hides_controller_fields(setup):
    catalog = setup.tools.catalog()
    assert len(catalog["tools"]) == 3
    for tool in catalog["tools"]:
        assert not client.CONTROLLER_FIELDS & set(tool["function"]["parameters"]["properties"])
        assert tool["function"]["parameters"]["additionalProperties"] is False
    assert "2D demo" in " ".join(catalog["guidance"])


def test_forward_target_graph_snapshot_and_controller_settings(setup):
    original = arguments()
    result = setup.tools.call("aria__predict", original)
    assert result["ok"] and result["scientific_validation"] is False
    sent = setup.calls[0]["arguments"]
    assert sent["model"] == "controller-model" and sent["mode"] == "aria"
    assert sent["processing"]["target_property"] == original["target_property"]
    assert sent["kg"] != str(setup.kg) and Path(sent["kg"]).read_bytes() == setup.kg.read_bytes()
    assert "target_property" not in original["processing"]
    request = read_json(setup.root / "session/call-0001/request.json")
    assert request["arguments"] == original and request["effective_arguments"] == sent


@pytest.mark.parametrize("extra", [{"kg": "/tmp/evil.json"}, {"model": "other"}, {"mode": "aria_full"}])
def test_reserved_fields_refused_without_spending(setup, extra):
    with pytest.raises(jsonschema.ValidationError):
        setup.tools.call("aria__predict", dict(arguments(), **extra))
    assert not setup.calls and setup.tools.budget()["reserved_calls"] == 0


@pytest.mark.parametrize("name, args", [
    ("aria__missing", {}), ("aria__explain", {}), ("aria__predict", {}),
    ("aria__predict", {"material": 1, "target_property": "x"}),
    ("aria__predict", {"material": "x", "target_property": "y", "processing": {"x": 4}}),
    ("aria__diagnose", {"extra": True}), ("aria__predict", {"material": "x" * 65537, "target_property": "y"})
])
def test_invalid_arguments_reserve_nothing(setup, name, args):
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        setup.tools.call(name, args)
    assert not setup.calls and setup.tools.budget()["reserved_calls"] == 0


@pytest.mark.parametrize("change", [
    {"allowed_tools": ["explain"]}, {"mode": "aria_full"}, {"mode": "aria_search"},
    {"schema_version": True}, {"max_calls": True}, {"timeout_seconds": 0},
    {"allowed_tools": ["predict", "predict"]}, {"model": "--bad"}, {"kg_path": "missing.json"}
])
def test_invalid_config_refused(setup, change):
    config = read_json(setup.config);config.update(change);write_json(setup.config, config)
    with pytest.raises(ValueError):
        core.AriaTools(setup.config, setup.root / "new-session")


def test_explain_explicit_opt_in_and_direction(setup):
    config = read_json(setup.config)
    config.update(allowed_tools=["explain"], allow_external_search=True)
    write_json(setup.config, config)
    tools = core.AriaTools(setup.config, setup.root / "explain-session")
    with pytest.raises(ValueError, match="direction"):
        tools.call("aria__explain", {"direction": "bad", "target_property": "x"})
    with pytest.raises(ValueError, match="material"):
        tools.call("aria__explain", {"direction": "forward", "target_property": "x"})
    assert tools.call("aria__explain", dict(arguments(), direction="forward"))["ok"]
    assert setup.calls[-1]["arguments"]["processing"]["target_property"] == "carrier mobility"


def test_design_dispatch(setup):
    assert setup.tools.call("aria__design", {"target_material": "MoS2", "target_property": "mobility"})["ok"]
    sent = setup.calls[0]
    assert sent["tool"] == "design" and sent["arguments"]["target_material"] == "MoS2"


@pytest.mark.parametrize("constraint", ["calls", "reserved_time"])
def test_persistent_budgets(setup, constraint):
    config = read_json(setup.config)
    config["max_calls"] = 1 if constraint == "calls" else 10
    config["max_reserved_wall_seconds"] = 5 if constraint == "reserved_time" else 50
    write_json(setup.config, config)
    tools = core.AriaTools(setup.config, setup.root / "budget-session")
    assert tools.call("aria__diagnose", {})["ok"]
    reopened = core.AriaTools(setup.config, setup.root / "budget-session")
    with pytest.raises(ValueError, match="exhausted"):
        reopened.call("aria__diagnose", {})
    assert len(setup.calls) == 1 and reopened.budget()["reserved_wall_seconds"] == 5


@pytest.mark.parametrize("failure, expected", [
    ("mcp", "mcp_error"), ("application", "aria_error"), ("exception", "transport_error"),
    ("timeout", "timeout"), ("oversized", "response_limit"), ("invalid", "invalid_output"),
    ("cancelled", "interrupted")
])
def test_failure_accounting(setup, monkeypatch, failure, expected):
    async def exchange(*_, **kwargs):
        if failure == "exception": raise RuntimeError("secret-marker")
        if failure == "timeout": raise TimeoutError()
        if failure == "cancelled": raise asyncio.CancelledError()
        if failure == "oversized": return {"structuredContent": {"x": "a" * 5000}}
        if failure == "mcp": return {"isError": True, "content": []}
        if failure == "application": return {"isError": False, "structuredContent": {"error": "Ollama unavailable"}}
        return {"structuredContent": {"answer": {}, "tier": 9, "confidence": 1, "reasoning_type": "bad"}}
    monkeypatch.setattr(core, "exchange", exchange)
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError): setup.tools.call("aria__predict", arguments())
    else:
        result = setup.tools.call("aria__predict", arguments())
        assert not result["ok"] and result["status"] == expected
    saved = read_json(setup.root / "session/call-0001/result.json")
    assert saved["status"] == expected and "secret-marker" not in json.dumps(saved)
    assert setup.tools.budget()["reserved_calls"] == 1
    with pytest.raises(ValueError, match="completed"):
        setup.tools.evidence(["call-0001"])


@pytest.mark.parametrize("field, value", [("tier", True), ("confidence", True), ("confidence", -1), ("confidence", 1.1)])
def test_invalid_inference_metadata(setup, monkeypatch, field, value):
    async def exchange(*_, **kwargs):
        data={"answer": {}, "tier": 1, "confidence": 0.8, "reasoning_type": "synthetic"};data[field]=value
        return {"structuredContent": data}
    monkeypatch.setattr(core, "exchange", exchange)
    assert setup.tools.call("aria__predict", arguments())["status"] == "invalid_output"


@pytest.mark.parametrize("changed", ["config", "catalog", "kg", "installation"])
def test_binding_changes_refused(setup, changed):
    if changed == "installation":setup.identity["version"] = "changed"
    else:
        path=getattr(setup, changed);path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="changed"):
        setup.tools.budget()


@pytest.mark.parametrize("changed", ["request.json", "result.json", "knowledge-graph.json"])
def test_evidence_integrity_and_budget(setup, changed):
    assert setup.tools.call("aria__diagnose", {})["ok"]
    packet = setup.tools.evidence(["call-0001"])
    assert packet["trust"] == "unverified_reasoning" and packet["scientific_validation"] is False
    with pytest.raises(ValueError, match="byte budget"):
        setup.tools.evidence(["call-0001"], max_bytes=1)
    path = setup.root / "session/call-0001" / changed
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="changed"):
        setup.tools.evidence(["call-0001"])


def test_stale_reservation_not_refunded(setup, monkeypatch):
    setup.tools.call("aria__diagnose", {})
    ledger_path = setup.root / "session/aria-ledger.json"
    ledger=read_json(ledger_path);ledger["calls"][0]["status"]="running";write_json(ledger_path, ledger)
    monkeypatch.setattr(core, "pid_alive", lambda pid: False)
    reopened=core.AriaTools(setup.config, setup.root / "session")
    assert read_json(ledger_path)["calls"][0]["status"]=="interrupted"
    assert reopened.budget()["reserved_wall_seconds"] == 5


def test_async_controller_and_routing(setup):
    async def scenario():
        with pytest.raises(RuntimeError, match="acall"):
            setup.tools.call("aria__diagnose", {})
        return await setup.tools.acall("aria__diagnose", {})
    assert asyncio.run(scenario())["ok"]
    science=SimpleNamespace(catalog=lambda: {"tools": [{"function": {"name": "dft"}}]}, call=lambda *args:"science")
    hub=ResearchToolHub(science, aria=setup.tools)
    assert len(hub.catalog()["tools"])==4 and hub.call("dft", {})=="science"
    assert hub.call("aria__diagnose", {})["ok"]
    with pytest.raises(ValueError, match="not configured"):
        ResearchToolHub(science).call("aria__diagnose", {})


@pytest.mark.parametrize("raw, status", [
    ({"isError": True}, "mcp_error"), ({"structuredContent": {"error": "failure"}}, "aria_error"),
    ({"content": [{"type": "text", "text": '{"structure":{}}'}]}, "completed"),
    ({"content": [{"type": "text", "text": "not-json"}]}, "invalid_output"),
    ({"structuredContent": []}, "invalid_output"), ({"content": []}, "invalid_output"),
    ({"structuredContent": {"result": {"error": "Ollama unavailable"}}}, "aria_error"),
    ({"structuredContent": {"result": {"structure": {"num_nodes": 2}}}}, "completed"),
    ({"structuredContent": {"result": []}}, "invalid_output"),
    ({"content": [{"type": "text", "text": '{"result":{"error":"missing model"}}'}]}, "aria_error")
])
def test_payload_variants(raw, status):
    assert client.payload(raw)[1] == status


def test_actual_sdk_stdio_without_ollama(tmp_path, monkeypatch):
    """Actual SDK framing with a synthetic server, no ARIA or inference runtime."""
    import mcp
    server=tmp_path/'synthetic_server.py'
    server.write_text('''from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
mcp=FastMCP("synthetic-aria")
@mcp.tool()
def diagnose(kg: str="") -> Dict[str, Any]:
    return {"structure": {"num_nodes": 2, "num_edges": 1}, "kg": kg}
if __name__=="__main__":mcp.run(transport="stdio")
''')
    original=mcp.StdioServerParameters
    def parameters(**kwargs):
        kwargs['args']=[str(server)]
        return original(**kwargs)
    monkeypatch.setattr(mcp, "StdioServerParameters", parameters)
    catalog=asyncio.run(client.exchange('discover', timeout=10, stderr_path=tmp_path/'discover.stderr'))
    assert catalog['tools'][0]['name']=='diagnose'
    raw=asyncio.run(client.exchange('call', timeout=10, stderr_path=tmp_path/'call.stderr',tool='diagnose',arguments={'kg':'synthetic'}))
    assert set(raw['structuredContent']) == {'result'}
    data,status=client.payload(raw)
    assert status=='completed' and data['structure']['num_nodes']==2


def test_wrapped_reasoning_and_application_error(setup, monkeypatch):
    async def success(*_, **kwargs):
        return {"isError": False, "structuredContent": {"result": {
            "answer": {"hypothesis": "synthetic"}, "tier": 1,
            "confidence": 0.75, "reasoning_type": "direct_path"}}}
    monkeypatch.setattr(core, "exchange", success)
    result = setup.tools.call("aria__predict", arguments())
    assert result["ok"] and result["data"]["tier"] == 1
    assert "result" in result["raw"]["structuredContent"]
    async def failure(*_, **kwargs):
        return {"isError": False, "structuredContent": {"result": {"error": "Ollama missing", "hint": "install later"}}}
    monkeypatch.setattr(core, "exchange", failure)
    result = setup.tools.call("aria__predict", arguments())
    assert not result["ok"] and result["status"] == "aria_error"
    assert result["data"]["error"] == "Ollama missing"
    assert setup.tools.budget()["reserved_calls"] == 2

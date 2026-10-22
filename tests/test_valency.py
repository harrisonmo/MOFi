"""Offline contracts for Valency tools, grounding, OAuth storage and MCP transport."""

import asyncio
import copy
import jsonschema
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
import httpx
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from rsi.hpc.core import read_json, sha256, write_json
from rsi.literature import client, valency
from rsi.literature.grounding import GroundedBackend, ResearchToolHub

TOOL = {"name": "search_papers", "description": "Synthetic literature search",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                        "required": ["query"], "additionalProperties": False},
        "annotations": {"readOnlyHint": True}}
RAW = {"content": [{"type": "text", "text": "Synthetic paper: https://example.org/paper"}],
       "structuredContent": {"papers": [{"title": "Synthetic methane research",
                                          "url": "https://example.org/paper"}]}, "isError": False}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"schema_version": 1, "endpoint": client.ENDPOINT, "tools": [TOOL]})
    config = tmp_path / "config.json"
    write_json(config, {"schema_version": 1, "catalog_path": "catalog.json",
                       "allowed_tools": ["search_papers"], "max_calls": 2,
                       "timeout_seconds": 5, "max_response_bytes": 4096})
    calls = []

    async def invoke(name, arguments, auth, timeout):
        calls.append((name, arguments, timeout))
        return copy.deepcopy(RAW)

    monkeypatch.setattr(valency, "invoke", invoke)
    tools = valency.ValencyTools(config, tmp_path / "session", auth_path=tmp_path / "oauth.json")
    return SimpleNamespace(config=config, catalog=catalog, tools=tools, calls=calls, root=tmp_path)


def test_catalog_and_successful_evidence(setup):
    function = setup.tools.catalog()["tools"][0]["function"]
    assert function["name"] == "valency__search_papers" and function["strict"] is False
    result = setup.tools.call(function["name"], {"query": "methane MOFs"})
    assert result["ok"] and result["data"] == RAW
    assert setup.calls == [("search_papers", {"query": "methane MOFs"}, 5)]
    assert setup.tools.budget()["reserved_calls"] == 1
    packet = setup.tools.evidence(["call-0001"])
    assert packet["records"][0]["request"]["arguments"] == {"query": "methane MOFs"}
    assert packet["trust"] == "untrusted_external_literature"


@pytest.mark.parametrize("name, arguments", [
    ("not_allowed", {"query": "MOFs"}), ("valency__search_papers", {}),
    ("valency__search_papers", {"query": 4}),
    ("valency__search_papers", {"query": "x", "extra": True}),
    ("valency__search_papers", {"query": "x" * 65537}),
])
def test_invalid_calls_spend_nothing(setup, name, arguments):
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        setup.tools.call(name, arguments)
    assert not setup.calls and setup.tools.budget()["reserved_calls"] == 0


def test_reopened_budget_is_not_reset(setup):
    for _ in range(2):
        setup.tools.call("valency__search_papers", {"query": "x"})
    reopened = valency.ValencyTools(setup.config, setup.root / "session")
    with pytest.raises(ValueError, match="exhausted"):
        reopened.call("valency__search_papers", {"query": "x"})
    assert len(setup.calls) == 2


@pytest.mark.parametrize("failure", ["remote", "exception", "oversized", "cancelled"])
def test_failed_calls_are_charged_and_not_exported(setup, monkeypatch, failure):
    async def invoke(*_):
        if failure == "exception":
            raise RuntimeError("secret-access-token")
        if failure == "cancelled":
            raise asyncio.CancelledError()
        if failure == "remote":
            return {"isError": True, "content": []}
        return {"content": [{"type": "text", "text": "x" * 5000}]}

    monkeypatch.setattr(valency, "invoke", invoke)
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            setup.tools.call("valency__search_papers", {"query": "x"})
    else:
        result = setup.tools.call("valency__search_papers", {"query": "x"})
        assert not result["ok"]
    assert setup.tools.budget()["reserved_calls"] == 1
    with pytest.raises(ValueError, match="successful"):
        setup.tools.evidence(["call-0001"])
    saved = (setup.root / "session/call-0001/result.json").read_text()
    assert "secret-access-token" not in saved


@pytest.mark.parametrize("file", ["config", "catalog"])
def test_changed_registration_refuses_existing_session(setup, file):
    path = getattr(setup, file)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        setup.tools.budget()
    with pytest.raises(ValueError, match="changed"):
        valency.ValencyTools(setup.config, setup.root / "session")


def test_evidence_hash_and_context_size_enforced(setup):
    setup.tools.call("valency__search_papers", {"query": "x"})
    with pytest.raises(ValueError, match="context budget"):
        setup.tools.evidence(["call-0001"], max_bytes=10)
    target = setup.root / "session/call-0001/result.json"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        setup.tools.evidence(["call-0001"])


def test_write_tool_not_exposed(setup):
    catalog = read_json(setup.catalog)
    catalog["tools"][0]["annotations"]["readOnlyHint"] = False
    write_json(setup.catalog, catalog)
    with pytest.raises(ValueError, match="changing remote state"):
        valency.ValencyTools(setup.config, setup.root / "new-session")


def test_empty_allowlist_exposes_nothing(setup):
    config = read_json(setup.config)
    config["allowed_tools"] = []
    write_json(setup.config, config)
    tools = valency.ValencyTools(setup.config, setup.root / "empty-session")
    assert tools.catalog()["tools"] == []


def test_sync_call_rejected_in_async_controller(setup):
    async def scenario():
        with pytest.raises(RuntimeError, match="acall"):
            setup.tools.call("valency__search_papers", {"query": "x"})
        return await setup.tools.acall("valency__search_papers", {"query": "x"})
    assert asyncio.run(scenario())["ok"]


class FakeBackend:
    def __init__(self):
        self.prompts = []

    def identity(self):
        return {"kind": "synthetic"}

    def run(self, prompt, schema, output_dir):
        self.prompts.append(prompt)
        Path(output_dir).mkdir(parents=True)
        return {"status": "ok", "response": {"choice": "A"}, "usage": None}


def packet_file(setup):
    setup.tools.call("valency__search_papers", {"query": "x"})
    path = setup.root / "packet.json"
    write_json(path, setup.tools.evidence(["call-0001"]))
    return path


def test_grounding_wraps_backend_and_freezes_context(setup):
    path = packet_file(setup)
    backend = FakeBackend()
    wrapper = GroundedBackend(backend, path, sha256(path))
    result = wrapper.run("Original task", {}, setup.root / "model-call")
    assert result["status"] == "ok" and result["literature"]["packet_sha256"] == sha256(path)
    assert "Original task" in backend.prompts[0] and "untrusted external literature" in backend.prompts[0]
    assert "https://example.org/paper" in backend.prompts[0]
    assert read_json(setup.root / "model-call/literature-context.json")["records"]
    assert wrapper.identity()["backend"] == {"kind": "synthetic"}
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        wrapper.run("x", {}, setup.root / "second-model-call")
    assert len(backend.prompts) == 1


def test_grounding_rejects_oversized_packet(setup):
    path = packet_file(setup)
    with pytest.raises(ValueError, match="context budget"):
        GroundedBackend(FakeBackend(), path, sha256(path), max_context_bytes=1024)


def test_factory_opt_in_and_default(setup, monkeypatch):
    from rsi.backends import factory
    backend = FakeBackend()
    monkeypatch.setattr(factory, "_create_backend", lambda config: backend)
    assert factory.create_backend({}) is backend
    path = packet_file(setup)
    wrapped = factory.create_backend({"literature": {"evidence_path": str(path),
                                                   "evidence_sha256": sha256(path)}})
    assert isinstance(wrapped, GroundedBackend)
    with pytest.raises(ValueError):
        factory.create_backend({"literature": {"evidence_path": str(path)}})


def test_combined_dispatch(setup):
    science = SimpleNamespace(catalog=lambda: {"tools": [{"type": "function", "function": {"name": "dft"}}]},
                              call=lambda name, args: {"science": name})
    hub = ResearchToolHub(science, setup.tools)
    assert len(hub.catalog()["tools"]) == 2
    assert hub.call("dft", {}) == {"science": "dft"}
    assert hub.call("valency__search_papers", {"query": "x"})["ok"]


def test_oauth_storage_roundtrip_and_private_mode(tmp_path):
    storage = client.OAuthFiles(tmp_path / "auth/oauth.json")
    async def scenario():
        assert await storage.get_tokens() is None
        await storage.set_client_info(OAuthClientInformationFull(client_id="synthetic-client", redirect_uris=[client.REDIRECT]))
        await storage.set_tokens(OAuthToken(access_token="synthetic-token", token_type="Bearer", expires_in=3600))
        assert (await storage.get_tokens()).access_token == "synthetic-token"
        assert (await storage.get_client_info()).client_id == "synthetic-client"
    asyncio.run(scenario())
    if os.name == "posix":
        assert storage.path.stat().st_mode & 0o777 == 0o600
    data = read_json(storage.path)
    data["endpoint"] = "https://example.invalid"
    write_json(storage.path, data)
    with pytest.raises(ValueError, match="binding"):
        asyncio.run(storage.get_tokens())


def test_unattended_missing_login_opens_no_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(client.webbrowser, "open", lambda _: pytest.fail("Unexpected browser"))
    async def scenario():
        async with client.session(tmp_path / "missing.json"):
            pytest.fail("Should not connect")
    with pytest.raises(client.LoginRequired):
        asyncio.run(scenario())


def test_sdk_streamable_http_wire(tmp_path, monkeypatch):
    """Actual SDK initialization, discovery and tool call, mocked HTTP only."""
    real_client = httpx.AsyncClient
    methods = []
    def handle(request):
        if request.method == "DELETE":
            return httpx.Response(200)
        assert request.headers.get("Authorization") == "Bearer synthetic-token"
        message = json.loads(request.content)
        methods.append(message["method"])
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            data = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "synthetic-bond", "version": "1"}}
        elif method == "tools/list":
            data = {"tools": [TOOL]}
        elif method == "tools/call":
            data = RAW
        else:
            pytest.fail(f"Unexpected method {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": data})
    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    auth = tmp_path / "wire-auth.json"
    storage = client.OAuthFiles(auth)
    asyncio.run(storage.set_tokens(OAuthToken(access_token="synthetic-token", token_type="Bearer", expires_in=3600)))
    discovered = asyncio.run(client.discover(auth))
    assert discovered["tools"][0]["name"] == TOOL["name"]
    assert asyncio.run(client.invoke(TOOL["name"], {"query": "MOFs"}, auth, 5))["isError"] is False
    assert "initialize" in methods and "tools/list" in methods and "tools/call" in methods

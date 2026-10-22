"""Bounded ARIA tool calls with frozen inputs and persistent attempt accounting."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import jsonschema

from rsi.hpc.core import locked, pid_alive, read_json, sha256, write_json
from .client import CONTROLLER_FIELDS, REVISION, TOOLS, exchange, installation, payload

PREFIX = "aria__"
ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def settings(path):
    path = Path(path).resolve()
    config = read_json(path)
    fields = {"schema_version", "catalog_path", "kg_path", "model", "mode", "allow_external_search",
              "allowed_tools", "max_calls", "timeout_seconds", "max_reserved_wall_seconds", "max_response_bytes"}
    if not isinstance(config, dict) or set(config) != fields or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Invalid ARIA config")
    for key, low, high in (("max_calls", 1, 10000), ("timeout_seconds", 1, 600),
                           ("max_reserved_wall_seconds", 1, 6000000), ("max_response_bytes", 1024, 4194304)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if not isinstance(config["model"], str) or not config["model"] or config["model"].startswith("-"):
        raise ValueError("model must be a controller-selected Ollama model name")
    if config["mode"] not in {"baseline", "naive_kg", "aria", "aria_search", "aria_full"}:
        raise ValueError("Unsupported ARIA reasoning mode")
    if type(config["allow_external_search"]) is not bool:
        raise ValueError("allow_external_search must be boolean")
    allowed = config["allowed_tools"]
    if (not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed)
            or len(allowed) != len(set(allowed)) or set(allowed) - TOOLS):
        raise ValueError("Invalid ARIA tool allowlist")
    if not config["allow_external_search"] and (
            "explain" in allowed or config["mode"] in {"aria_search", "aria_full"}):
        raise ValueError("explain and search modes require allow_external_search=true")
    if not isinstance(config["catalog_path"], str) or not config["catalog_path"]:
        raise ValueError("catalog_path must be a path string")
    root, identity = installation()
    kg = config["kg_path"]
    if kg is not None and (not isinstance(kg, str) or not kg):
        raise ValueError("kg_path must be a path string or null for the bundled 2D demo")
    kg = (path.parent / kg).resolve() if kg is not None else root / "data/aria_2d_kg_demo.json"
    if not kg.is_file():
        raise ValueError("Configured ARIA knowledge graph does not exist")
    return path, config, kg, identity


def _public(tool):
    schema = copy.deepcopy(tool["inputSchema"])
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("ARIA tool needs an object input schema")
    for key in CONTROLLER_FIELDS:
        schema["properties"].pop(key, None)
    schema["required"] = [key for key in schema.get("required", []) if key not in CONTROLLER_FIELDS]
    schema["additionalProperties"] = False
    jsonschema.validators.validator_for(schema).check_schema(schema)
    return {"type": "function", "function": {
        "name": PREFIX + tool["name"], "strict": False, "parameters": schema,
        "description": tool.get("description", "") + " ARIA reasoning is advisory; its confidence "
        "is not a validated probability or scientific measurement. Model, graph and mode are controller-owned."}}


def discover(config_path, output_path):
    """Explicit setup: start the pinned server, list tools, and save their schemas."""
    _, config, _, identity = settings(config_path)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh catalog output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = asyncio.run(exchange("discover", timeout=min(config["timeout_seconds"], 60),
                               stderr_path=output.with_suffix(".stderr.log")))
    names = [tool["name"] for tool in raw["tools"]]
    if len(names) != len(set(names)) or not TOOLS.issubset(names):
        raise ValueError("The installed server does not expose all four expected ARIA tools")
    for tool in raw["tools"]:
        if tool["name"] in TOOLS:
            _public(tool)
    catalog = {"schema_version": 1, "provider": "aria", "transport": "stdio",
               "retrieved_at": now(), "installation": identity, **raw}
    write_json(output, catalog)
    return catalog


class AriaTools:
    """One controller-owned registry, separate from science and literature ledgers."""

    def __init__(self, config_path, session_root):
        self.config_path, self.config, self.kg, self.installation = settings(config_path)
        self.catalog_path = (self.config_path.parent / self.config["catalog_path"]).resolve()
        catalog = read_json(self.catalog_path)
        if (catalog.get("schema_version") != 1 or catalog.get("provider") != "aria"
                or catalog.get("transport") != "stdio" or catalog.get("installation") != self.installation):
            raise ValueError("Discover a catalog for this pinned ARIA installation first")
        remote = {tool["name"]: tool for tool in catalog["tools"]}
        if len(remote) != len(catalog["tools"]) or set(self.config["allowed_tools"]) - set(remote):
            raise ValueError("Unknown or duplicate ARIA tools")
        self.remote = {PREFIX + name: remote[name] for name in self.config["allowed_tools"]}
        self.tools = {name: _public(tool) for name, tool in self.remote.items()}
        self.session_root = Path(session_root).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.session_root / "aria-ledger.json"
        self.binding = self._binding()
        with locked(self.ledger_path):
            if self.ledger_path.exists():
                ledger = self._ledger()
                for entry in ledger["calls"]:
                    if entry["status"] == "running" and not pid_alive(entry["controller_pid"]):
                        entry.update(status="interrupted", finished_at=now())
            else:
                ledger = {"schema_version": 1, "binding": self.binding, "calls": []}
            write_json(self.ledger_path, ledger)

    def _binding(self):
        _, identity = installation()
        paths = [self.config_path, self.catalog_path, self.kg, ROOT / "rsi/hpc/core.py",
                 ROOT / "rsi/literature/grounding.py"] + sorted(Path(__file__).parent.glob("*.py"))
        return {"installation": identity, "files": {str(path): sha256(path) for path in paths}}

    def _ledger(self):
        ledger = read_json(self.ledger_path)
        if ledger["binding"] != self.binding or self._binding() != self.binding:
            raise ValueError("ARIA config, source, catalog or graph changed; use a new session")
        return ledger

    def catalog(self):
        self._ledger()
        return {"tools": copy.deepcopy(list(self.tools.values())),
                "guidance": ["ARIA produces hypotheses and explanations; independently evaluate scientific claims.",
                             "The configured graph is the bundled 2D demo." if self.config["kg_path"] is None else
                             "The knowledge graph is controller-supplied; its scientific coverage needs review.",
                             "Internal LLM requests and Ollama compute are not measured by this call ledger.",
                             "ARIA's direct literature searches are enabled." if self.config["allow_external_search"] else
                             "ARIA's direct literature search modes are disabled."],
                "model": self.config["model"], "mode": self.config["mode"],
                "kg_sha256": self.binding["files"][str(self.kg)], "aria_revision": REVISION}

    def budget(self):
        with locked(self.ledger_path):
            ledger = self._ledger()
            used = sum(entry["reserved_wall_seconds"] for entry in ledger["calls"])
            return {"reserved_calls": len(ledger["calls"]), "max_calls": self.config["max_calls"],
                    "remaining_calls": self.config["max_calls"] - len(ledger["calls"]),
                    "reserved_wall_seconds": used,
                    "remaining_reserved_wall_seconds": self.config["max_reserved_wall_seconds"] - used,
                    "model_tokens": None, "internal_llm_calls": None, "ollama_compute_seconds": None}

    def _arguments(self, name, arguments):
        if name not in self.tools or not isinstance(arguments, dict) or len(encode(arguments)) > 65536:
            raise ValueError("Unknown ARIA tool or invalid/oversized arguments")
        jsonschema.validate(arguments, self.tools[name]["function"]["parameters"])
        arguments = copy.deepcopy(arguments)
        remote = self.remote[name]
        properties = remote["inputSchema"]["properties"]
        for key, value in {"kg": str(self.kg), "model": self.config["model"], "mode": self.config["mode"]}.items():
            if key in properties:
                arguments[key] = value
        # The pinned upstream forward method does not propagate target_property.
        # Supply it in the processing dictionary that actually reaches routing.
        if name == PREFIX + "predict" or (name == PREFIX + "explain" and arguments.get("direction") == "forward"):
            arguments["processing"] = dict(arguments.get("processing") or {}, target_property=arguments["target_property"])
        if name == PREFIX + "explain":
            if arguments["direction"] not in {"forward", "inverse"}:
                raise ValueError("explain direction must be forward or inverse")
            required = "material" if arguments["direction"] == "forward" else "target_material"
            if not arguments.get(required):
                raise ValueError(f"explain requires {required}")
        jsonschema.validate(arguments, remote["inputSchema"])
        return arguments

    def call(self, name, arguments):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(name, arguments))
        raise RuntimeError("Use await aria.acall(...) inside an async controller")

    async def acall(self, name, arguments):
        effective = self._arguments(name, arguments)
        timeout = self.config["timeout_seconds"]
        with locked(self.ledger_path):
            ledger = self._ledger()
            used = sum(entry["reserved_wall_seconds"] for entry in ledger["calls"])
            if len(ledger["calls"]) >= self.config["max_calls"] or used + timeout > self.config["max_reserved_wall_seconds"]:
                raise ValueError("ARIA session budget exhausted")
            call_id = f"call-{len(ledger['calls']) + 1:04d}"
            output = self.session_root / call_id
            output.mkdir(exist_ok=False)
            # Snapshot the controller-selected graph for this call.
            graph = output / "knowledge-graph.json"
            graph.write_bytes(self.kg.read_bytes())
            if sha256(graph) != self.binding["files"][str(self.kg)]:
                raise ValueError("ARIA graph changed before snapshot")
            effective["kg"] = str(graph)
            request = {"tool": self.remote[name]["name"], "arguments": arguments, "effective_arguments": effective,
                       "requested_at": now(), "kg_sha256": sha256(graph), "aria_revision": REVISION}
            write_json(output / "request.json", request)
            ledger["calls"].append({"call_id": call_id, "status": "running", "controller_pid": os.getpid(),
                                    "reserved_wall_seconds": timeout, "request_sha256": sha256(output / "request.json")})
            write_json(self.ledger_path, ledger)
        import time
        started = time.monotonic()
        result = {"schema_version": 1, "provider": "aria", "call_id": call_id, "ok": False, "data": None,
                  "status": "transport_error", "scientific_validation": False,
                  "cost": {"reserved_wall_seconds": timeout, "model_tokens": None, "internal_llm_calls": None}}
        try:
            raw = await exchange("call", timeout=timeout, stderr_path=output / "server.stderr.log",
                                 tool=request["tool"], arguments=effective)
            serialized = encode(raw)
            result.update(response_sha256=hashlib.sha256(serialized).hexdigest(), response_bytes=len(serialized))
            if len(serialized) > self.config["max_response_bytes"]:
                result["status"] = "response_limit"
            else:
                data, status = payload(raw)
                if status == "completed" and request["tool"] != "diagnose":
                    if (not {"answer", "tier", "confidence", "reasoning_type"}.issubset(data)
                            or type(data["tier"]) is not int or data["tier"] not in {1, 2, 3}
                            or type(data["confidence"]) not in {int, float}
                            or not math.isfinite(data["confidence"]) or not 0 <= data["confidence"] <= 1):
                        status = "invalid_output"
                result.update(status=status, ok=status == "completed", data=data, raw=raw)
        except asyncio.CancelledError:
            result["status"] = "interrupted"
            raise
        except TimeoutError:
            result["status"] = "timeout"
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["error"] = "ARIA request failed; inspect local server logs and installation/model availability."
        finally:
            result.update(elapsed_seconds=time.monotonic() - started, finished_at=now())
            write_json(output / "result.json", result)
            with locked(self.ledger_path):
                ledger = self._ledger()
                entry = next(item for item in ledger["calls"] if item["call_id"] == call_id)
                entry.update(status=result["status"], result_sha256=sha256(output / "result.json"),
                             finished_at=result["finished_at"])
                write_json(self.ledger_path, ledger)
        return result

    def evidence(self, call_ids, *, max_bytes=65536):
        """Export successful advisory results, without certifying their claims."""
        if (not isinstance(call_ids, list) or not call_ids
                or not all(isinstance(item, str) for item in call_ids)
                or len(call_ids) != len(set(call_ids))):
            raise ValueError("Select distinct successful ARIA call IDs")
        with locked(self.ledger_path):
            ledger = self._ledger()
            entries = {entry["call_id"]: entry for entry in ledger["calls"]}
            records = []
            for call_id in call_ids:
                entry = entries.get(call_id)
                if entry is None or entry["status"] != "completed":
                    raise ValueError("Only completed ARIA calls can be exported")
                output = self.session_root / call_id
                request_path, result_path = output / "request.json", output / "result.json"
                if sha256(request_path) != entry["request_sha256"] or sha256(result_path) != entry["result_sha256"]:
                    raise ValueError("ARIA evidence changed after execution")
                request = read_json(request_path)
                if sha256(output / "knowledge-graph.json") != request["kg_sha256"]:
                    raise ValueError("ARIA call graph changed after execution")
                records.append({"request": request, "result": read_json(result_path),
                                "request_sha256": entry["request_sha256"],
                                "result_sha256": entry["result_sha256"]})
            packet = {"schema_version": 1, "provider": "aria", "trust": "unverified_reasoning",
                      "scientific_validation": False, "binding": self.binding, "records": records}
            if len(encode(packet)) > max_bytes:
                raise ValueError("ARIA evidence exceeds byte budget")
            return packet

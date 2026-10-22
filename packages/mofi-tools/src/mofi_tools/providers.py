"""Portable catalog tools and a bridge to a controller-owned MOFi workspace."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import jsonschema

from .catalog import MofDatabaseCatalog


CATALOG_TOOL = "mofi_tool_catalog"
CATALOG_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class CatalogProvider:
    """A standalone, read-only catalog with no engine or workspace dependency."""

    def __init__(self):
        self.database = MofDatabaseCatalog()
        data = Path(__file__).parent / "data"
        manifest = json.loads((data / "mof-database-catalog-v1.json").read_text(encoding="utf-8"))
        parameters = copy.deepcopy(manifest["parameters"])
        options = self.database.options()
        parameters["properties"]["method"]["enum"] = options["methods"] + [None]
        parameters["properties"]["identifier"]["enum"] = options["identifiers"] + [None]
        parameters["properties"]["required_fields"]["items"]["enum"] = options["fields"]
        self._catalog = {
            "profile": "catalog-only",
            "tools": [{"type": "function", "function": {
                "name": "mof_database_catalog", "description": manifest["description"],
                "parameters": parameters}}],
            "unavailable_tools": {"scientific_engines": "Use --workspace and its science Python environment to enable configured scientific tools."},
            "guidance": ["The bundled database table is unverified survey metadata, not structures or scientific evidence."],
        }

    def catalog(self):
        return copy.deepcopy(self._catalog)

    def call(self, name, arguments):
        if name != "mof_database_catalog":
            raise ValueError(f"Unknown tool: {name}")
        jsonschema.validate(arguments, self._catalog["tools"][0]["function"]["parameters"])
        return self.database.query(arguments)


class WorkspaceProvider:
    """Use the existing JSON bridge in a separately installed science environment.

    Only operator-supplied settings select the interpreter, workspace, config and
    session. Model arguments never become a command or a file path to execute.
    The hub retains its validation, reservations, execution and evidence ledgers.
    """

    def __init__(self, workspace, python, config, session):
        self.workspace = Path(workspace).resolve()
        self.python = str(Path(python).resolve())
        self.config = self._path(config)
        self.session = self._path(session)
        if not (self.workspace / "rsi/science/hub_bridge.py").is_file():
            raise ValueError("--workspace must contain the MOFi science hub bridge")
        if not Path(self.python).is_file() or not self.config.is_file():
            raise ValueError("Science Python and hub configuration must exist")
        self.binding = None
        self._catalog = self._request("catalog")
        self._definitions = {tool["function"]["name"]: tool["function"] for tool in self._catalog["tools"]}
        if CATALOG_TOOL in self._definitions:
            raise ValueError(f"Workspace tool name is reserved by the MCP adapter: {CATALOG_TOOL}")

    def _path(self, value):
        return (self.workspace / value).resolve()

    def _request(self, command, **values):
        request = {"command": command, "config": str(self.config), "session": str(self.session), **values}
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = os.pathsep.join((str(self.workspace), str(self.workspace / "Crystal-Builder")))
        # Engine messages go to stderr. Only the JSON reply is captured; stdout
        # from the MCP process remains exclusively a protocol transport.
        result = subprocess.run(
            [self.python, "-m", "rsi.science.hub_bridge"], cwd=self.workspace,
            env=env, input=json.dumps(request, allow_nan=False), stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, encoding="utf-8", check=False,
            timeout=60 if command == "catalog" else None,
        )
        if result.returncode:
            raise RuntimeError(f"Science bridge exited with code {result.returncode}; inspect stderr")
        try:
            reply = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("Science bridge did not return one JSON reply") from exc
        if not isinstance(reply, dict) or reply.get("ok") is not True:
            error = reply.get("error", {}) if isinstance(reply, dict) else {}
            raise RuntimeError(f"Science bridge rejected request: {error.get('message', 'invalid envelope')}")
        if not isinstance(reply.get("binding"), dict) or not isinstance(reply.get("result"), dict):
            raise RuntimeError("Science bridge reply is missing its result or binding")
        if self.binding is not None and reply["binding"] != self.binding:
            raise RuntimeError("Science hub binding changed; start a fresh MCP session")
        self.binding = copy.deepcopy(reply["binding"])
        return reply["result"]

    def catalog(self):
        return copy.deepcopy(self._catalog)

    def call(self, name, arguments):
        if name not in self._definitions:
            raise ValueError(f"Unknown or unavailable tool: {name}")
        jsonschema.validate(arguments, self._definitions[name]["parameters"])
        return self._request("call", tool=name, arguments=arguments)

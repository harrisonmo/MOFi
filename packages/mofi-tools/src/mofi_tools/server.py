"""Expose the existing bounded tool schemas through MCP over standard I/O."""

from __future__ import annotations

import json

import anyio
import jsonschema
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.helper_types import ReadResourceContents

from . import __version__
from .providers import CATALOG_SCHEMA, CATALOG_TOOL


def create_server(provider):
    server = Server(
        "mofi", version=__version__,
        instructions="Use mofi_tool_catalog to inspect available scientific tools and controller limits. Database entries are unverified metadata. Scientific calls retain the MOFi hub's budgets and evidence checks; never infer scientific validity from a successful transport.",
    )
    catalog = provider.catalog()
    definitions = {item["function"]["name"]: item["function"] for item in catalog["tools"]}
    definitions[CATALOG_TOOL] = {
        "name": CATALOG_TOOL,
        "description": "List available MOFi tools, unavailable engines, guidance and initial controller limits. Use science_budget_status for live scientific budgets.",
        "parameters": CATALOG_SCHEMA,
    }
    call_lock = anyio.Lock()

    @server.list_tools()
    async def list_tools():
        return [types.Tool(
            name=name, description=definition["description"], inputSchema=definition["parameters"],
            annotations=types.ToolAnnotations(
                readOnlyHint=name in {CATALOG_TOOL, "mof_database_catalog", "science_budget_status"},
                destructiveHint=False,
                idempotentHint=name in {CATALOG_TOOL, "mof_database_catalog", "science_budget_status"},
                openWorldHint=name not in {CATALOG_TOOL, "mof_database_catalog", "science_budget_status"},
            ),
        ) for name, definition in definitions.items()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name not in definitions:
                raise ValueError(f"Unknown or unavailable tool: {name}")
            jsonschema.validate(arguments, definitions[name]["parameters"])
            async with call_lock:
                result = provider.catalog() if name == CATALOG_TOOL else await anyio.to_thread.run_sync(
                    lambda: provider.call(name, arguments), abandon_on_cancel=False,
                )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result, allow_nan=False))],
                structuredContent=result, isError=result.get("ok") is False,
            )
        except (ValueError, OSError, RuntimeError, jsonschema.ValidationError) as exc:
            result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result))],
                structuredContent=result, isError=True,
            )

    @server.list_resources()
    async def list_resources():
        return [types.Resource(uri="mofi://catalog", name="MOFi tool catalog", mimeType="application/json")]

    @server.read_resource()
    async def read_resource(uri):
        if str(uri) != "mofi://catalog":
            raise ValueError("Unknown MOFi resource")
        return [ReadResourceContents(content=json.dumps(provider.catalog(), allow_nan=False), mime_type="application/json")]

    return server


async def serve(provider):
    server = create_server(provider)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

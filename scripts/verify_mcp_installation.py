"""Exercise an installed MCP wheel outside the source tree without paid calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check(python, arguments, label):
    parameters = StdioServerParameters(command=str(python), args=["-I", "-m", "mofi_tools", "serve", *arguments])
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            if initialized.serverInfo.name != "mofi":
                raise ValueError("Unexpected server identity")
            names = sorted(tool.name for tool in (await session.list_tools()).tools)
            filters = dict(name_contains="QMOF", method=None, data_type_contains=None,
                           identifier=None, required_fields=None, usable_link_only=None)
            reply = await session.call_tool("mof_database_catalog", filters)
            payload = reply.structuredContent or {}
            payload = payload.get("data", payload)
            if reply.isError or payload.get("matches") != 1:
                raise ValueError("Installed catalog query failed")
            result = {"profile": label, "tools": names, "qmof_matches": payload["matches"]}
            if "science_budget_status" in names:
                budget = await session.call_tool("science_budget_status", {})
                if budget.isError or budget.structuredContent["data"]["calls_used"] != 0:
                    raise ValueError("Smoke inspection unexpectedly spent a scientific call")
                result["paid_scientific_calls"] = 0
            return result


async def verify(args):
    python = args.python.resolve()
    workspace = args.workspace.resolve() if args.workspace else None
    science_python = args.science_python.resolve() if args.science_python else None
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="mofi-installed-wheel-") as temporary:
        try:
            os.chdir(temporary)
            results = [await check(python, [], "installed-catalog")]
            if workspace:
                results.append(await check(python, ["--workspace", str(workspace),
                    "--science-python", str(science_python), "--session", str(Path(temporary) / "hub")],
                    "installed-workspace"))
        finally:
            os.chdir(original)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--science-python", type=Path)
    args = parser.parse_args()
    if bool(args.workspace) != bool(args.science_python):
        parser.error("--workspace and --science-python must be supplied together")
    print(json.dumps(asyncio.run(verify(args)), indent=2))


if __name__ == "__main__":
    main()

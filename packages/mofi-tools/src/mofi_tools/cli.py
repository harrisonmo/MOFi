"""Command-line inspection and stdio serving for portable MOFi tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anyio
import jsonschema

from . import __version__
from .providers import CatalogProvider, WorkspaceProvider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=("catalog", "query", "serve"))
    parser.add_argument("--arguments", default="{}", help="JSON filters for the standalone database query")
    parser.add_argument("--workspace", type=Path, help="Optional MOFi source checkout for scientific tools")
    parser.add_argument("--science-python", type=Path, help="Python executable with the science environment installed")
    parser.add_argument("--hub-config", type=Path, default=Path("configs/science-hub.json"))
    parser.add_argument("--session", type=Path, help="Persistent hub session directory; use a separate directory per client")
    args = parser.parse_args(argv)
    if args.workspace is None and (args.science_python is not None or args.session is not None):
        parser.error("--science-python and --session require --workspace")
    if args.workspace is not None and (args.science_python is None or args.session is None):
        parser.error("--workspace requires --science-python and --session")
    try:
        provider = WorkspaceProvider(args.workspace, args.science_python, args.hub_config, args.session) if args.workspace else CatalogProvider()
        if args.command == "serve":
            from .server import serve
            anyio.run(serve, provider)
        else:
            if args.command == "catalog":
                result = provider.catalog()
            else:
                arguments = json.loads(args.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Query arguments must be a JSON object")
                defaults = dict(name_contains=None, method=None, data_type_contains=None,
                                identifier=None, required_fields=None, usable_link_only=None)
                result = provider.call("mof_database_catalog", {**defaults, **arguments})
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, jsonschema.ValidationError) as exc:
        parser.exit(1, f"MOFi tools stopped: {exc}\n")
    return 0


def mcp_main():
    return main(["serve", *sys.argv[1:]])

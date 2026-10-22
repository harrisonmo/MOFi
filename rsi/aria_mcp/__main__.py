"""Explicit ARIA MCP setup, model-facing catalog, calls and evidence export."""

import argparse
import json
from pathlib import Path
import sys

from rsi.hpc.core import read_json, sha256
from .core import AriaTools, PREFIX, discover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("discover", "list", "budget", "call", "export"):
        item = sub.add_parser(command)
        item.add_argument("--config", type=Path, default=Path("configs/aria-mcp.json"))
        if command == "discover":
            item.add_argument("--output", type=Path, default=Path("runs/aria/catalog.json"))
        else:
            item.add_argument("--session", type=Path, required=True)
        if command == "call":
            item.add_argument("--tool", required=True)
            item.add_argument("--arguments-file", type=Path, required=True)
        if command == "export":
            item.add_argument("--call-id", action="append", required=True)
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--max-bytes", type=int, default=65536)
    args = parser.parse_args()
    try:
        if args.command == "discover":
            result = discover(args.config, args.output)
            print(f"Discovered {len(result['tools'])} ARIA MCP tools. Catalog: {args.output}")
            return
        tools = AriaTools(args.config, args.session)
        if args.command == "list":
            result = tools.catalog()
        elif args.command == "budget":
            result = tools.budget()
        elif args.command == "call":
            name = args.tool if args.tool.startswith(PREFIX) else PREFIX + args.tool
            result = tools.call(name, read_json(args.arguments_file))
        else:
            if not 1024 <= args.max_bytes <= 4194304:
                raise ValueError("Evidence byte budget must be between 1024 and 4194304")
            if args.output.exists():
                raise FileExistsError("Choose a fresh ARIA evidence output path")
            result = tools.evidence(args.call_id, max_bytes=args.max_bytes)
            content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if len(content.encode("utf-8")) > args.max_bytes:
                raise ValueError("Formatted evidence exceeds byte cap; select fewer calls or increase the cap")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(f"Evidence: {args.output}\nSHA-256: {sha256(args.output)}")
            return
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "call" and not result["ok"]:
            sys.exit(1)
    except Exception as exc:
        # Full server messages and graphs stay in local session evidence.
        print(f"ARIA operation failed ({type(exc).__name__}). Check the pinned installation, "
              "catalog, graph, configuration, model availability and session budget.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

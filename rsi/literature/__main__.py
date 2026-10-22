"""Explicit login, discovery, bounded tool calls and evidence export."""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

from rsi.hpc.core import read_json, sha256, write_json
from .client import DEFAULT_AUTH, discover, session
from .valency import PREFIX, ValencyTools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("login", "discover"):
        item = sub.add_parser(command)
        item.add_argument("--auth-file", type=Path, default=DEFAULT_AUTH)
        if command == "discover":
            item.add_argument("--output", type=Path, default=Path("runs/valency/catalog.json"))
    for command in ("list", "budget", "call", "export"):
        item = sub.add_parser(command)
        item.add_argument("--config", type=Path, default=Path("configs/valency.json"))
        item.add_argument("--session", type=Path, required=True)
        item.add_argument("--auth-file", type=Path, default=DEFAULT_AUTH)
        if command == "call":
            item.add_argument("--tool", required=True)
            item.add_argument("--arguments-file", type=Path, required=True)
        if command == "export":
            item.add_argument("--call-id", action="append", required=True)
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--max-bytes", type=int, default=65536)
    args = parser.parse_args()
    try:
        if args.command == "login":
            async def login():
                async with session(args.auth_file, interactive=True, timeout=300):
                    pass
            asyncio.run(login())
            print("Valency OAuth login completed. Run discover to save the current tool catalog.")
            return
        if args.command == "discover":
            if args.output.exists():
                raise FileExistsError("Choose a fresh catalog output path; do not replace a frozen catalog")
            result = asyncio.run(discover(args.auth_file))
            result["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
            print(f"Discovered {len(result['tools'])} tools. Catalog: {args.output}")
            print("Review read-only tools and add their exact names to configs/valency.json allowed_tools.")
            return
        tools = ValencyTools(args.config, args.session, auth_path=args.auth_file)
        if args.command == "list":
            result = tools.catalog()
        elif args.command == "budget":
            result = tools.budget()
        elif args.command == "call":
            name = args.tool if args.tool.startswith(PREFIX) else PREFIX + args.tool
            result = tools.call(name, read_json(args.arguments_file))
        else:
            if args.max_bytes < 1024:
                raise ValueError("Evidence byte budget must be at least 1024")
            result = tools.evidence(args.call_id, max_bytes=args.max_bytes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                raise FileExistsError("Choose a fresh evidence output path")
            write_json(args.output, result)
            if args.output.stat().st_size > args.max_bytes:
                args.output.unlink()
                raise ValueError("Formatted evidence exceeds byte cap; choose fewer calls or a larger cap")
            print(f"Evidence: {args.output}\nSHA-256: {sha256(args.output)}")
            return
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "call" and not result["ok"]:
            sys.exit(1)
    except Exception as exc:
        # Do not print HTTP/SDK exceptions, which can include OAuth credentials.
        print(f"Valency operation failed ({type(exc).__name__}). Check configuration, login, "
              "catalog, session budget and connectivity. No automatic retry was made.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Prepare or run one Zeo++ recipe, or list/dispatch the registered pore tool."""

import argparse
import json

from . import core
from .registry import TOOL, ZeoppToolRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze a recipe into a new run folder")
    prepare.add_argument("--recipe", required=True)
    prepare.add_argument("--run", required=True)
    run = commands.add_parser("run", help="Execute a prepared run folder")
    run.add_argument("--run", required=True)
    for name in ("list", "call"):
        tool = commands.add_parser(name, help=f"{name} the registered tool")
        tool.add_argument("--config", default=str(core.ROOT / "configs/zeopp-tools.json"))
        tool.add_argument("--session", default=str(core.ROOT / "runs/zeopp-tools-default"))
        if name == "call":
            tool.add_argument("--arguments", required=True,
                              help='JSON object, e.g. {"task_id":"hkust1-n2"}')
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = core.prepare(args.recipe, args.run)
        elif args.command == "run":
            result = core.execute(args.run)
        else:
            registry = ZeoppToolRegistry(args.config, args.session)
            result = (registry.catalog() if args.command == "list"
                      else registry.call(TOOL, json.loads(args.arguments)))
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("ok") is False:
            parser.exit(1)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Zeo++ tool stopped: {exc}\n")


if __name__ == "__main__":
    main()

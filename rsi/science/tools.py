"""Discover scientific tools or dispatch a model tool call through a fixed catalog."""

import argparse
import json

from .tool_registry import ROOT, ScientificToolRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "call"))
    parser.add_argument("--config", default=str(ROOT / "configs/scientific-tools.json"))
    parser.add_argument("--session", default=str(ROOT / "runs/scientific-tools-default"))
    parser.add_argument("--tool", default="methane_gcmc_capacity")
    parser.add_argument("--arguments", help='JSON object, e.g. {"task_id":"methane-mfi-example"}')
    args = parser.parse_args()
    try:
        registry = ScientificToolRegistry(args.config, args.session)
        if args.command == "list":
            result = registry.catalog()
        else:
            if args.arguments is None:
                parser.error("call requires --arguments")
            result = registry.call(args.tool, json.loads(args.arguments))
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("ok") is False:
            parser.exit(1)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Scientific tool stopped: {exc}\n")


if __name__ == "__main__":
    main()

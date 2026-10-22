"""Command line interface: prepare a frozen run, then execute it once."""

import argparse
import json

from .core import execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare", help="Validate a CIF, explicit force field and finite budget")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Run every frozen pressure/seed job; never resume a failed attempt")
    run.add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.config, args.output) if args.command == "prepare" else execute(args.run)
    except (OSError, ValueError, RuntimeError, TimeoutError, ImportError) as exc:
        parser.exit(1, f"GCMC stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

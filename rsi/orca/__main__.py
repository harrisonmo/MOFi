"""Command line interface: prepare a frozen ORCA run, execute it, or parse a native output file."""

import argparse
import json
from pathlib import Path

from .core import execute, prepare
from .parse import parse_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare", help="Validate the molecule, method, spin and budget; write the input")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Execute a prepared run once (RSI_ORCA names the binary)")
    run.add_argument("--run", required=True)
    parse = sub.add_parser("parse", help="Parse termination, SCF, optimization and energies from job.out")
    parse.add_argument("--log", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.output)
        elif args.command == "parse":
            result = parse_output(Path(args.log).read_text(encoding="utf-8", errors="replace"))
        else:
            result = execute(args.run)
    except (OSError, ValueError, RuntimeError, TimeoutError, ImportError) as exc:
        parser.exit(1, f"ORCA stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

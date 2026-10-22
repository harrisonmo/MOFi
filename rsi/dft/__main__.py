"""Command line interface: prepare a frozen run, execute it, resume it, or parse a native log."""

import argparse
import json
from pathlib import Path

from .core import execute, prepare
from .parse import parse_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare", help="Validate structure, explicit method, constraints and budget")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Execute the first attempt of a prepared run")
    run.add_argument("--run", required=True)
    resume = sub.add_parser("resume", help="Continue a timed-out or interrupted run from its checkpoint")
    resume.add_argument("--run", required=True)
    parse = sub.add_parser("parse", help="Parse SCF convergence and energies from a native log")
    parse.add_argument("--backend", required=True, choices=("gpaw", "cp2k"))
    parse.add_argument("--log", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.config, args.output)
        elif args.command == "parse":
            result = parse_native(args.backend, Path(args.log).read_text(encoding="utf-8", errors="replace"))
        else:
            result = execute(args.run, resume=args.command == "resume")
    except (OSError, ValueError, RuntimeError, TimeoutError, ImportError) as exc:
        parser.exit(1, f"DFT stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

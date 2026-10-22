"""Prepare or execute a bounded Qwen3-8B QLoRA candidate."""

import argparse
import json

from rsi.hpc import Scheduler
from .core import prepare, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admission = commands.add_parser("prepare")
    admission.add_argument("--config", required=True)
    admission.add_argument("--data", required=True)
    admission.add_argument("--run", required=True)
    training = commands.add_parser("train")
    training.add_argument("--run", required=True)
    training.add_argument("--model-path", required=True)
    training.add_argument("--hpc-config", required=True)
    training.add_argument("--session", required=True)
    training.add_argument("--backend", default="local")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config, args.data, args.run)
    else:
        result = execute(args.run, args.model_path, Scheduler(args.hpc_config, args.session), args.backend)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Command line interface: probe compute, run one job spec, or inspect, wait for or cancel session jobs."""

import argparse
import json

from .core import Scheduler, probe, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="Report CPUs, GPUs and Slurm commands visible from this host")
    for name, text in (("run", "Submit a job spec and wait for it"), ("submit", "Submit a Slurm job spec and return"),
                       ("status", "Show session budget, usage and jobs"), ("wait", "Wait for a Slurm job"),
                       ("cancel", "Cancel a Slurm job")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--config", required=True)
        command.add_argument("--session", required=True)
        if name in {"run", "submit"}:
            command.add_argument("--spec", required=True)
            command.add_argument("--backend")
            command.add_argument("--capacity-timeout", type=float, default=0.0)
        if name in {"wait", "cancel"}:
            command.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            result = probe()
        else:
            scheduler = Scheduler(args.config, args.session)
            if args.command == "run":
                result = scheduler.run(read_json(args.spec), args.backend, args.capacity_timeout)
            elif args.command == "submit":
                result = scheduler.submit(read_json(args.spec), args.backend, args.capacity_timeout)
                if result["backend_type"] == "local":
                    # A local job belongs to its controller process; returning would orphan it.
                    result = scheduler.wait(result["id"])
            elif args.command == "status":
                result = scheduler.status()
            elif args.command == "wait":
                result = scheduler.wait(args.job)
            else:
                result = scheduler.cancel(args.job)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"HPC stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

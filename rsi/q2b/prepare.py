"""Prepare the four approved sources once, with durable resource accounting."""
import argparse
import json
from pathlib import Path

from rsi.experiment import write_json
from .limits import Ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.output / "science_budget.json")
    for index in range(4):
        target = args.output / f"source_{index}.json"
        if target.exists():
            continue
        value = ledger.run("source", str(index), {"action": "source", "index": index})
        write_json(target, value)
        print(json.dumps({k: value[k] for k in ("role", "counts", "guest")}), flush=True)


if __name__ == "__main__":
    main()

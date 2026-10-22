#!/usr/bin/env python3
"""Select a deterministic, bounded replay subset from an extxyz training file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ase.io import iread, write


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")

    rng = random.Random(args.seed)
    reservoir: list[tuple[int, object]] = []
    total = 0
    for index, atoms in enumerate(iread(args.input, index=":")):
        total += 1
        if len(reservoir) < args.max_frames:
            reservoir.append((index, atoms.copy()))
            continue
        replacement = rng.randint(0, index)
        if replacement < args.max_frames:
            reservoir[replacement] = (index, atoms.copy())

    reservoir.sort(key=lambda pair: pair[0])
    selected = [atoms for _, atoms in reservoir]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, selected, format="extxyz")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    payload = {
        "input": str(args.input),
        "output": str(args.output),
        "total_input_frames": total,
        "selected_frames": len(selected),
        "max_frames": args.max_frames,
        "seed": args.seed,
        "selection": "deterministic_reservoir_sample_without_replacement",
        "output_sha256": digest,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a new audited rMD17 extxyz package for micro-MLIP RSI v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import build_rmd17_dataset


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-v1.json"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing verified rMD17 NPZ files")
    parser.add_argument("--output-root", type=Path, required=True, help="New destination; it must not already exist")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_json(args.config)
    manifest = build_rmd17_dataset(
        config=config,
        raw_root=args.raw_root,
        output_root=args.output_root,
        config_sha256=sha256(args.config),
    )
    summary = {
        "benchmark_id": manifest["benchmark_id"],
        "dataset_root": str(args.output_root),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "molecules": {
            molecule: {
                "role": entry["role"],
                "atoms_per_frame": entry["atoms_per_frame"],
                "valid_frames": entry["files"]["valid"]["frames"],
                "test_frames": entry["files"]["test"]["frames"],
            }
            for molecule, entry in manifest["molecules"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

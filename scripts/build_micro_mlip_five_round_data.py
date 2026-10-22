#!/usr/bin/env python3
"""Build a fresh, audited rMD17 five-round selection-tree data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_round_data import build_five_round_dataset


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    manifest = build_five_round_dataset(config, args.raw_root, args.output_root, sha256(args.config))
    print(
        json.dumps(
            {
                "status": "MATERIALIZED_PENDING_AUDIT",
                "dataset_root": str(args.output_root),
                "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
                "molecules": sorted(manifest["molecules"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Emit the complete 2^4 scratch Prompt/Data/Tool/Workflow registry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.factorial import full_scratch_pdtw_registry


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-pdtw-full-scratch-screen-v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite full-factorial registry: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = full_scratch_pdtw_registry(config)
    result["created_at"] = utc_now()
    result["config_sha256"] = sha256(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ready": result["ready_cell_count"], "blocked": result["blocked_cell_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

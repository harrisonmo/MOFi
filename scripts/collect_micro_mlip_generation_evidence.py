#!/usr/bin/env python3
"""Assemble safe g0/g1 Evolution-Loop evidence from pre-release receipts.

Inputs must be the ``validation_only_evidence/*.json`` documents written by
the five-round controller before it releases the independent evaluator.  The
collector deliberately does not accept a campaign root, state file, model
directory, or evaluator result path: that narrow interface prevents a later
outer-policy proposal from receiving independent outcomes by mistake.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.generation_evidence import (
    GENERATION_SCHEMA,
    VALIDATION_ONLY_SCHEMA,
    sanitize_generation_evidence,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_documents(paths: Iterable[Path], generation: str) -> Dict[str, Any]:
    """Merge matching per-molecule pre-release documents and revalidate them."""

    source_paths = list(paths)
    if not source_paths:
        raise ValueError("At least one validation-only evidence document is required")
    episodes = []
    sources = []
    shared_cell_id: str | None = None
    shared_levels: Mapping[str, Any] | None = None
    observed_molecule_seed: set[tuple[str, int]] = set()
    for path in source_paths:
        source = load_json(path)
        if source.get("schema") != VALIDATION_ONLY_SCHEMA:
            raise ValueError(f"Unexpected validation-only schema: {path}")
        cell_id = source.get("factorial_cell_id")
        levels = source.get("factorial_levels")
        if shared_cell_id is None:
            shared_cell_id = cell_id
            shared_levels = levels
        elif cell_id != shared_cell_id or levels != shared_levels:
            raise ValueError("All inputs to one policy proposal must have the same factorial cell")
        molecule = source.get("molecule")
        seed = source.get("seed")
        if not isinstance(molecule, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Invalid molecule/seed receipt: {path}")
        pair = (molecule, seed)
        if pair in observed_molecule_seed:
            raise ValueError("A molecule/seed pre-release receipt was supplied twice")
        observed_molecule_seed.add(pair)
        raw_episodes = source.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) != 4:
            raise ValueError(f"Expected exactly four arm receipts: {path}")
        episodes.extend(raw_episodes)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "campaign_id": source.get("campaign_id"),
                "molecule": molecule,
                "seed": seed,
            }
        )
    unsafe = {
        "schema": GENERATION_SCHEMA,
        "generation": generation,
        "factorial_cell_id": shared_cell_id,
        "factorial_levels": dict(shared_levels) if isinstance(shared_levels, Mapping) else None,
        "episodes": episodes,
    }
    safe = sanitize_generation_evidence(unsafe)
    safe["created_at"] = utc_now()
    safe["source_validation_only_receipts"] = sorted(
        sources, key=lambda row: (str(row["campaign_id"]), str(row["molecule"]), int(row["seed"]))
    )
    safe["factorial_cell_id"] = shared_cell_id
    safe["factorial_levels"] = dict(shared_levels) if isinstance(shared_levels, Mapping) else None
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--input", dest="inputs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite generation evidence: {args.output}")
    payload = collect_documents(args.inputs, args.generation)
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "generation": payload["generation"],
                "episode_count": len(payload["episodes"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

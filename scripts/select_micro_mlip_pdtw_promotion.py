#!/usr/bin/env python3
"""Select one frozen g0/g1 policy for the toluene holdout using validation only.

The selection estimand is the within-molecule/seed paired effect of the
outer-generated arm versus its information-removed control.  The script never
accepts campaign directories or evaluator files: it receives only the narrow
generation-evidence documents created before independent release.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.generation_evidence import GENERATION_SCHEMA, sanitize_generation_evidence
from rsi.micro_mlip.five_round import policy_hash


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


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot compute a mean of no paired observations")
    return float(statistics.fmean(items))


def manifest_key(manifest: Mapping[str, Any]) -> tuple[str, str]:
    generation = manifest.get("generation")
    cell_id = manifest.get("factorial_cell_id")
    if not isinstance(generation, str) or not generation or not isinstance(cell_id, str) or not cell_id:
        raise ValueError("Policy manifest has no generation/cell identity")
    return generation, cell_id


def validated_manifest(path: Path) -> Dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != "micro-mlip-five-round-policies-v1":
        raise ValueError(f"Unexpected policy manifest schema: {path}")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"fixed", "outer_generated", "manual_control", "removed"}:
        raise ValueError(f"Policy manifest does not have the matched four arms: {path}")
    for arm, entry in arms.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise ValueError(f"Policy manifest arm is invalid: {path}/{arm}")
        if entry.get("policy_sha256") != policy_hash(entry["policy"]):
            raise ValueError(f"Policy manifest hash mismatch: {path}/{arm}")
    return manifest


def candidate_score(evidence_path: Path, policy_path: Path, force_threshold: float) -> Dict[str, Any]:
    """Return the fixed validation-only score for one generation/cell policy."""

    raw = load_json(evidence_path)
    if raw.get("schema") != GENERATION_SCHEMA:
        raise ValueError(f"Unexpected generation-evidence schema: {evidence_path}")
    evidence = sanitize_generation_evidence(raw)
    generation = evidence["generation"]
    cell_id = raw.get("factorial_cell_id")
    if not isinstance(cell_id, str):
        raise ValueError("Generation evidence has no factorial cell identity")
    manifest = validated_manifest(policy_path)
    if manifest_key(manifest) != (generation, cell_id):
        raise ValueError("Policy manifest and validation evidence have different generation/cell identities")
    manifest_levels = manifest.get("factorial_levels")
    if manifest_levels != raw.get("factorial_levels"):
        raise ValueError("Policy manifest and validation evidence have different cell levels")
    episodes = evidence["episodes"]
    by_pair: Dict[tuple[str, int], Dict[str, Mapping[str, Any]]] = {}
    for episode in episodes:
        pair = (str(episode["molecule"]), int(episode["seed"]))
        arms = by_pair.setdefault(pair, {})
        arm = str(episode["arm"])
        if arm in arms:
            raise ValueError("Evidence includes a duplicated arm pairing")
        arms[arm] = episode
    pair_rows = []
    expected_hashes = {arm: manifest["arms"][arm]["policy_sha256"] for arm in manifest["arms"]}
    for (molecule, seed), arms in sorted(by_pair.items()):
        if set(arms) != set(expected_hashes):
            raise ValueError("Evidence does not retain a complete four-arm paired comparison")
        for arm, episode in arms.items():
            if episode["policy_sha256"] != expected_hashes[arm]:
                raise ValueError("Evidence episode does not match its frozen policy manifest")
        outer_rounds = arms["outer_generated"]["rounds"]
        removed_rounds = arms["removed"]["rounds"]
        final_outer = outer_rounds[-1]["validation_metrics"]
        final_removed = removed_rounds[-1]["validation_metrics"]
        outer_force = [row["validation_metrics"]["force_component_mae_meV_per_A"] for row in outer_rounds]
        removed_force = [row["validation_metrics"]["force_component_mae_meV_per_A"] for row in removed_rounds]
        outer_energy = [row["validation_metrics"]["energy_mae_meV_per_atom"] for row in outer_rounds]
        removed_energy = [row["validation_metrics"]["energy_mae_meV_per_atom"] for row in removed_rounds]
        outer_time = sum(row["training_duration_seconds"] + row["validation_duration_seconds"] for row in outer_rounds)
        removed_time = sum(row["training_duration_seconds"] + row["validation_duration_seconds"] for row in removed_rounds)
        outer_threshold = next((index + 1 for index, value in enumerate(outer_force) if value <= force_threshold), None)
        removed_threshold = next((index + 1 for index, value in enumerate(removed_force) if value <= force_threshold), None)
        pair_rows.append(
            {
                "molecule": molecule,
                "seed": seed,
                "final_validation_force_delta_outer_minus_removed_meV_per_A": (
                    final_outer["force_component_mae_meV_per_A"] - final_removed["force_component_mae_meV_per_A"]
                ),
                "final_validation_energy_delta_outer_minus_removed_meV_per_atom": (
                    final_outer["energy_mae_meV_per_atom"] - final_removed["energy_mae_meV_per_atom"]
                ),
                "validation_force_auc_delta_outer_minus_removed_meV_per_A": mean(outer_force) - mean(removed_force),
                "cumulative_validation_time_delta_outer_minus_removed_seconds": outer_time - removed_time,
                "outer_labels_to_force_threshold": outer_threshold,
                "removed_labels_to_force_threshold": removed_threshold,
            }
        )
    return {
        "generation": generation,
        "factorial_cell_id": cell_id,
        "factorial_levels": manifest_levels,
        "evidence": str(evidence_path),
        "evidence_sha256": sha256(evidence_path),
        "policy_manifest": str(policy_path),
        "policy_manifest_sha256": sha256(policy_path),
        "outer_policy_sha256": expected_hashes["outer_generated"],
        "paired_rows": pair_rows,
        "summary": {
            "pair_count": len(pair_rows),
            "mean_final_validation_force_delta_outer_minus_removed_meV_per_A": mean(
                row["final_validation_force_delta_outer_minus_removed_meV_per_A"] for row in pair_rows
            ),
            "mean_validation_force_auc_delta_outer_minus_removed_meV_per_A": mean(
                row["validation_force_auc_delta_outer_minus_removed_meV_per_A"] for row in pair_rows
            ),
            "mean_final_validation_energy_delta_outer_minus_removed_meV_per_atom": mean(
                row["final_validation_energy_delta_outer_minus_removed_meV_per_atom"] for row in pair_rows
            ),
            "mean_cumulative_validation_time_delta_outer_minus_removed_seconds": mean(
                row["cumulative_validation_time_delta_outer_minus_removed_seconds"] for row in pair_rows
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-round-config", type=Path, required=True)
    parser.add_argument("--evidence", dest="evidences", type=Path, action="append", required=True)
    parser.add_argument("--policy-manifest", dest="policies", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite promotion selection: {args.output}")
    if len(args.evidences) != len(args.policies):
        raise ValueError("Every generation-evidence file requires one matching policy manifest")
    config = load_json(args.five_round_config)
    force_threshold = float(config["validation_residual_diagnostic"]["force_mae_threshold_meV_per_A"])
    candidates = [
        candidate_score(evidence, policy, force_threshold)
        for evidence, policy in zip(args.evidences, args.policies)
    ]
    keys = [(row["generation"], row["factorial_cell_id"]) for row in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("Each generation/cell policy may appear only once in the promotion set")
    candidates.sort(
        key=lambda row: (
            row["summary"]["mean_final_validation_force_delta_outer_minus_removed_meV_per_A"],
            row["summary"]["mean_validation_force_auc_delta_outer_minus_removed_meV_per_A"],
            row["generation"],
            row["factorial_cell_id"],
        )
    )
    selected = candidates[0]
    payload = {
        "schema": "micro-mlip-pdtw-promotion-selection-v1",
        "status": "FROZEN_BEFORE_POLICY_HOLDOUT",
        "created_at": utc_now(),
        "five_round_config": str(args.five_round_config),
        "five_round_config_sha256": sha256(args.five_round_config),
        "selection_rule": {
            "primary": "minimize mean paired final validation force MAE (outer_generated minus removed)",
            "secondary": "minimize mean paired five-round validation force AUC (outer_generated minus removed)",
            "tie_break": "lexicographic generation then factorial cell ID",
            "force_threshold_meV_per_A": force_threshold,
        },
        "candidates": candidates,
        "selected": selected,
        "information_boundary": (
            "Only development training/validation trajectories are used. Independent evaluation metrics and "
            "the toluene policy-holdout data are excluded from the selection rule."
        ),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_generation": selected["generation"],
                "selected_cell": selected["factorial_cell_id"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

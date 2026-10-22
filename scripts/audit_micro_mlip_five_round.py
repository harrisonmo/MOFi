#!/usr/bin/env python3
"""Fail closed on hashes, tree continuity, and test isolation for five-round data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

from ase.io import iread

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_round import ADAPTIVE_ROUNDS, ALLOWED_SAMPLERS, ROUND_COUNT, child_prefix, initial_prefix


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def add_error(errors: List[str], label: str, detail: str) -> None:
    errors.append(f"{label}:{detail}")


def expected_ids(molecule: str, indices: Sequence[int]) -> List[str]:
    return [f"rmd17:{molecule}:train_01:{int(index)}" for index in indices]


def audit_prefix(
    dataset_root: Path,
    molecule: str,
    prefix: str,
    entry: Mapping[str, Any],
    train_pool: set[int],
    initial_labels: int,
    labels_per_round: int,
    prefixes: Mapping[str, Any],
    errors: List[str],
    tracked: Dict[str, str],
) -> None:
    label = f"{molecule}:{prefix}"
    round_index = entry.get("round_index")
    history = entry.get("action_history")
    indices = entry.get("source_train_indices")
    newly_selected = entry.get("new_source_train_indices")
    if not isinstance(round_index, int) or round_index < 1 or round_index > ROUND_COUNT:
        add_error(errors, label, "invalid_round_index")
        return
    if not isinstance(history, list) or len(history) != round_index - 1:
        add_error(errors, label, "history_length")
    elif any(action not in ALLOWED_SAMPLERS for action in history):
        add_error(errors, label, "history_sampler")
    if not isinstance(indices, list) or len(indices) != initial_labels + labels_per_round * (round_index - 1):
        add_error(errors, label, "label_count")
        return
    if len(set(indices)) != len(indices) or not set(indices).issubset(train_pool):
        add_error(errors, label, "invalid_source_indices")
    if not isinstance(newly_selected, list):
        add_error(errors, label, "missing_increment")
    elif round_index == 1:
        if newly_selected != indices:
            add_error(errors, label, "initial_increment")
    else:
        parent, _, action = prefix.rpartition("__")
        if not parent or action not in ALLOWED_SAMPLERS or parent not in prefixes:
            add_error(errors, label, "invalid_parent")
        else:
            parent_indices = prefixes[parent].get("source_train_indices", [])
            if indices[: len(parent_indices)] != parent_indices:
                add_error(errors, label, "noncumulative_parent")
            if newly_selected != indices[len(parent_indices) :]:
                add_error(errors, label, "increment_not_suffix")
            if len(newly_selected) != labels_per_round or set(newly_selected).intersection(parent_indices):
                add_error(errors, label, "invalid_increment")
            if prefix != child_prefix(parent, action):
                add_error(errors, label, "noncanonical_prefix")
    relative_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        add_error(errors, label, "missing_file_receipt")
        return
    path = dataset_root / relative_path
    if not path.is_file():
        add_error(errors, label, "missing_extxyz")
        return
    observed_hash = sha256(path)
    tracked[str(path)] = observed_hash
    if observed_hash != expected_hash:
        add_error(errors, label, "extxyz_hash")
    try:
        ids = [atoms.info.get("rmd17_record_id") for atoms in iread(path, format="extxyz")]
    except Exception as exc:
        add_error(errors, label, f"unreadable_extxyz:{type(exc).__name__}")
        return
    if ids != expected_ids(molecule, indices):
        add_error(errors, label, "extxyz_record_order")


def validate(config_path: Path, dataset_root: Path, raw_root: Path) -> Dict[str, Any]:
    """Return all detected violations instead of trusting a materialization receipt."""

    errors: List[str] = []
    config = load_json(config_path)
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["dataset_manifest:missing_file"], "dataset_root": str(dataset_root)}
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "micro-mlip-rmd17-five-round-dataset-v1":
        add_error(errors, "manifest", "unexpected_schema")
    if manifest.get("benchmark_id") != config.get("benchmark_id"):
        add_error(errors, "manifest", "benchmark_id")
    if manifest.get("config_sha256") != sha256(config_path):
        add_error(errors, "manifest", "config_hash")
    protocol = config["five_round_data_protocol"]
    if manifest.get("five_round_data_protocol", {}).get("selection_rule") != "coordinates_and_atomic_numbers_only":
        add_error(errors, "manifest", "selection_rule")
    if int(protocol["round_count"]) != ROUND_COUNT or tuple(protocol["adaptive_rounds"]) != ADAPTIVE_ROUNDS:
        add_error(errors, "config", "not_five_rounds")

    tracked: Dict[str, str] = {str(config_path): sha256(config_path), str(manifest_path): sha256(manifest_path)}
    parent_protocol = config["parent_benchmark"]["data_protocol"]
    initial_labels = int(protocol["initial_labels"])
    labels_per_round = int(protocol["labels_added_per_adaptive_round"])
    for molecule, source in config["raw_files"].items():
        entry = manifest.get("molecules", {}).get(molecule)
        if not isinstance(entry, dict):
            add_error(errors, molecule, "missing_manifest_entry")
            continue
        if entry.get("role") != config["molecule_roles"].get(molecule):
            add_error(errors, molecule, "role")
        for split_name in ("train", "test"):
            raw_path = raw_root / source[split_name]["filename"]
            if not raw_path.is_file():
                add_error(errors, molecule, f"missing_raw_{split_name}")
            else:
                observed = sha256(raw_path)
                tracked[str(raw_path)] = observed
                if observed != source[split_name]["sha256"]:
                    add_error(errors, molecule, f"raw_hash_{split_name}")
        partition = entry.get("source_train_partition_indices", {})
        required = ("train_pool", "valid", "reserve")
        if any(name not in partition for name in required):
            add_error(errors, molecule, "partition_missing")
            continue
        groups = {name: set(partition[name]) for name in required}
        expected_sizes = {
            "train_pool": int(parent_protocol["train_pool_size"]),
            "valid": int(parent_protocol["validation_size"]),
            "reserve": int(parent_protocol["reserve_size"]),
        }
        if any(len(groups[name]) != expected_sizes[name] for name in required):
            add_error(errors, molecule, "partition_sizes")
        if len(set().union(*groups.values())) != int(parent_protocol["source_train_frames"]):
            add_error(errors, molecule, "partition_overlap_or_gap")
        episodes = entry.get("episodes", {})
        prefixes = episodes.get("prefixes", {})
        if episodes.get("initial_prefix") != initial_prefix() or len(prefixes) != 31:
            add_error(errors, molecule, "tree_shape")
        for prefix, prefix_entry in prefixes.items():
            audit_prefix(
                dataset_root,
                molecule,
                prefix,
                prefix_entry,
                groups["train_pool"],
                initial_labels,
                labels_per_round,
                prefixes,
                errors,
                tracked,
            )
        for file_name in ("valid", "test"):
            file_entry = entry.get("files", {}).get(file_name, {})
            path_text = file_entry.get("path")
            if not isinstance(path_text, str):
                add_error(errors, molecule, f"{file_name}_path")
                continue
            path = dataset_root / path_text
            if not path.is_file():
                add_error(errors, molecule, f"{file_name}_missing")
            else:
                observed = sha256(path)
                tracked[str(path)] = observed
                if observed != file_entry.get("sha256"):
                    add_error(errors, molecule, f"{file_name}_hash")

    return {
        "schema": "micro-mlip-five-round-data-audit-v1",
        "valid": not errors,
        "errors": errors,
        "benchmark_id": config.get("benchmark_id"),
        "dataset_root": str(dataset_root),
        "tracked_sha256": tracked,
        "test_access": "NOT_USED_BY_SELECTION_TREE_OR_POLICY_DECISIONS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.config, args.dataset_root, args.raw_root)
    output = args.output or args.dataset_root / "dataset_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

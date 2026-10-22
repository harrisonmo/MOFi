#!/usr/bin/env python3
"""Fail-closed validation for a materialized micro-MLIP rMD17 data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from ase.io import iread

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import KCAL_MOL_TO_EV, sha256


DEFAULT_CONFIG = ROOT / "configs" / "micro-mlip-rsi-v1.json"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def add_error(errors: List[str], label: str, detail: str) -> None:
    errors.append(f"{label}:{detail}")


def read_ids_and_validate_frames(
    path: Path,
    expected_frames: int,
    errors: List[str],
    label: str,
) -> List[str]:
    if not path.is_file():
        add_error(errors, label, "missing_file")
        return []
    ids: List[str] = []
    try:
        frames = list(iread(path, format="extxyz"))
    except Exception as exc:
        add_error(errors, label, f"unreadable_extxyz:{type(exc).__name__}")
        return ids
    if len(frames) != expected_frames:
        add_error(errors, label, f"frame_count:{len(frames)}!=expected:{expected_frames}")
    for position, atoms in enumerate(frames):
        if bool(np.any(atoms.pbc)):
            add_error(errors, label, f"periodic_frame:{position}")
        if "REF_energy" not in atoms.info:
            add_error(errors, label, f"missing_REF_energy:{position}")
        else:
            try:
                if not np.isfinite(float(atoms.info["REF_energy"])):
                    add_error(errors, label, f"nonfinite_REF_energy:{position}")
            except (TypeError, ValueError):
                add_error(errors, label, f"invalid_REF_energy:{position}")
        force = atoms.arrays.get("REF_forces")
        if force is None:
            add_error(errors, label, f"missing_REF_forces:{position}")
        elif force.shape != (len(atoms), 3) or not np.isfinite(force).all():
            add_error(errors, label, f"invalid_REF_forces:{position}")
        record_id = atoms.info.get("rmd17_record_id")
        if not isinstance(record_id, str):
            add_error(errors, label, f"missing_record_id:{position}")
        else:
            ids.append(record_id)
    return ids


def expected_ids(molecule: str, source_partition: str, indices: Sequence[int]) -> List[str]:
    return [f"rmd17:{molecule}:{source_partition}:{int(index)}" for index in indices]


def validate(config_path: Path, dataset_root: Path, raw_root: Path) -> Dict[str, Any]:
    """Validate hashes, split integrity, labels, and label-blind selection outputs."""

    errors: List[str] = []
    config = load_json(config_path)
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return {
            "valid": False,
            "errors": ["dataset_manifest:missing_file"],
            "dataset_root": str(dataset_root),
        }
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "micro-mlip-rmd17-dataset-v1":
        add_error(errors, "manifest", "unexpected_schema")
    if manifest.get("benchmark_id") != config.get("benchmark_id"):
        add_error(errors, "manifest", "benchmark_id_mismatch")
    if manifest.get("config_sha256") != sha256(config_path):
        add_error(errors, "manifest", "config_hash_mismatch")
    conversion = manifest.get("unit_conversion", {})
    if conversion.get("kcal_mol_to_eV") != KCAL_MOL_TO_EV:
        add_error(errors, "manifest", "unit_conversion_mismatch")
    protocol = config["data_protocol"]
    if manifest.get("data_protocol", {}).get("selection_rule") != "coordinates_and_atomic_numbers_only":
        add_error(errors, "manifest", "selection_rule_not_label_blind")

    tracked_hashes: Dict[str, str] = {
        str(config_path): sha256(config_path),
        str(manifest_path): sha256(manifest_path),
    }
    for molecule, source in config["raw_files"].items():
        entry = manifest.get("molecules", {}).get(molecule)
        if not isinstance(entry, dict):
            add_error(errors, molecule, "missing_manifest_entry")
            continue
        if entry.get("role") != config["molecule_roles"].get(molecule):
            add_error(errors, molecule, "role_mismatch")
        raw_sources = entry.get("raw_sources", {})
        for split_name in ("train", "test"):
            configured = source[split_name]
            raw_path = raw_root / configured["filename"]
            if not raw_path.is_file():
                add_error(errors, molecule, f"missing_raw_{split_name}")
            else:
                observed = sha256(raw_path)
                tracked_hashes[str(raw_path)] = observed
                if observed != configured["sha256"]:
                    add_error(errors, molecule, f"raw_hash_mismatch_{split_name}")
            source_entry = raw_sources.get(split_name, {})
            if source_entry.get("filename") != configured["filename"]:
                add_error(errors, molecule, f"raw_filename_mismatch_{split_name}")
            if source_entry.get("sha256") != configured["sha256"]:
                add_error(errors, molecule, f"raw_manifest_hash_mismatch_{split_name}")

        partition = entry.get("source_train_partition_indices", {})
        required_sets = ("train_pool", "valid", "reserve")
        if any(name not in partition for name in required_sets):
            add_error(errors, molecule, "missing_partition")
            continue
        expected_lengths = {
            "train_pool": int(protocol["train_pool_size"]),
            "valid": int(protocol["validation_size"]),
            "reserve": int(protocol["reserve_size"]),
        }
        partition_sets = {}
        for name in required_sets:
            values = partition[name]
            if not isinstance(values, list) or len(values) != expected_lengths[name]:
                add_error(errors, molecule, f"partition_length_{name}")
                continue
            if len(set(values)) != len(values):
                add_error(errors, molecule, f"partition_duplicate_{name}")
            if any(not isinstance(value, int) or value < 0 or value >= int(protocol["source_train_frames"]) for value in values):
                add_error(errors, molecule, f"partition_index_range_{name}")
            partition_sets[name] = set(values)
        if len(partition_sets) == 3:
            union = set().union(*partition_sets.values())
            if len(union) != int(protocol["source_train_frames"]):
                add_error(errors, molecule, "partition_not_complete_disjoint")

        files = entry.get("files", {})
        for file_name, indices, source_partition in (
            ("valid", partition["valid"], "train_01"),
            ("test", list(range(int(protocol["source_test_frames"]))), "test_01"),
        ):
            file_entry = files.get(file_name, {})
            relative_path = file_entry.get("path")
            if not isinstance(relative_path, str):
                add_error(errors, molecule, f"missing_{file_name}_path")
                continue
            extxyz_path = dataset_root / relative_path
            if extxyz_path.is_file():
                observed = sha256(extxyz_path)
                tracked_hashes[str(extxyz_path)] = observed
                if observed != file_entry.get("sha256"):
                    add_error(errors, molecule, f"{file_name}_sha256_mismatch")
            ids = read_ids_and_validate_frames(
                extxyz_path,
                len(indices),
                errors,
                f"{molecule}:{file_name}",
            )
            if ids != expected_ids(molecule, source_partition, indices):
                add_error(errors, molecule, f"{file_name}_record_id_mismatch")

        selections = entry.get("selections", {})
        for strategy in protocol["selection_strategies"]:
            selection_entries = selections.get(strategy, {})
            for budget in protocol["label_budgets"]:
                selection = selection_entries.get(str(budget), {})
                indices = selection.get("source_train_indices", [])
                if len(indices) != int(budget) or len(set(indices)) != int(budget):
                    add_error(errors, molecule, f"selection_size:{strategy}:{budget}")
                if not set(indices).issubset(set(partition["train_pool"])):
                    add_error(errors, molecule, f"selection_outside_train_pool:{strategy}:{budget}")
                relative_path = selection.get("path")
                if not isinstance(relative_path, str):
                    add_error(errors, molecule, f"selection_missing_path:{strategy}:{budget}")
                    continue
                extxyz_path = dataset_root / relative_path
                if extxyz_path.is_file():
                    observed = sha256(extxyz_path)
                    tracked_hashes[str(extxyz_path)] = observed
                    if observed != selection.get("sha256"):
                        add_error(errors, molecule, f"selection_sha256:{strategy}:{budget}")
                ids = read_ids_and_validate_frames(
                    extxyz_path,
                    int(budget),
                    errors,
                    f"{molecule}:{strategy}:{budget}",
                )
                if ids != expected_ids(molecule, "train_01", indices):
                    add_error(errors, molecule, f"selection_record_id:{strategy}:{budget}")

    return {
        "valid": not errors,
        "errors": errors,
        "benchmark_id": config.get("benchmark_id"),
        "dataset_root": str(dataset_root),
        "tracked_sha256": tracked_hashes,
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

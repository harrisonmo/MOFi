"""Label-blind materialization of the five-round rMD17 selection tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .data import (
    KCAL_MOL_TO_EV,
    RMD17Arrays,
    deterministic_partition,
    load_rmd17_npz,
    select_geometry_coverage,
    select_random,
    sha256,
    stable_seed,
    write_extxyz,
    write_json,
)
from .five_round import ALLOWED_SAMPLERS, ADAPTIVE_ROUNDS, ROUND_COUNT, child_prefix, initial_prefix


def _checked_source(path: Path, expected_sha256: str) -> Tuple[RMD17Arrays, str]:
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"Raw source hash mismatch for {path.name}: expected {expected_sha256}, got {observed}")
    return load_rmd17_npz(path), observed


def _selection_file(prefix: str, round_index: int) -> str:
    safe_prefix = prefix.replace("__", "-")
    return f"episodes/round_{round_index:02d}_{safe_prefix}.xyz"


def _entry(
    output_root: Path,
    molecule_root: Path,
    arrays: RMD17Arrays,
    molecule: str,
    prefix: str,
    round_index: int,
    selected_indices: Sequence[int],
    newly_selected: Sequence[int],
    history: Sequence[str],
) -> Dict[str, Any]:
    relative_path = _selection_file(prefix, round_index)
    path = molecule_root / relative_path
    return {
        "path": str(path.relative_to(output_root)),
        "sha256": write_extxyz(path, arrays, selected_indices, molecule, "train_01"),
        "round_index": round_index,
        "frames": len(selected_indices),
        "source_train_indices": [int(value) for value in selected_indices],
        "new_source_train_indices": [int(value) for value in newly_selected],
        "action_history": list(history),
    }


def build_five_round_dataset(
    config: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    config_sha256: str,
) -> Dict[str, Any]:
    """Materialize every reachable, label-blind selection prefix exactly once.

    ``output_root`` must be a fresh directory.  The produced tree exposes only
    precomputed train files, their source indices, and hash receipts to the
    controller.  Selection sees coordinates and atomic numbers, never labels
    from the candidate pool.
    """

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_root}")
    protocol = config["five_round_data_protocol"]
    if int(protocol["round_count"]) != ROUND_COUNT:
        raise ValueError("The current controller requires exactly five training rounds")
    if tuple(protocol["adaptive_rounds"]) != ADAPTIVE_ROUNDS:
        raise ValueError("Five-round adaptive stages must be 2, 3, 4, and 5")
    samplers = tuple(protocol["selection_strategies"])
    if samplers != ALLOWED_SAMPLERS:
        raise ValueError("Five-round data contract requires the two registered samplers in canonical order")
    initial_labels = int(protocol["initial_labels"])
    labels_per_round = int(protocol["labels_added_per_adaptive_round"])
    final_labels = initial_labels + labels_per_round * len(ADAPTIVE_ROUNDS)
    if final_labels != int(protocol["final_labels"]):
        raise ValueError("Final label count does not match the five-round cumulative budget")

    base_protocol = config["parent_benchmark"]["data_protocol"]
    if int(base_protocol["train_pool_size"]) < final_labels:
        raise ValueError("Train pool does not support the requested five-round label budget")
    raw_root = raw_root.resolve()

    # Verify every raw source before writing a new immutable package.
    preflight: Dict[str, Tuple[RMD17Arrays, RMD17Arrays, str, str]] = {}
    for molecule, source in config["raw_files"].items():
        train_path = raw_root / source["train"]["filename"]
        test_path = raw_root / source["test"]["filename"]
        train, train_hash = _checked_source(train_path, source["train"]["sha256"])
        test, test_hash = _checked_source(test_path, source["test"]["sha256"])
        if not np.array_equal(train.atomic_numbers, test.atomic_numbers):
            raise ValueError(f"Atomic numbers differ between train and test for {molecule}")
        preflight[molecule] = (train, test, train_hash, test_hash)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-rmd17-five-round-dataset-v1",
        "benchmark_id": config["benchmark_id"],
        "config_sha256": config_sha256,
        "parent_benchmark": config["parent_benchmark"],
        "unit_conversion": {
            "input_energy": "kcal/mol",
            "input_forces": "kcal/(mol Angstrom)",
            "output_energy": "eV",
            "output_forces": "eV/Angstrom",
            "kcal_mol_to_eV": KCAL_MOL_TO_EV,
        },
        "five_round_data_protocol": {
            **protocol,
            "selection_rule": "coordinates_and_atomic_numbers_only",
            "selection_tree_nodes_per_molecule": 31,
        },
        "molecules": {},
    }

    for molecule, source in config["raw_files"].items():
        train, test, train_hash, test_hash = preflight[molecule]
        partition = deterministic_partition(
            train.frames,
            int(base_protocol["train_pool_size"]),
            int(base_protocol["validation_size"]),
            int(base_protocol["reserve_size"]),
            stable_seed(base_protocol["split_seed"], molecule, "partition"),
        )
        train_pool = partition["train_pool"]
        molecule_root = output_root / molecule
        initial = select_random(
            train_pool,
            initial_labels,
            stable_seed(protocol["selection_seed"], molecule, "five_round_initial"),
        )
        prefixes: Dict[str, Dict[str, Any]] = {
            initial_prefix(): _entry(
                output_root,
                molecule_root,
                train,
                molecule,
                initial_prefix(),
                1,
                initial,
                initial,
                [],
            )
        }
        frontier = [initial_prefix()]
        for round_index in ADAPTIVE_ROUNDS:
            next_frontier: List[str] = []
            for parent in frontier:
                parent_entry = prefixes[parent]
                parent_indices = parent_entry["source_train_indices"]
                remaining = sorted(set(train_pool).difference(parent_indices))
                if len(remaining) < labels_per_round:
                    raise AssertionError("Selection tree exhausted train-pool support unexpectedly")
                for sampler in samplers:
                    if sampler == "random":
                        added = select_random(
                            remaining,
                            labels_per_round,
                            stable_seed(protocol["selection_seed"], molecule, parent, sampler, round_index),
                        )
                    else:
                        added = select_geometry_coverage(train.coordinates_angstrom, remaining, labels_per_round)
                    child = child_prefix(parent, sampler)
                    if child in prefixes:
                        raise AssertionError(f"Duplicate five-round tree prefix: {child}")
                    selected = list(parent_indices) + list(added)
                    prefixes[child] = _entry(
                        output_root,
                        molecule_root,
                        train,
                        molecule,
                        child,
                        round_index,
                        selected,
                        added,
                        list(parent_entry["action_history"]) + [sampler],
                    )
                    next_frontier.append(child)
            frontier = next_frontier
        if len(prefixes) != 31:
            raise AssertionError("Five-round two-sampler tree must have exactly 31 unique prefixes")

        valid_path = molecule_root / "valid.xyz"
        test_path = molecule_root / "test.xyz"
        manifest["molecules"][molecule] = {
            "role": config["molecule_roles"][molecule],
            "atomic_numbers": [int(value) for value in train.atomic_numbers],
            "atoms_per_frame": train.atoms,
            "raw_sources": {
                "train": {"filename": source["train"]["filename"], "sha256": train_hash, "frames": train.frames},
                "test": {"filename": source["test"]["filename"], "sha256": test_hash, "frames": test.frames},
            },
            "source_train_partition_indices": partition,
            "files": {
                "valid": {
                    "path": str(valid_path.relative_to(output_root)),
                    "sha256": write_extxyz(valid_path, train, partition["valid"], molecule, "train_01"),
                    "frames": len(partition["valid"]),
                },
                "test": {
                    "path": str(test_path.relative_to(output_root)),
                    "sha256": write_extxyz(test_path, test, list(range(test.frames)), molecule, "test_01"),
                    "frames": test.frames,
                },
            },
            "episodes": {
                "initial_prefix": initial_prefix(),
                "prefixes": prefixes,
            },
        }
    manifest_path = output_root / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    manifest["dataset_manifest_sha256"] = sha256(manifest_path)
    return manifest

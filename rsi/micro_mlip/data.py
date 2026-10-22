"""Data preparation and metric helpers for the micro-MLIP benchmark.

This module deliberately keeps selection label-blind.  Geometry-only selection
can inspect atomic numbers and coordinates, but never reference energies or
forces outside the frames included in the selected label budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


KCAL_MOL_TO_EV = 0.0433641153087705
REQUIRED_RMD17_KEYS = ("coords", "energies", "forces", "nuclear_charges")


@dataclass(frozen=True)
class RMD17Arrays:
    """Validated arrays from one rMD17 NPZ partition."""

    atomic_numbers: np.ndarray
    coordinates_angstrom: np.ndarray
    energies_kcal_mol: np.ndarray
    forces_kcal_mol_angstrom: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.coordinates_angstrom.shape[0])

    @property
    def atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    """Derive a cross-process deterministic NumPy seed from public inputs."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def load_rmd17_npz(path: Path) -> RMD17Arrays:
    """Load and validate one rMD17 NPZ file without trusting its contents."""

    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_RMD17_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing rMD17 keys: {missing}")
        atomic_numbers = np.asarray(archive["nuclear_charges"], dtype=np.int64)
        coordinates = np.asarray(archive["coords"], dtype=np.float64)
        energies = np.asarray(archive["energies"], dtype=np.float64)
        forces = np.asarray(archive["forces"], dtype=np.float64)

    if atomic_numbers.ndim != 1 or atomic_numbers.size == 0 or np.any(atomic_numbers <= 0):
        raise ValueError(f"{path} has invalid nuclear_charges")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (atomic_numbers.size, 3):
        raise ValueError(f"{path} has incompatible coords shape {coordinates.shape}")
    if energies.shape != (coordinates.shape[0],):
        raise ValueError(f"{path} has incompatible energies shape {energies.shape}")
    if forces.shape != coordinates.shape:
        raise ValueError(f"{path} has incompatible forces shape {forces.shape}")
    if not all(np.isfinite(value).all() for value in (coordinates, energies, forces)):
        raise ValueError(f"{path} contains non-finite coordinates, energies, or forces")
    return RMD17Arrays(atomic_numbers, coordinates, energies, forces)


def deterministic_partition(
    frame_count: int,
    train_pool_size: int,
    validation_size: int,
    reserve_size: int,
    seed: int,
) -> Dict[str, List[int]]:
    """Create a fixed disjoint partition of one source training split."""

    if frame_count != train_pool_size + validation_size + reserve_size:
        raise ValueError(
            "Source training-frame count must equal train_pool_size + validation_size + reserve_size"
        )
    order = np.random.default_rng(seed).permutation(frame_count)
    boundaries = (train_pool_size, train_pool_size + validation_size)
    partition = {
        "train_pool": sorted(int(index) for index in order[: boundaries[0]]),
        "valid": sorted(int(index) for index in order[boundaries[0] : boundaries[1]]),
        "reserve": sorted(int(index) for index in order[boundaries[1] :]),
    }
    if len(set().union(*(set(value) for value in partition.values()))) != frame_count:
        raise AssertionError("Partition indices overlap")
    return partition


def geometry_features(coordinates_angstrom: np.ndarray) -> np.ndarray:
    """Return rigid-motion-invariant pair-distance features without labels."""

    if coordinates_angstrom.ndim != 3 or coordinates_angstrom.shape[2] != 3:
        raise ValueError("Expected coordinates with shape (frames, atoms, 3)")
    atoms = coordinates_angstrom.shape[1]
    upper = np.triu_indices(atoms, k=1)
    deltas = coordinates_angstrom[:, upper[0], :] - coordinates_angstrom[:, upper[1], :]
    return np.linalg.norm(deltas, axis=2)


def select_random(candidate_indices: Sequence[int], budget: int, seed: int) -> List[int]:
    """Select a deterministic random label batch from the permitted pool."""

    if budget <= 0 or budget > len(candidate_indices):
        raise ValueError(f"Invalid label budget {budget} for pool size {len(candidate_indices)}")
    chosen = np.random.default_rng(seed).choice(np.asarray(candidate_indices), size=budget, replace=False)
    return [int(index) for index in chosen]


def select_geometry_coverage(
    coordinates_angstrom: np.ndarray,
    candidate_indices: Sequence[int],
    budget: int,
) -> List[int]:
    """Use deterministic farthest-point coverage over coordinate-only features."""

    if budget <= 0 or budget > len(candidate_indices):
        raise ValueError(f"Invalid label budget {budget} for pool size {len(candidate_indices)}")
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    features = geometry_features(coordinates_angstrom[candidates])
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-12] = 1.0
    features = (features - center) / scale

    first = int(np.argmax(np.sum(features**2, axis=1)))
    chosen_positions = [first]
    nearest_squared_distance = np.sum((features - features[first]) ** 2, axis=1)
    nearest_squared_distance[first] = -np.inf
    while len(chosen_positions) < budget:
        next_position = int(np.argmax(nearest_squared_distance))
        chosen_positions.append(next_position)
        next_distance = np.sum((features - features[next_position]) ** 2, axis=1)
        nearest_squared_distance = np.minimum(nearest_squared_distance, next_distance)
        nearest_squared_distance[chosen_positions] = -np.inf
    return [int(candidates[position]) for position in chosen_positions]


def _atoms_for_indices(
    arrays: RMD17Arrays,
    indices: Iterable[int],
    molecule: str,
    source_partition: str,
):
    """Construct label-bearing non-periodic ASE structures for an audited subset."""

    from ase import Atoms

    records = []
    for index in indices:
        atoms = Atoms(
            numbers=arrays.atomic_numbers,
            positions=arrays.coordinates_angstrom[index],
            pbc=False,
        )
        atoms.info["REF_energy"] = float(arrays.energies_kcal_mol[index] * KCAL_MOL_TO_EV)
        atoms.info["rmd17_record_id"] = f"rmd17:{molecule}:{source_partition}:{int(index)}"
        atoms.info["config_type"] = f"rmd17_{molecule}_{source_partition}"
        atoms.arrays["REF_forces"] = (
            arrays.forces_kcal_mol_angstrom[index] * KCAL_MOL_TO_EV
        )
        records.append(atoms)
    return records


def write_extxyz(
    path: Path,
    arrays: RMD17Arrays,
    indices: Sequence[int],
    molecule: str,
    source_partition: str,
) -> str:
    """Write a labeled extxyz file and return its SHA256 digest."""

    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(
        path,
        _atoms_for_indices(arrays, indices, molecule, source_partition),
        format="extxyz",
    )
    return sha256(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a deterministic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checked_source(path: Path, expected_sha256: str) -> Tuple[RMD17Arrays, str]:
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Raw source hash mismatch for {path.name}: expected {expected_sha256}, got {observed_sha256}"
        )
    return load_rmd17_npz(path), observed_sha256


def build_rmd17_dataset(
    config: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    config_sha256: str,
) -> Dict[str, Any]:
    """Materialize the configured rMD17 benchmark in a new output directory.

    The output directory must not already exist.  This prevents accidental
    replacement of a frozen benchmark artifact or a completed experiment.
    """

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_root}")
    raw_root = raw_root.resolve()

    split_config = config["data_protocol"]
    budgets = [int(value) for value in split_config["label_budgets"]]
    strategies = list(split_config["selection_strategies"])
    unknown_strategies = set(strategies).difference({"random", "geometry_coverage"})
    if unknown_strategies:
        raise ValueError(f"Unsupported label-blind selection strategies: {sorted(unknown_strategies)}")

    # Validate every source before creating an output directory. A hash failure
    # must not leave a partially materialized data package that looks usable.
    preflight: Dict[str, Tuple[RMD17Arrays, RMD17Arrays, str, str]] = {}
    for molecule, source in config["raw_files"].items():
        train_path = raw_root / source["train"]["filename"]
        test_path = raw_root / source["test"]["filename"]
        train, train_sha256 = _checked_source(train_path, source["train"]["sha256"])
        test, test_sha256 = _checked_source(test_path, source["test"]["sha256"])
        if not np.array_equal(train.atomic_numbers, test.atomic_numbers):
            raise ValueError(f"Atomic numbers differ between rMD17 train and test partitions for {molecule}")
        preflight[molecule] = (train, test, train_sha256, test_sha256)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()

    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-rmd17-dataset-v1",
        "benchmark_id": config["benchmark_id"],
        "config_sha256": config_sha256,
        "source_dataset": config["source_dataset"],
        "unit_conversion": {
            "input_energy": "kcal/mol",
            "input_forces": "kcal/(mol Angstrom)",
            "output_energy": "eV",
            "output_forces": "eV/Angstrom",
            "kcal_mol_to_eV": KCAL_MOL_TO_EV,
        },
        "data_protocol": {
            "split_seed": int(split_config["split_seed"]),
            "train_pool_size": int(split_config["train_pool_size"]),
            "validation_size": int(split_config["validation_size"]),
            "reserve_size": int(split_config["reserve_size"]),
            "label_budgets": budgets,
            "selection_strategies": strategies,
            "selection_rule": "coordinates_and_atomic_numbers_only",
        },
        "molecules": {},
    }

    for molecule, source in config["raw_files"].items():
        train_path = raw_root / source["train"]["filename"]
        test_path = raw_root / source["test"]["filename"]
        train, test, train_sha256, test_sha256 = preflight[molecule]
        partition = deterministic_partition(
            train.frames,
            int(split_config["train_pool_size"]),
            int(split_config["validation_size"]),
            int(split_config["reserve_size"]),
            stable_seed(split_config["split_seed"], molecule, "partition"),
        )
        molecule_root = output_root / molecule
        valid_file = molecule_root / "valid.xyz"
        test_file = molecule_root / "test.xyz"
        files: Dict[str, Dict[str, Any]] = {
            "valid": {
                "path": str(valid_file.relative_to(output_root)),
                "sha256": write_extxyz(valid_file, train, partition["valid"], molecule, "train_01"),
                "frames": len(partition["valid"]),
            },
            "test": {
                "path": str(test_file.relative_to(output_root)),
                "sha256": write_extxyz(
                    test_file, test, list(range(test.frames)), molecule, "test_01"
                ),
                "frames": test.frames,
            },
        }
        selections: Dict[str, Dict[str, Any]] = {}
        for strategy in strategies:
            strategy_entries: Dict[str, Any] = {}
            for budget in budgets:
                if strategy == "random":
                    selected_indices = select_random(
                        partition["train_pool"],
                        budget,
                        stable_seed(split_config["split_seed"], molecule, strategy, budget),
                    )
                else:
                    selected_indices = select_geometry_coverage(
                        train.coordinates_angstrom, partition["train_pool"], budget
                    )
                selection_file = molecule_root / "selections" / f"{strategy}_{budget}.xyz"
                strategy_entries[str(budget)] = {
                    "path": str(selection_file.relative_to(output_root)),
                    "sha256": write_extxyz(
                        selection_file, train, selected_indices, molecule, "train_01"
                    ),
                    "frames": len(selected_indices),
                    "source_train_indices": selected_indices,
                }
            selections[strategy] = strategy_entries
        manifest["molecules"][molecule] = {
            "role": config["molecule_roles"][molecule],
            "atomic_numbers": [int(value) for value in train.atomic_numbers],
            "atoms_per_frame": train.atoms,
            "raw_sources": {
                "train": {
                    "filename": train_path.name,
                    "sha256": train_sha256,
                    "frames": train.frames,
                },
                "test": {
                    "filename": test_path.name,
                    "sha256": test_sha256,
                    "frames": test.frames,
                },
            },
            "source_train_partition_indices": partition,
            "files": files,
            "selections": selections,
        }

    manifest_path = output_root / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    manifest["dataset_manifest_sha256"] = sha256(manifest_path)
    return manifest


def regression_metrics(
    reference_energies_ev: np.ndarray,
    predicted_energies_ev: np.ndarray,
    atom_counts: np.ndarray,
    reference_forces_ev_angstrom: np.ndarray,
    predicted_forces_ev_angstrom: np.ndarray,
) -> Dict[str, float]:
    """Compute fixed independent-evaluator metrics for a collection of frames."""

    reference_energies_ev = np.asarray(reference_energies_ev, dtype=np.float64)
    predicted_energies_ev = np.asarray(predicted_energies_ev, dtype=np.float64)
    atom_counts = np.asarray(atom_counts, dtype=np.float64)
    reference_forces_ev_angstrom = np.asarray(reference_forces_ev_angstrom, dtype=np.float64)
    predicted_forces_ev_angstrom = np.asarray(predicted_forces_ev_angstrom, dtype=np.float64)
    if reference_energies_ev.shape != predicted_energies_ev.shape or reference_energies_ev.ndim != 1:
        raise ValueError("Energy arrays must be one-dimensional and identically shaped")
    if atom_counts.shape != reference_energies_ev.shape or np.any(atom_counts <= 0):
        raise ValueError("atom_counts must be positive and aligned with energy arrays")
    if reference_forces_ev_angstrom.shape != predicted_forces_ev_angstrom.shape:
        raise ValueError("Force arrays must be identically shaped")
    if reference_forces_ev_angstrom.shape[0] != reference_energies_ev.shape[0]:
        raise ValueError("Force and energy frame counts must agree")
    values = (
        reference_energies_ev,
        predicted_energies_ev,
        reference_forces_ev_angstrom,
        predicted_forces_ev_angstrom,
    )
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Cannot summarize non-finite predictions or labels")
    energy_error = predicted_energies_ev - reference_energies_ev
    energy_per_atom_error = energy_error / atom_counts
    force_error = predicted_forces_ev_angstrom - reference_forces_ev_angstrom
    return {
        "energy_mae_eV": float(np.mean(np.abs(energy_error))),
        "energy_rmse_eV": float(np.sqrt(np.mean(energy_error**2))),
        "energy_mae_meV_per_atom": float(1000.0 * np.mean(np.abs(energy_per_atom_error))),
        "energy_rmse_meV_per_atom": float(1000.0 * np.sqrt(np.mean(energy_per_atom_error**2))),
        "force_component_mae_meV_per_A": float(1000.0 * np.mean(np.abs(force_error))),
        "force_component_rmse_meV_per_A": float(1000.0 * np.sqrt(np.mean(force_error**2))),
    }

"""Shared model-resolution and metric evaluation helpers for micro-MLIP receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase.io import iread

from .data import regression_metrics


def resolve_model(run_dir: Path, explicit_model: Optional[Path] = None) -> Path:
    """Resolve one saved MACE model without guessing between multiple exports."""

    if explicit_model is not None:
        if not explicit_model.is_file():
            raise FileNotFoundError(f"Requested model does not exist: {explicit_model}")
        return explicit_model
    candidates = sorted(path for path in (run_dir / "model").rglob("*.model") if path.is_file())
    if len(candidates) != 1:
        raise ValueError("Expected exactly one saved .model; supply --model when multiple models exist")
    return candidates[0]


def make_mace_calculator(model: Path, device: str, default_dtype: str):
    """Load MACE behind the known narrow PyTorch 2.2 compatibility shim."""

    import torch

    if not hasattr(torch, "compiler"):
        class CompilerNamespace:
            pass

        torch.compiler = CompilerNamespace()  # type: ignore[attr-defined]
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
    from mace.calculators import MACECalculator

    try:
        return MACECalculator(model_paths=str(model), device=device, default_dtype=default_dtype)
    except TypeError:
        return MACECalculator(model_paths=[str(model)], device=device, default_dtype=default_dtype)


def evaluate_extxyz(model: Path, data_file: Path, device: str, default_dtype: str) -> Dict[str, float]:
    """Evaluate a closed model against a named extxyz artifact."""

    calculator = make_mace_calculator(model, device, default_dtype)
    reference_energies: List[float] = []
    predicted_energies: List[float] = []
    atom_counts: List[int] = []
    reference_forces: List[np.ndarray] = []
    predicted_forces: List[np.ndarray] = []
    for atoms in iread(data_file, format="extxyz"):
        reference_energy = atoms.info.get("REF_energy")
        reference_force = atoms.arrays.get("REF_forces")
        if reference_energy is None or reference_force is None:
            raise ValueError("Evaluation extxyz is missing REF_energy or REF_forces")
        atoms.calc = calculator
        reference_energies.append(float(reference_energy))
        predicted_energies.append(float(atoms.get_potential_energy()))
        atom_counts.append(len(atoms))
        reference_forces.append(np.asarray(reference_force, dtype=np.float64))
        predicted_forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
    if not reference_energies:
        raise ValueError("Evaluation extxyz contains no frames")
    return regression_metrics(
        np.asarray(reference_energies),
        np.asarray(predicted_energies),
        np.asarray(atom_counts),
        np.stack(reference_forces),
        np.stack(predicted_forces),
    )

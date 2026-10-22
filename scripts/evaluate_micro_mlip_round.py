#!/usr/bin/env python3
"""Independently evaluate a completed micro-MLIP run on its frozen test set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from ase.io import iread

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import regression_metrics, sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_model(run_dir: Path, explicit_model: Optional[Path]) -> Path:
    if explicit_model is not None:
        if not explicit_model.is_file():
            raise FileNotFoundError(f"Requested model does not exist: {explicit_model}")
        return explicit_model
    candidates = sorted(path for path in (run_dir / "model").rglob("*.model") if path.is_file())
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one saved .model; supply --model explicitly when a run retained multiple models"
        )
    return candidates[0]


def verified_test_file(dataset_dir: Path, molecule: str) -> Path:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    try:
        entry = manifest["molecules"][molecule]["files"]["test"]
        test_path = dataset_dir / entry["path"]
        expected_hash = entry["sha256"]
    except KeyError as exc:
        raise ValueError(f"Dataset manifest is missing test data for {molecule}") from exc
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing frozen test file: {test_path}")
    if sha256(test_path) != expected_hash:
        raise ValueError(f"Frozen test file hash mismatch: {test_path}")
    return test_path


def make_mace_calculator(model: Path, device: str, default_dtype: str):
    """Import MACE after the known narrow PyTorch 2.2 compatibility shim."""

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


def evaluate(model: Path, test_file: Path, device: str, default_dtype: str) -> Dict[str, float]:
    """Predict all test frames and summarize scalar and componentwise errors."""

    calculator = make_mace_calculator(model, device, default_dtype)
    reference_energies: List[float] = []
    predicted_energies: List[float] = []
    atom_counts: List[int] = []
    reference_forces: List[np.ndarray] = []
    predicted_forces: List[np.ndarray] = []
    for atoms in iread(test_file, format="extxyz"):
        reference_energy = atoms.info.get("REF_energy")
        reference_force = atoms.arrays.get("REF_forces")
        if reference_energy is None or reference_force is None:
            raise ValueError("Test extxyz is missing REF_energy or REF_forces")
        atoms.calc = calculator
        reference_energies.append(float(reference_energy))
        predicted_energies.append(float(atoms.get_potential_energy()))
        atom_counts.append(len(atoms))
        reference_forces.append(np.asarray(reference_force, dtype=np.float64))
        predicted_forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
    if not reference_energies:
        raise ValueError("Frozen test file contains no frames")
    return regression_metrics(
        np.asarray(reference_energies),
        np.asarray(predicted_energies),
        np.asarray(atom_counts),
        np.stack(reference_forces),
        np.stack(predicted_forces),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float32")
    args = parser.parse_args()

    run_receipt = load_json(args.run_dir / "run.json")
    training_result = load_json(args.run_dir / "result.json")
    if training_result.get("status") != "TRAINING_COMPLETED_PENDING_INDEPENDENT_TEST_EVALUATION":
        raise RuntimeError("Independent evaluation requires a successful, closed training run")
    if run_receipt.get("molecule") != args.molecule:
        raise ValueError("The requested molecule does not match the immutable training receipt")
    output_path = args.run_dir / "independent_evaluation.json"
    if output_path.exists():
        print(output_path.read_text(encoding="utf-8"))
        return 0
    model = resolve_model(args.run_dir, args.model)
    test_file = verified_test_file(args.dataset_dir, args.molecule)
    started = time.monotonic()
    metrics = evaluate(model, test_file, args.device, args.default_dtype)
    receipt = {
        "schema": "micro-mlip-independent-evaluation-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "run_id": run_receipt["run_id"],
        "molecule": args.molecule,
        "model": str(model),
        "model_sha256": sha256(model),
        "test_file": str(test_file),
        "test_file_sha256": sha256(test_file),
        "device": args.device,
        "default_dtype": args.default_dtype,
        "python_version": sys.version,
        "platform": platform.platform(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "test_metrics": metrics,
    }
    write_json(output_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

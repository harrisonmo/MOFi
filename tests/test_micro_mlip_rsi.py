"""Tests for the audited, label-blind micro-MLIP RSI benchmark scaffold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.io import write

from rsi.micro_mlip.data import (
    KCAL_MOL_TO_EV,
    build_rmd17_dataset,
    regression_metrics,
    select_geometry_coverage,
    sha256,
)
from rsi.micro_mlip.pilot import development_specs, holdout_specs, policy_action, policy_hash


ROOT = Path(__file__).parents[1]
AUDIT = ROOT / "scripts" / "audit_micro_mlip_rsi.py"
ROUND = ROOT / "scripts" / "run_micro_mlip_round.py"
CPU_SMOKE = ROOT / "benchmarks" / "micro_mlip_rsi" / "cpu_smoke_20261020.json"
CPU_ADMISSION = ROOT / "configs" / "micro-mlip-rsi-cpu-admission-v1.json"
POLICY_BLOCK = ROOT / "scripts" / "record_micro_mlip_cpu_admission_block.py"
CPU_ADMISSION_DEVELOPMENT = ROOT / "benchmarks" / "micro_mlip_rsi" / "cpu_admission_development_20261020.json"


def make_npz(path: Path, frames: int, atoms: int, offset: float) -> None:
    coordinates = np.arange(frames * atoms * 3, dtype=np.float64).reshape(frames, atoms, 3)
    coordinates = coordinates / 10.0 + offset
    energies = np.linspace(-4.0, 3.0, frames, dtype=np.float64) + offset
    forces = np.full((frames, atoms, 3), offset + 0.25, dtype=np.float64)
    charges = np.array([6] + [1] * (atoms - 1), dtype=np.int64)
    np.savez(path, coords=coordinates, energies=energies, forces=forces, nuclear_charges=charges)


def fake_config(raw_root: Path) -> dict:
    raw_files = {}
    molecule_roles = {}
    for molecule, offset in (("alpha", 0.0), ("beta", 0.5)):
        train = raw_root / f"{molecule}_train.npz"
        test = raw_root / f"{molecule}_test.npz"
        make_npz(train, frames=10, atoms=3, offset=offset)
        make_npz(test, frames=4, atoms=3, offset=offset + 1.0)
        raw_files[molecule] = {
            "train": {"filename": train.name, "sha256": sha256(train)},
            "test": {"filename": test.name, "sha256": sha256(test)},
        }
        molecule_roles[molecule] = "development"
    return {
        "benchmark_id": "synthetic-micro-mlip-v1",
        "source_dataset": {"name": "synthetic-rmd17"},
        "molecule_roles": molecule_roles,
        "raw_files": raw_files,
        "data_protocol": {
            "split_seed": 19,
            "source_train_frames": 10,
            "source_test_frames": 4,
            "train_pool_size": 6,
            "validation_size": 2,
            "reserve_size": 2,
            "label_budgets": [2, 4],
            "selection_strategies": ["random", "geometry_coverage"],
        },
    }


def test_builder_and_fail_closed_audit_materialize_label_blind_splits(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = fake_config(raw_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "dataset"
    manifest = build_rmd17_dataset(config, raw_root, output_root, sha256(config_path))

    assert manifest["unit_conversion"]["kcal_mol_to_eV"] == KCAL_MOL_TO_EV
    alpha = manifest["molecules"]["alpha"]
    partition = alpha["source_train_partition_indices"]
    assert set(partition["train_pool"]).isdisjoint(partition["valid"])
    assert set(partition["train_pool"]).isdisjoint(partition["reserve"])
    assert set(partition["valid"]).isdisjoint(partition["reserve"])
    for strategy in ("random", "geometry_coverage"):
        for budget in (2, 4):
            selected = alpha["selections"][strategy][str(budget)]["source_train_indices"]
            assert len(selected) == budget
            assert set(selected).issubset(partition["train_pool"])

    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--config",
            str(config_path),
            "--dataset-root",
            str(output_root),
            "--raw-root",
            str(raw_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads((output_root / "dataset_validation.json").read_text(encoding="utf-8"))
    assert audit["valid"] is True


def test_source_hash_failure_does_not_leave_a_partial_dataset(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = fake_config(raw_root)
    config["raw_files"]["alpha"]["train"]["sha256"] = "0" * 64
    output_root = tmp_path / "must_not_exist"
    try:
        build_rmd17_dataset(config, raw_root, output_root, "synthetic")
    except ValueError as exc:
        assert "Raw source hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected raw-hash validation to fail")
    assert not output_root.exists()


def test_geometry_coverage_is_deterministic_and_does_not_accept_labels():
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )
    first = select_geometry_coverage(coordinates, [0, 1, 2, 3], 2)
    second = select_geometry_coverage(coordinates, [0, 1, 2, 3], 2)
    assert first == second
    assert set(first).issubset({0, 1, 2, 3})
    assert len(first) == 2


def write_round_dataset(root: Path) -> Path:
    molecule_root = root / "ethanol"
    selection_path = molecule_root / "selections" / "random_128.xyz"
    valid_path = molecule_root / "valid.xyz"
    selection_path.parent.mkdir(parents=True)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], pbc=False)
    atoms.info["REF_energy"] = -1.0
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    atoms.info["rmd17_record_id"] = "rmd17:ethanol:train_01:0"
    write(selection_path, [atoms], format="extxyz")
    write(valid_path, [atoms], format="extxyz")
    manifest = {
        "benchmark_id": "micro-mlip-rsi-rmd17-v1",
        "molecules": {
            "ethanol": {
                "files": {"valid": {"path": "ethanol/valid.xyz", "sha256": sha256(valid_path)}},
                "selections": {
                    "random": {
                        "128": {
                            "path": "ethanol/selections/random_128.xyz",
                            "sha256": sha256(selection_path),
                        }
                    }
                },
            }
        },
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_round_receipt_excludes_test_data_and_is_idempotent(tmp_path: Path):
    dataset = write_round_dataset(tmp_path / "dataset")
    runner = tmp_path / "runner.py"
    runner.write_text("raise SystemExit(0)\n", encoding="utf-8")
    output = tmp_path / "runs"
    command = [
        sys.executable,
        str(ROUND),
        "--dataset-dir",
        str(dataset),
        "--molecule",
        "ethanol",
        "--selection",
        "random",
        "--label-budget",
        "128",
        "--method",
        "scratch_micro",
        "--arm",
        "fixed",
        "--seed",
        "20261020",
        "--output-root",
        str(output),
        "--runner",
        str(runner),
        "--dry-run",
    ]
    first = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    run_dir = next(output.iterdir())
    receipt = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "--test_file" not in receipt["command"]
    assert receipt["inputs"]["test_policy"] == "NOT_PASSED_TO_TRAINING_COMMAND"
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["status"] == "DRY_RUN_ONLY"
    second = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr


def test_metrics_report_per_atom_energy_and_force_components():
    metrics = regression_metrics(
        reference_energies_ev=np.array([0.0, 0.0]),
        predicted_energies_ev=np.array([0.2, -0.1]),
        atom_counts=np.array([2, 1]),
        reference_forces_ev_angstrom=np.zeros((2, 1, 3)),
        predicted_forces_ev_angstrom=np.full((2, 1, 3), 0.05),
    )
    assert np.isclose(metrics["energy_mae_eV"], 0.15)
    assert np.isclose(metrics["energy_mae_meV_per_atom"], 100.0)
    assert np.isclose(metrics["force_component_mae_meV_per_A"], 50.0)




def test_cpu_admission_grid_and_holdout_policies_are_bounded_and_hashed():
    admission = json.loads(CPU_ADMISSION.read_text(encoding="utf-8"))
    development = development_specs(admission)
    assert len(development) == admission["budget"]["development_training_rounds"] == 8
    assert {row["molecule"] for row in development} == {"ethanol", "malonaldehyde"}
    assert {row["arm"] for row in development} == {"development_calibration"}

    actions = {
        "fixed": {"selection": "random", "label_budget": 128, "method": "scratch_micro"},
        "outer_generated": {"selection": "geometry_coverage", "label_budget": 256, "method": "scratch_micro"},
        "manual_control": {"selection": "geometry_coverage", "label_budget": 128, "method": "scratch_micro"},
        "removed": {"selection": "random", "label_budget": 256, "method": "scratch_micro"},
    }
    policies = {
        "arms": {
            arm: {"action": action, "action_sha256": policy_hash(action)}
            for arm, action in actions.items()
        }
    }
    holdout = holdout_specs(admission, policies)
    assert len(holdout) == admission["budget"]["holdout_training_rounds"] == 4
    assert {row["arm"] for row in holdout} == set(actions)
    assert {row["molecule"] for row in holdout} == {"toluene"}
    assert policy_action(actions["fixed"]) == actions["fixed"]
    try:
        policy_action({"selection": "energy_stratified", "label_budget": 128, "method": "scratch_micro"})
    except ValueError:
        pass
    else:
        raise AssertionError("An unregistered label-aware selection policy must be rejected")


def test_policy_block_script_refuses_to_block_a_successful_outer_receipt(tmp_path: Path):
    campaign = tmp_path / "campaign"
    policy_root = tmp_path / "policies"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps({"status": "DEVELOPMENT_COMPLETED_PENDING_FROZEN_POLICIES"}), encoding="utf-8"
    )
    (policy_root / "outer_generated").mkdir(parents=True)
    (policy_root / "outer_generated" / "result.json").write_text(
        json.dumps({"status": "COMPLETED", "exit_code": 0}), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(POLICY_BLOCK),
            "--campaign-root",
            str(campaign),
            "--policy-root",
            str(policy_root),
            "--reason",
            "synthetic",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (campaign / "policy_generation_failure.json").exists()

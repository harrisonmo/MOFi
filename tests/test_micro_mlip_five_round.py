"""Regression tests for the five-round validation-only MLIP controller contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.generation_evidence import (
    sanitize_generation_evidence,
    validation_only_document,
)
from rsi.micro_mlip.five_round import (
    RESIDUAL_CLASSES,
    child_prefix,
    choose_action,
    conditional_policy,
    policy_hash,
    raw_threshold_policy,
    raw_validation_context,
    residual_signature,
    static_policy,
)
from rsi.micro_mlip.five_round_data import build_five_round_dataset
from rsi.micro_mlip.factorial import (
    GENERATOR,
    full_scratch_pdtw_cells,
    full_scratch_pdtw_registry,
    registry,
    resolution_v_cells,
)
from rsi.micro_mlip.five_axis import (
    assert_policy_matches_cell,
    manual_cell_policy,
    raw_policy_from_response,
)
from scripts.audit_micro_mlip_five_round import validate
from scripts.analyze_micro_mlip_full_pdtw_program import (
    factorial_effect_rows,
    generation_evolution_rows,
    residual_action_transitions,
)
from scripts.run_micro_mlip_foundation_method_admission import (
    admitted_config,
    pretraining_forgetting_metrics,
    staged_config,
)
from scripts.propose_micro_mlip_five_axis_cell_policies import response_policy
from scripts.propose_micro_mlip_five_round_policies import safe_development_evidence
from scripts.collect_micro_mlip_generation_evidence import collect_documents
from scripts.run_micro_mlip_five_round_campaign import (
    ARMS,
    assert_policy_method,
    assert_policy_binding,
    assert_resource_gate,
    initial_episode_state,
    release_group,
    state_after_round,
    validated_policies,
)
from scripts.run_micro_mlip_five_axis_screen import runnable_cells


ROOT = Path(__file__).parents[1]
FIVE_ROUND_RUNNER = ROOT / "scripts" / "run_micro_mlip_five_round.py"
POLICY_BINDER = ROOT / "scripts" / "bind_micro_mlip_five_round_policies.py"
RESOURCE_GATE = ROOT / "scripts" / "check_micro_mlip_resource_gate.py"
ANALYZER = ROOT / "scripts" / "analyze_micro_mlip_five_round_campaign.py"
FIVE_AXIS_AUDITOR = ROOT / "scripts" / "audit_micro_mlip_five_axis_policies.py"
FIVE_AXIS_CONFIG = ROOT / "configs" / "micro-mlip-five-axis-screen-v1.json"
FIVE_AXIS_REGISTRY = ROOT / "benchmarks" / "micro_mlip_rsi" / "five_axis_resolution_v2_registry_20261021.json"
FIVE_AXIS_POLICY_ROOT = ROOT / "benchmarks" / "micro_mlip_rsi" / "five_axis_policies_g0_20261021"
PDTW_SCREEN_CONFIG = ROOT / "configs" / "micro-mlip-pdtw-full-scratch-screen-v1.json"
PDTW_REGISTRY = ROOT / "benchmarks" / "micro_mlip_rsi" / "pdtw_full_scratch_registry_20261021.json"
PDTW_POLICY_ROOT = ROOT / "benchmarks" / "micro_mlip_rsi" / "pdtw_full_policies_g0_20261021"
PDTW_POLICY_AUDIT = ROOT / "benchmarks" / "micro_mlip_rsi" / "pdtw_full_policy_audit_v2_20261021.json"
PDTW_PROMOTION = ROOT / "scripts" / "select_micro_mlip_pdtw_promotion.py"
FIVE_ROUND_CONFIG = ROOT / "configs" / "micro-mlip-rsi-five-round-v1.json"
METHOD_DERIVER = ROOT / "scripts" / "derive_micro_mlip_method_policy_manifest.py"


def make_npz(path: Path, frames: int, atoms: int, offset: float) -> None:
    coordinates = np.arange(frames * atoms * 3, dtype=np.float64).reshape(frames, atoms, 3)
    coordinates = coordinates / 10.0 + offset
    energies = np.linspace(-4.0, 3.0, frames, dtype=np.float64) + offset
    forces = np.full((frames, atoms, 3), offset + 0.25, dtype=np.float64)
    charges = np.array([6] + [1] * (atoms - 1), dtype=np.int64)
    np.savez(path, coords=coordinates, energies=energies, forces=forces, nuclear_charges=charges)


def five_round_config(raw_root: Path) -> dict:
    train = raw_root / "alpha_train.npz"
    test = raw_root / "alpha_test.npz"
    make_npz(train, frames=20, atoms=3, offset=0.0)
    make_npz(test, frames=4, atoms=3, offset=1.0)
    return {
        "benchmark_id": "synthetic-micro-mlip-five-round-v1",
        "parent_benchmark": {
            "data_protocol": {
                "split_seed": 19,
                "source_train_frames": 20,
                "source_test_frames": 4,
                "train_pool_size": 10,
                "validation_size": 5,
                "reserve_size": 5,
            }
        },
        "molecule_roles": {"alpha": "development"},
        "raw_files": {
            "alpha": {
                "train": {"filename": train.name, "sha256": sha256(train)},
                "test": {"filename": test.name, "sha256": sha256(test)},
            }
        },
        "five_round_data_protocol": {
            "round_count": 5,
            "adaptive_rounds": [2, 3, 4, 5],
            "initial_labels": 2,
            "labels_added_per_adaptive_round": 1,
            "final_labels": 6,
            "selection_seed": 23,
            "selection_strategies": ["random", "geometry_coverage"],
        },
        "scientific_environment": {
            "training": {
                "max_num_epochs": 1,
                "early_stopping_patience": 1,
                "batch_size": 1,
                "valid_batch_size": 1,
                "default_dtype": "float32",
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "loss": "weighted",
                "energy_weight": 1.0,
                "forces_weight": 10.0,
                "seeds": [7],
            },
            "methods": {
                "scratch_micro": {
                    "status": "READY_FOR_CPU_ADMISSION",
                    "foundation": False,
                    "e0s": "average",
                    "r_max_A": 5.0,
                    "num_interactions": 2,
                    "hidden_irreps": "4x0e + 4x1o",
                    "lr": 0.01,
                    "ema": False,
                    "clip_grad": 1.0,
                }
            },
        },
        "validation_residual_diagnostic": {
            "energy_mae_threshold_meV_per_atom": 10.0,
            "force_mae_threshold_meV_per_A": 115.0,
            "force_tail_ratio_threshold": 1.45,
            "relative_improvement_floor": 0.03,
        },
    }


def test_five_round_data_tree_is_label_blind_and_auditable(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / "dataset"
    manifest = build_five_round_dataset(config, raw_root, dataset, sha256(config_path))

    prefixes = manifest["molecules"]["alpha"]["episodes"]["prefixes"]
    assert len(prefixes) == 31
    assert prefixes["initial"]["frames"] == 2
    leaf = child_prefix(child_prefix(child_prefix(child_prefix("initial", "random"), "random"), "random"), "random")
    assert prefixes[leaf]["frames"] == 6
    assert len(prefixes[leaf]["source_train_indices"]) == 6
    assert set(prefixes[leaf]["source_train_indices"]).issubset(
        set(manifest["molecules"]["alpha"]["source_train_partition_indices"]["train_pool"])
    )
    audited = validate(config_path, dataset, raw_root)
    assert audited["valid"], audited["errors"]


def test_five_round_runner_dry_run_uses_selected_prefix_without_test_file(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / "dataset"
    build_five_round_dataset(config, raw_root, dataset, sha256(config_path))
    audit = validate(config_path, dataset, raw_root)
    (dataset / "dataset_validation.json").write_text(json.dumps(audit), encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("raise SystemExit(0)\n", encoding="utf-8")
    output = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(FIVE_ROUND_RUNNER),
            "--config", str(config_path),
            "--dataset-dir", str(dataset),
            "--molecule", "alpha",
            "--prefix", "initial",
            "--round-index", "1",
            "--method", "scratch_micro",
            "--arm", "fixed",
            "--seed", "7",
            "--output-root", str(output),
            "--runner", str(runner),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_directory = next(output.iterdir())
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert run["label_budget"] == 2
    assert run["selection_prefix"] == "initial"
    assert "--test_file" not in run["command"]
    assert run["inputs"]["test_policy"] == "NOT_PASSED_TO_TRAINING_COMMAND"


def test_method_specific_policy_and_multihead_dry_run_require_pinned_inputs(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config["scientific_environment"]["methods"]["multihead_pseudolabel_replay"] = {
        "status": "READY_AFTER_FOUNDATION_AND_REPLAY_ADMISSION",
        "foundation": True,
        "multiheads_finetuning": True,
        "e0s": "estimated",
        "lr": 0.0001,
        "ema": True,
        "ema_decay": 0.99999,
        "clip_grad": 1.0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / "dataset"
    build_five_round_dataset(config, raw_root, dataset, sha256(config_path))
    audit = validate(config_path, dataset, raw_root)
    (dataset / "dataset_validation.json").write_text(json.dumps(audit), encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("raise SystemExit(0)\n", encoding="utf-8")
    foundation = tmp_path / "foundation.model"
    replay = tmp_path / "replay.xyz"
    foundation.write_text("pinned foundation placeholder\n", encoding="utf-8")
    replay.write_text("pinned replay placeholder\n", encoding="utf-8")
    output = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(FIVE_ROUND_RUNNER),
            "--config", str(config_path),
            "--dataset-dir", str(dataset),
            "--molecule", "alpha",
            "--prefix", "initial",
            "--round-index", "1",
            "--method", "multihead_pseudolabel_replay",
            "--arm", "fixed",
            "--seed", "7",
            "--output-root", str(output),
            "--runner", str(runner),
            "--foundation-model", str(foundation),
            "--replay-file", str(replay),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = json.loads(next(output.iterdir()).joinpath("run.json").read_text(encoding="utf-8"))
    assert "--test_file" not in run["command"]
    assert run["command"][run["command"].index("--multiheads_finetuning") + 1] == "true"
    assert "--pseudolabel_replay" in run["command"]
    assert run["command"][run["command"].index("--pt_valid_file") + 1] == str(replay)
    assert run["source_sha256"]["foundation_model"] == sha256(foundation)
    assert run["source_sha256"]["replay_file"] == sha256(replay)
    policy = static_policy(
        {2: "random", 3: "random", 4: "random", 5: "random"},
        method="multihead_pseudolabel_replay",
    )
    assert_policy_method(
        {arm: {"method": "multihead_pseudolabel_replay"} for arm in ARMS},
        "multihead_pseudolabel_replay",
    )
    assert policy["method"] == "multihead_pseudolabel_replay"


def test_foundation_method_admission_stages_distinct_replay_validation_and_forgetting(tmp_path: Path):
    base = {
        "scientific_environment": {
            "training": {"max_num_epochs": 20, "early_stopping_patience": 5},
            "methods": {
                "scratch_micro": {"status": "READY", "foundation": False},
                "foundation_naive": {"status": "BLOCKED", "foundation": True, "e0s": "estimated", "lr": 0.001, "ema": True, "ema_decay": 0.999, "clip_grad": 1.0},
                "lora_rank4": {"status": "BLOCKED", "foundation": True, "e0s": "estimated", "lr": 0.005, "ema": True, "ema_decay": 0.995, "clip_grad": 10.0, "lora_rank": 4, "lora_alpha": 1.0},
                "layer_freeze5": {"status": "BLOCKED", "foundation": True, "e0s": "estimated", "lr": 0.001, "ema": True, "ema_decay": 0.999, "clip_grad": 1.0, "freeze": 5},
            },
        }
    }
    staged = staged_config(
        base,
        foundation_hash="a" * 64,
        replay_hash="b" * 64,
        replay_validation_hash="c" * 64,
        dry_run_epochs=1,
    )
    multihead = staged["scientific_environment"]["methods"]["multihead_pseudolabel_replay"]
    assert multihead["replay_validation_required"] is True
    assert multihead["measure_pretraining_forgetting"] is True
    assert staged["scientific_environment"]["training"]["max_num_epochs"] == 1
    assert "multihead_pseudolabel_replay" not in base["scientific_environment"]["methods"]
    admitted = admitted_config(
        base,
        foundation_hash="a" * 64,
        replay_hash="b" * 64,
        replay_validation_hash="c" * 64,
        admission_id="synthetic-admission",
    )
    assert admitted["scientific_environment"]["methods"]["lora_rank4"]["status"].startswith("READY")

    results = tmp_path / "run" / "results"
    results.mkdir(parents=True)
    records = [
        {"mode": "eval", "head": "pt_head", "epoch": None, "mae_e_per_atom": 0.1, "rmse_e_per_atom": 0.2, "mae_f": 0.3, "rmse_f": 0.4},
        {"mode": "eval", "head": "pt_head", "epoch": 0, "mae_e_per_atom": 0.12, "rmse_e_per_atom": 0.24, "mae_f": 0.33, "rmse_f": 0.45},
    ]
    (results / "synthetic_train.txt").write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    forgetting = pretraining_forgetting_metrics(results.parent)
    assert forgetting["final_minus_initial"]["rmse_f_final_minus_initial"] == pytest.approx(0.05)


def test_validation_residual_route_is_typed_and_removed_route_cannot_branch():
    diagnostic = {
        "energy_mae_threshold_meV_per_atom": 10.0,
        "force_mae_threshold_meV_per_A": 115.0,
        "force_tail_ratio_threshold": 1.45,
        "relative_improvement_floor": 0.03,
    }
    signature = residual_signature(
        {
            "energy_mae_meV_per_atom": 8.0,
            "force_component_mae_meV_per_A": 170.0,
            "force_component_rmse_meV_per_A": 280.0,
        },
        None,
        diagnostic,
    )
    assert signature["residual_class"] == "force_tail_or_undercoverage"
    conditional = conditional_policy(
        {
            round_index: {
                residual: "geometry_coverage" if "force" in residual else "random"
                for residual in RESIDUAL_CLASSES
            }
            for round_index in (2, 3, 4, 5)
        },
        "random",
    )
    assert choose_action(conditional, 2, signature)["selection"] == "geometry_coverage"
    removed = static_policy({2: "random", 3: "random", 4: "random", 5: "random"}, route="removed")
    assert choose_action(removed, 2, signature)["selection"] == "random"
    assert len(policy_hash(conditional)) == 64


def test_raw_validation_tool_route_has_no_residual_class_and_uses_a_frozen_threshold():
    diagnostic = {
        "energy_mae_threshold_meV_per_atom": 10.0,
        "force_mae_threshold_meV_per_A": 115.0,
        "force_tail_ratio_threshold": 1.45,
        "relative_improvement_floor": 0.03,
    }
    metrics = {
        "energy_mae_meV_per_atom": 8.0,
        "force_component_mae_meV_per_A": 170.0,
        "force_component_rmse_meV_per_A": 280.0,
    }
    context = raw_validation_context(metrics, None)
    assert "residual_class" not in context
    rules = {
        round_index: {
            "metric": "force_component_mae_meV_per_A",
            "comparison": "gte",
            "threshold": 115.0,
            "if_true": "geometry_coverage",
            "if_false": "random",
            "if_missing": "random",
        }
        for round_index in (2, 3, 4, 5)
    }
    policy = raw_threshold_policy(rules)
    choice = choose_action(policy, 2, context)
    assert choice["selection"] == "geometry_coverage"
    assert choice["residual_class"] is None
    assert choice["decision_source"] == "frozen_raw_threshold_rule"
    state = initial_episode_state("alpha", 7, "manual_control", policy)
    state_after_round(
        {"validation_residual_diagnostic": diagnostic},
        state,
        {
            "round_index": 1,
            "prefix": "initial",
            "run_id": "alpha__seed7__manual__round1__initial",
            "run_directory": "/synthetic/run",
            "validation_metrics": metrics,
            "status": "COMPLETED",
        },
    )
    assert state["decisions"][0]["validation_context"]["schema"] == "micro-mlip-raw-validation-v1"
    assert "residual_class" not in state["decisions"][0]["validation_context"]


def test_five_axis_cell_contract_distinguishes_raw_and_typed_tools():
    diagnostic = {
        "energy_mae_threshold_meV_per_atom": 10.0,
        "force_mae_threshold_meV_per_A": 115.0,
        "force_tail_ratio_threshold": 1.45,
        "relative_improvement_floor": 0.03,
    }
    raw_levels = {"prompt": "low", "data": "high", "tool": "low", "workflow": "high", "model": "low"}
    raw_policy = manual_cell_policy(raw_levels, diagnostic)
    assert raw_policy["diagnostic_route"] == "raw_validation_v1"
    assert_policy_matches_cell(raw_policy, raw_levels)
    typed_levels = {**raw_levels, "tool": "high"}
    typed_policy = manual_cell_policy(typed_levels, diagnostic)
    assert typed_policy["diagnostic_route"] == "validation_residual_v1"
    assert_policy_matches_cell(typed_policy, typed_levels)
    response = {
        str(round_index): {
            "condition_id": "force_mae_high",
            "if_true": "geometry_coverage",
            "if_false": "random",
            "if_missing": "random",
        }
        for round_index in (2, 3, 4, 5)
    }
    generated_raw = raw_policy_from_response(response, diagnostic)
    assert_policy_matches_cell(generated_raw, raw_levels)
    with pytest.raises(ValueError, match="excluded by its Data level"):
        assert_policy_matches_cell(generated_raw, {**raw_levels, "data": "low"})


def test_cell_policy_compiler_enforces_the_registered_workflow_interface():
    diagnostic = {
        "energy_mae_threshold_meV_per_atom": 10.0,
        "force_mae_threshold_meV_per_A": 115.0,
        "force_tail_ratio_threshold": 1.45,
        "relative_improvement_floor": 0.03,
    }
    typed_levels = {"prompt": "high", "data": "high", "tool": "high", "workflow": "high", "model": "low"}
    typed_response = {
        "rationale": "Use coverage for force-oriented operational patterns.",
        "rounds": {
            str(round_index): {
                "default_action": "random",
                "by_residual_class": {
                    residual: "geometry_coverage" if "force" in residual else "random"
                    for residual in RESIDUAL_CLASSES
                },
            }
            for round_index in (2, 3, 4, 5)
        },
    }
    typed = response_policy(typed_response, typed_levels, diagnostic)
    assert typed["diagnostic_route"] == "validation_residual_v1"
    raw_levels = {**typed_levels, "tool": "low", "prompt": "low"}
    raw_response = {
        "rationale": "Use a raw force threshold only.",
        "rounds": {
            str(round_index): {
                "condition_id": "force_mae_high",
                "if_true": "geometry_coverage",
                "if_false": "random",
                "if_missing": "random",
            }
            for round_index in (2, 3, 4, 5)
        },
    }
    raw = response_policy(raw_response, raw_levels, diagnostic)
    assert raw["diagnostic_route"] == "raw_validation_v1"
    static_levels = {**typed_levels, "workflow": "low"}
    static_response = {
        "rationale": "Keep a static schedule.",
        "rounds": {str(round_index): "random" for round_index in (2, 3, 4, 5)},
    }
    static = response_policy(static_response, static_levels, diagnostic)
    assert static["diagnostic_route"] == "validation_residual_v1"


def test_resolution_v_registry_has_sixteen_cells_and_never_marks_model_high_as_run_ready(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config["five_axes"] = {
        factor: {"low": f"{factor}-low", "high": f"{factor}-high"}
        for factor in ("prompt", "data", "tool", "workflow", "model")
    }
    cells = resolution_v_cells()
    assert len(cells) == 16
    assert GENERATOR == "model = -(prompt * data * tool * workflow)"
    assert {row["levels"]["model"] for row in cells} == {"low", "high"}
    result = registry(config)
    assert result["ready_cell_count"] == 8
    assert result["blocked_cell_count"] == 8
    assert all(
        row["status"] == "BLOCKED"
        for row in result["cells"]
        if row["levels"]["model"] == "high"
    )
    by_id = {row["cell_id"]: row for row in result["cells"]}
    assert by_id["P0D0T0W0M0"]["status"] == "READY"
    assert by_id["P1D1T1W1M0"]["status"] == "READY"


def test_full_scratch_pdtw_registry_contains_every_loop_layer_combination(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config["benchmark_id"] = "synthetic-full-pdtw"
    config["five_axes"] = {
        factor: {"low": f"{factor}-low", "high": f"{factor}-high"}
        for factor in ("prompt", "data", "tool", "workflow", "model")
    }
    cells = full_scratch_pdtw_cells()
    assert len(cells) == 16
    assert {cell["cell_id"] for cell in cells} == {
        f"P{p}D{d}T{t}W{w}M0"
        for p in (0, 1)
        for d in (0, 1)
        for t in (0, 1)
        for w in (0, 1)
    }
    result = full_scratch_pdtw_registry(config)
    assert result["ready_cell_count"] == 16
    assert result["blocked_cell_count"] == 0
    assert {cell["levels"]["model"] for cell in result["cells"]} == {"low"}


def test_controller_records_a_validation_decision_and_requires_four_final_arms(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    policy = static_policy({2: "geometry_coverage", 3: "random", 4: "random", 5: "random"})
    state = initial_episode_state("alpha", 7, "fixed", policy)
    state_after_round(
        config,
        state,
        {
            "round_index": 1,
            "prefix": "initial",
            "run_id": "alpha__seed7__fixed__round1__initial",
            "run_directory": str(tmp_path / "run"),
            "validation_metrics": {
                "energy_mae_meV_per_atom": 8.0,
                "force_component_mae_meV_per_A": 140.0,
                "force_component_rmse_meV_per_A": 210.0,
            },
            "status": "COMPLETED",
        },
    )
    assert state["current_prefix"] == "initial__geometry_coverage"
    assert state["decisions"][0]["decision"]["selection"] == "geometry_coverage"
    assert state["test_access"] == "NOT_OPENED"

    class Args:
        campaign_root = tmp_path / "campaign"

    states = {
        arm: {
            "status": "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER",
            "rounds": [
                {
                    "round_index": 5,
                    "status": "COMPLETED",
                    "run_id": f"alpha__seed7__{arm}__round5__final",
                    "validation_receipt_sha256": "a" * 64,
                }
            ],
        }
        for arm in ARMS
    }
    barrier = release_group(Args(), "alpha", 7, states)
    receipt = json.loads(barrier.read_text(encoding="utf-8"))
    assert receipt["status"] == "ALL_FOUR_ARMS_FINAL_TRAINING_VALIDATION_CLOSED"
    assert len(receipt["eligible_run_ids"]) == 4


def test_policy_environment_binding_prevents_unbound_campaign_start(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = five_round_config(raw_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / "dataset"
    build_five_round_dataset(config, raw_root, dataset, sha256(config_path))
    audit = validate(config_path, dataset, raw_root)
    (dataset / "dataset_validation.json").write_text(json.dumps(audit), encoding="utf-8")
    static = static_policy({2: "random", 3: "random", 4: "random", 5: "random"})
    conditional = conditional_policy(
        {round_index: {residual: "random" for residual in RESIDUAL_CLASSES} for round_index in (2, 3, 4, 5)},
        "random",
    )
    policies = {
        "schema": "micro-mlip-five-round-policies-v1",
        "arms": {
            "fixed": {"policy": static, "policy_sha256": policy_hash(static)},
            "outer_generated": {"policy": conditional, "policy_sha256": policy_hash(conditional)},
            "manual_control": {"policy": conditional, "policy_sha256": policy_hash(conditional)},
            "removed": {
                "policy": static_policy({2: "random", 3: "random", 4: "random", 5: "random"}, route="removed"),
                "policy_sha256": policy_hash(static_policy({2: "random", 3: "random", 4: "random", 5: "random"}, route="removed")),
            },
        },
    }
    policies_path = tmp_path / "policies.json"
    policies_path.write_text(json.dumps(policies), encoding="utf-8")
    binding = tmp_path / "binding.json"
    result = subprocess.run(
        [
            sys.executable,
            str(POLICY_BINDER),
            "--config", str(config_path),
            "--dataset-dir", str(dataset),
            "--policies", str(policies_path),
            "--output", str(binding),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    frozen = validated_policies(policies_path)
    assert len(assert_policy_binding(binding, config_path, dataset, policies_path, frozen, ROOT)) == 64
    config_path.write_text(json.dumps({**config, "unrelated_test_mutation": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        assert_policy_binding(binding, config_path, dataset, policies_path, frozen, ROOT)


def test_shared_resource_gate_requires_completed_formal_campaign_and_fresh_receipt(tmp_path: Path):
    formal_state = tmp_path / "formal_state.json"
    formal_state.write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    formal_done = tmp_path / "formal.done"
    formal_done.write_text("done\n", encoding="utf-8")
    gate_path = tmp_path / "resource_gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(RESOURCE_GATE),
            "--formal-launcher-pid", "999999",
            "--formal-state", str(formal_state),
            "--formal-done", str(formal_done),
            "--disk-path", str(tmp_path),
            "--minimum-free-gib", "0",
            "--output", str(gate_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(assert_resource_gate(gate_path, maximum_age_seconds=60)) == 64
    receipt = json.loads(gate_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "ADMITTED"


def test_postrelease_analysis_requires_and_preserves_a_complete_paired_trace(tmp_path: Path):
    campaign = tmp_path / "campaign"
    (campaign / "release_barriers").mkdir(parents=True)
    (campaign / "campaign_manifest.json").write_text(
        json.dumps({"status": "COMPLETED_PENDING_ANALYSIS"}), encoding="utf-8"
    )
    molecule = "alpha"
    seed = 7
    run_ids = {arm: f"{molecule}__seed{seed}__{arm}__round5" for arm in ARMS}
    barrier = campaign / "release_barriers" / f"{molecule}__seed{seed}.json"
    barrier.write_text(
        json.dumps(
            {
                "status": "ALL_FOUR_ARMS_FINAL_TRAINING_VALIDATION_CLOSED",
                "eligible_run_ids": [run_ids[arm] for arm in ARMS],
            }
        ),
        encoding="utf-8",
    )
    barrier_hash = sha256(barrier)
    arm_offsets = {"fixed": 8.0, "outer_generated": 4.0, "manual_control": 6.0, "removed": 9.0}
    for arm in ARMS:
        state_dir = campaign / "episodes" / f"{molecule}__seed{seed}__{arm}"
        state_dir.mkdir(parents=True)
        rounds = []
        decisions = []
        for round_index, budget in enumerate((2, 3, 4, 5, 6), start=1):
            run_id = run_ids[arm] if round_index == 5 else f"{molecule}__seed{seed}__{arm}__round{round_index}"
            run_dir = campaign / "runs" / run_id
            run_dir.mkdir(parents=True)
            validation_metrics = {
                "energy_mae_meV_per_atom": arm_offsets[arm] + 0.5 * (6 - round_index),
                "energy_rmse_meV_per_atom": arm_offsets[arm] + 1.0,
                "force_component_mae_meV_per_A": 100.0 + arm_offsets[arm] + 2.0 * (6 - round_index),
                "force_component_rmse_meV_per_A": 130.0 + arm_offsets[arm] + 2.0 * (6 - round_index),
            }
            run = {
                "run_id": run_id,
                "molecule": molecule,
                "arm": arm,
                "seed": seed,
                "round_index": round_index,
                "label_budget": budget,
                "selection_prefix": "initial",
                "selection_history": [],
                "inputs": {"test_policy": "NOT_PASSED_TO_TRAINING_COMMAND"},
            }
            result = {
                "status": (
                    "COMPLETED_INDEPENDENT_TEST_EVALUATION"
                    if round_index == 5
                    else "TRAINING_VALIDATED_PENDING_INDEPENDENT_TEST_EVALUATION"
                ),
                "independent_test_evaluation": "COMPLETED" if round_index == 5 else "NOT_STARTED",
            }
            validation = {
                "status": "COMPLETED",
                "run_id": run_id,
                "duration_seconds": 0.1,
                "test_access": "NOT_OPENED_OR_READ",
                "validation_metrics": validation_metrics,
            }
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (run_dir / "validation_evaluation.json").write_text(json.dumps(validation), encoding="utf-8")
            if round_index == 5:
                independent = {
                    "status": "COMPLETED",
                    "run_id": run_id,
                    "release_manifest_sha256": barrier_hash,
                    "test_metrics": validation_metrics,
                }
                (run_dir / "independent_evaluation.json").write_text(
                    json.dumps(independent), encoding="utf-8"
                )
            rounds.append(
                {
                    "round_index": round_index,
                    "status": "COMPLETED",
                    "run_id": run_id,
                    "run_directory": str(run_dir),
                    "training": {"exit_code": 0, "duration_seconds": 1.0 + round_index},
                }
            )
            if round_index > 1:
                decisions.append(
                    {
                        "before_round": round_index,
                        "decision": {"decision_source": "frozen_static_rule", "selection": "random"},
                        "validation_context": {"residual_class": "balanced_or_improving"},
                    }
                )
        state = {
            "status": "COMPLETED",
            "molecule": molecule,
            "seed": seed,
            "arm": arm,
            "rounds": rounds,
            "decisions": decisions,
        }
        (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    output = tmp_path / "analysis"
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--campaign-root", str(campaign),
            "--output-dir", str(output),
            "--generation", "synthetic-g0",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["pairing"]["status"] == "PASSED"
    assert analysis["pairing"]["raw_round_row_count"] == 20
    assert len(analysis["paired_effects"]) == 3
    outer_vs_removed = next(
        row for row in analysis["paired_effects"] if row["comparison"] == "outer_generated_minus_removed"
    )
    assert outer_vs_removed["final_test_delta_force_component_mae_meV_per_A"] < 0.0




def test_generation_evidence_is_frozen_before_release_and_safe_for_a_later_policy(tmp_path: Path):
    """A g1 prompt may use full validation curves but never independent metrics."""

    def final_state(molecule: str, seed: int, arm: str) -> dict:
        rounds = []
        decisions = []
        for round_index in range(1, 6):
            metrics = {
                "energy_mae_meV_per_atom": 12.0 - round_index,
                "force_component_mae_meV_per_A": 160.0 - 10.0 * round_index,
                "force_component_rmse_meV_per_A": 220.0 - 10.0 * round_index,
            }
            rounds.append(
                {
                    "round_index": round_index,
                    "prefix": "initial" if round_index == 1 else f"prefix-{round_index}",
                    "run_id": f"{molecule}__seed{seed}__{arm}__round{round_index}",
                    "run_directory": str(tmp_path / "runs" / arm / str(round_index)),
                    "training": {"exit_code": 0, "duration_seconds": float(round_index)},
                    "validation": {"exit_code": 0, "duration_seconds": 0.25},
                    "validation_metrics": metrics,
                    "validation_receipt_sha256": f"{round_index:x}" * 64,
                    "status": "COMPLETED",
                }
            )
            if round_index < 5:
                decisions.append(
                    {
                        "after_round": round_index,
                        "before_round": round_index + 1,
                        "source_prefix": "initial" if round_index == 1 else f"prefix-{round_index}",
                        "next_prefix": f"prefix-{round_index + 1}",
                        "validation_context": None,
                        "decision": {
                            "selection": "random",
                            "decision_source": "frozen_static_rule",
                            "residual_class": None,
                        },
                    }
                )
        return {
            "status": "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER",
            "molecule": molecule,
            "seed": seed,
            "arm": arm,
            "policy_sha256": arm[0] * 64,
            "rounds": rounds,
            "decisions": decisions,
            "independent_evaluation": {"test_metrics": {"force_component_mae_meV_per_A": 0.01}},
        }

    documents = []
    levels = {"prompt": "high", "data": "high", "tool": "high", "workflow": "high", "model": "low"}
    for molecule in ("ethanol", "malonaldehyde"):
        document = validation_only_document(
            campaign_id="synthetic-g0",
            factorial_cell_id="P1D1T1W1M0",
            factorial_levels=levels,
            molecule=molecule,
            seed=20261020,
            states={arm: final_state(molecule, 20261020, arm) for arm in ARMS},
        )
        assert "independent_evaluation" not in json.dumps(document)
        path = tmp_path / f"{molecule}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        documents.append(path)

    collected = collect_documents(documents, "g0")
    assert collected["development_molecules"] == ["ethanol", "malonaldehyde"]
    assert len(collected["episodes"]) == 8
    assert "test_metrics" not in json.dumps(collected)
    collected["episodes"][0]["rounds"][0]["test_metric"] = 999.0
    safe = sanitize_generation_evidence(collected)
    assert "test_metric" not in json.dumps(safe)

    evidence_path = tmp_path / "g0_evidence.json"
    evidence_path.write_text(json.dumps(collected), encoding="utf-8")
    proposal_input = safe_development_evidence(evidence_path)
    assert proposal_input["schema"] == "micro-mlip-five-round-generation-evidence-v1"
    assert len(proposal_input["episodes"]) == 8

    unsafe = json.loads(json.dumps(collected))
    unsafe["episodes"][0]["molecule"] = "toluene"
    with pytest.raises(ValueError, match="development molecules"):
        sanitize_generation_evidence(unsafe)




def test_full_pdtw_postrelease_contrasts_and_evolution_keep_the_pairing_axes_explicit():
    """The reporter must preserve all 16 P/D/T/W cells and generation identity."""

    rows = []
    for generation, seed, generation_offset in (("g0", 7, 0.0), ("g1", 8, -3.0)):
        for prompt in (0, 1):
            for data in (0, 1):
                for tool in (0, 1):
                    for workflow in (0, 1):
                        codes = {
                            "prompt": 1.0 if prompt else -1.0,
                            "data": 1.0 if data else -1.0,
                            "tool": 1.0 if tool else -1.0,
                            "workflow": 1.0 if workflow else -1.0,
                        }
                        cell_id = f"P{prompt}D{data}T{tool}W{workflow}M0"
                        response = (
                            generation_offset
                            + 5.0 * codes["prompt"]
                            + 2.0 * codes["data"]
                            - 1.0 * codes["tool"]
                            + 0.5 * codes["workflow"]
                        )
                        rows.append(
                            {
                                "generation": generation,
                                "molecule": "alpha",
                                "seed": seed,
                                "factorial_cell_id": cell_id,
                                "comparison": "outer_generated_minus_removed",
                                **{f"{axis}_level": "high" if code > 0 else "low" for axis, code in codes.items()},
                                "final_test_delta_force_component_mae_meV_per_A": response,
                                "final_test_delta_energy_mae_meV_per_atom": response / 10.0,
                                "validation_normalized_auc_delta_force_component_mae_meV_per_A": response * 2.0,
                                "validation_normalized_auc_delta_energy_mae_meV_per_atom": response / 5.0,
                                "cumulative_training_seconds_delta": response * 3.0,
                            }
                        )

    contrasts = factorial_effect_rows(rows)
    prompt_g0 = next(
        row
        for row in contrasts
        if row["generation"] == "g0"
        and row["response"] == "final_test_delta_force_component_mae_meV_per_A"
        and row["factor_term"] == "P"
    )
    tool_g1 = next(
        row
        for row in contrasts
        if row["generation"] == "g1"
        and row["response"] == "final_test_delta_force_component_mae_meV_per_A"
        and row["factor_term"] == "T"
    )
    assert prompt_g0["high_minus_low_contrast"] == pytest.approx(10.0)
    assert tool_g1["high_minus_low_contrast"] == pytest.approx(-2.0)
    evolution = generation_evolution_rows(rows)
    prompt_cell = next(
        row
        for row in evolution
        if row["factorial_cell_id"] == "P1D1T1W1M0"
        and row["response"] == "final_test_delta_force_component_mae_meV_per_A"
    )
    assert prompt_cell["g1_minus_g0"] == pytest.approx(-3.0)

    transitions = residual_action_transitions(
        [
            {
                "generation": "g0",
                "factorial_cell_id": "P1D1T1W1M0",
                "molecule": "alpha",
                "seed": 7,
                "arm": "outer_generated",
                "round_index": 1,
                "decision_selection": "geometry_coverage",
                "decision_source": "frozen_outer_policy",
                "residual_class": "force_tail_or_undercoverage",
                "validation_force_component_mae_meV_per_A": 120.0,
                "prompt_level": "high",
                "data_level": "high",
                "tool_level": "high",
                "workflow_level": "high",
            },
            {
                "generation": "g0",
                "factorial_cell_id": "P1D1T1W1M0",
                "molecule": "alpha",
                "seed": 7,
                "arm": "outer_generated",
                "round_index": 2,
                "decision_selection": None,
                "decision_source": None,
                "residual_class": None,
                "validation_force_component_mae_meV_per_A": 100.0,
                "prompt_level": "high",
                "data_level": "high",
                "tool_level": "high",
                "workflow_level": "high",
            },
        ]
    )
    assert transitions[0]["next_minus_current_validation_force_meV_per_A"] == pytest.approx(-20.0)
    assert "not evidence" in transitions[0]["interpretation"]

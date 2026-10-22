"""Controller binding and persistent native-tool budget tests; no model calls."""

from pathlib import Path
import subprocess
import sys

import pytest

from rsi.gcmc import core
from rsi.science.tool_registry import ScientificToolRegistry, TOOL


@pytest.fixture
def registry_inputs(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write

    write(tmp_path / "host.cif", Atoms("Si", cell=[20, 20, 20], pbc=True))
    ff = {"name": "test-only zero interaction", "source": "unit-test control",
          "framework_types": {"Si": {"element": "Si", "epsilon_k": 0., "sigma_angstrom": 1.}},
          "element_types": {"Si": "Si"}, "methane": {"epsilon_k": 0., "sigma_angstrom": 1.},
          "shifted": True, "tail_corrections": False}
    core.write_json(tmp_path / "ff.json", ff)
    config = core.read_json(Path(__file__).parents[1] / "configs/methane-gcmc-example.json")
    config.update(structure="host.cif", force_field="ff.json", seeds=[606],
                  initialization_cycles=50, production_cycles=100, cutoff_angstrom=4.,
                  budget={"max_jobs": 2, "max_total_cycles": 300, "timeout_seconds_per_job": 10,
                          "max_wall_seconds": 20, "max_supercell_atoms": 100})
    core.write_json(tmp_path / "task.json", config)
    registration = {"schema_version": 1, "tasks": {"approved": "task.json"},
                    "session_budget": {"max_calls": 1, "max_total_cycles": 300,
                                       "max_reserved_wall_seconds": 20}}
    core.write_json(tmp_path / "registry.json", registration)
    return tmp_path, registration


def mock_success(run):
    manifest = core.read_json(run / "manifest.json")
    config = core.read_json(run / "config.json")
    rows = []
    for job in manifest["jobs"]:
        job.update(status="completed", actual_cycles=150)
        rows.append({"pressure_bar": job["pressure_bar"], "replicate_seed": job["replicate_seed"],
                     "absolute_uptake_mmol_g": job["pressure_bar"], "engine_reported_error_mmol_g": .1})
    manifest["status"] = "completed"
    core.write_json(run / "manifest.json", manifest)
    summary = core.capacity_summary(rows, config)
    core.write_json(run / "capacity.json", summary)
    return summary


def test_catalog_returns_function_schema_without_running_science(registry_inputs):
    path, _ = registry_inputs
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    catalog = registry.catalog()
    function = catalog["tools"][0]["function"]
    assert function["name"] == TOOL
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["task_id"]["enum"] == ["approved"]
    assert catalog["tasks"][0]["reserved_cycles"] == 300
    assert not (path / "session").exists()


@pytest.mark.parametrize("name,args", [("unknown", {"task_id": "approved"}),
    (TOOL, {"task_id": "../../host.cif"}), (TOOL, {"task_id": "approved", "config": "other"}),
    (TOOL, {"task_id": "approved", "budget": 100000}), (TOOL, {"task_id": True}),
    (TOOL, {}), (TOOL, [])])
def test_agent_cannot_override_paths_budget_or_registration(registry_inputs, name, args):
    path, _ = registry_inputs
    result = ScientificToolRegistry(path / "registry.json", path / "session").call(name, args)
    assert not result["ok"] and result["cost"] is None
    assert not (path / "session").exists()


def test_success_dispatch_records_evidence_and_budget_survives_reopening(registry_inputs, monkeypatch):
    path, _ = registry_inputs
    monkeypatch.setattr(core, "execute", mock_success)
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    result = registry.call(TOOL, {"task_id": "approved"})
    assert result["ok"], result["error"]
    assert result["data"]["units"] == "mmol/g-dry-framework"
    assert result["cost"]["known_completed_cycles"] == 300
    assert result["cost"]["reserved_cycles"] == 300
    assert result["evidence"]["capacity_sha256"]
    assert not (path / "session/session.lock").exists()
    reopened = ScientificToolRegistry(path / "registry.json", path / "session")
    denied = reopened.call(TOOL, {"task_id": "approved"})
    assert not denied["ok"] and "budget exhausted" in denied["error"]["message"]
    ledger = core.read_json(path / "session/ledger.json")
    assert len(ledger["calls"]) == 1 and ledger["calls"][0]["status"] == "completed"


def test_changed_input_is_rejected_before_reserving_science(registry_inputs):
    path, _ = registry_inputs
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    (path / "host.cif").write_text("changed")
    result = registry.call(TOOL, {"task_id": "approved"})
    assert not result["ok"] and "input changed" in result["error"]["message"]
    assert not (path / "session").exists()


@pytest.mark.parametrize("limit", ["max_calls", "max_total_cycles", "max_reserved_wall_seconds"])
def test_all_session_limits_are_enforced(registry_inputs, monkeypatch, limit):
    path, registration = registry_inputs
    monkeypatch.setattr(core, "execute", mock_success)
    registration["session_budget"]["max_calls"] = 2
    registration["session_budget"]["max_total_cycles"] = 600
    registration["session_budget"]["max_reserved_wall_seconds"] = 40
    registration["session_budget"][limit] //= 2
    core.write_json(path / "registry.json", registration)
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    assert registry.call(TOOL, {"task_id": "approved"})["ok"]
    assert not registry.call(TOOL, {"task_id": "approved"})["ok"]


def test_session_budget_cannot_be_reset_by_editing_config(registry_inputs, monkeypatch):
    path, registration = registry_inputs
    monkeypatch.setattr(core, "execute", mock_success)
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    assert registry.call(TOOL, {"task_id": "approved"})["ok"]
    registration["session_budget"]["max_calls"] = 100
    core.write_json(path / "registry.json", registration)
    reopened = ScientificToolRegistry(path / "registry.json", path / "session")
    result = reopened.call(TOOL, {"task_id": "approved"})
    assert not result["ok"] and "cannot be reset" in result["error"]["message"]


def test_native_failure_retains_unknown_cost_and_consumes_reservation(registry_inputs, monkeypatch):
    path, _ = registry_inputs

    def fail(run):
        manifest = core.read_json(run / "manifest.json")
        manifest["status"] = "failed"
        manifest["jobs"][0].update(status="timeout", actual_cycles=None)
        core.write_json(run / "manifest.json", manifest)
        raise TimeoutError("mock native timeout")

    monkeypatch.setattr(core, "execute", fail)
    registry = ScientificToolRegistry(path / "registry.json", path / "session")
    result = registry.call(TOOL, {"task_id": "approved"})
    assert not result["ok"]
    assert result["cost"]["incomplete_jobs_with_unknown_actual_cycles"] == 1
    assert result["cost"]["reserved_cycles"] == 300
    assert result["evidence"]["manifest_sha256"] and result["evidence"]["capacity_sha256"] is None
    assert not registry.call(TOOL, {"task_id": "approved"})["ok"]


def test_stale_or_concurrent_lock_cannot_be_cleared_by_an_agent(registry_inputs):
    path, _ = registry_inputs
    (path / "session").mkdir()
    (path / "session/session.lock").write_text("other owner")
    result = ScientificToolRegistry(path / "registry.json", path / "session").call(TOOL, {"task_id": "approved"})
    assert not result["ok"]
    assert (path / "session/session.lock").read_text() == "other owner"


def test_cli_lists_registered_tool(registry_inputs):
    import json
    path, _ = registry_inputs
    completed = subprocess.run([sys.executable, "-m", "rsi.science.tools", "list", "--config",
                                str(path / "registry.json"), "--session", str(path / "session")],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["tools"][0]["function"]["name"] == TOOL
    assert not (path / "session").exists()

"""Independent contract, adversarial, invariance, provenance and pairing checks."""

import copy
import json
from pathlib import Path

import pytest

from rsi.tool_evidence import from_uff_event
from rsi.tool_runtime import Registry, digest, execute, load_spec
from rsi.tool_smoke import scripted_agent
from rsi.tool_trial import Ledger, episode, paired
from rsi.tool_validation import fixtures, validate_candidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec():
    return json.loads((ROOT / "tools/q2/manual-audit-v1.json").read_text())


def test_independent_gate_and_immutable_registry(spec):
    registry = Registry()
    record = registry.register(spec)
    assert record["validation"]["passed"]
    assert record["validation"]["total_checks"] == 26
    spec["outputs"]["energy_kj_mol"] = {"number": 0}
    assert registry.call("uff_audit", 1, fixtures()[0])["output"]["energy_kj_mol"] == pytest.approx(29.288)
    record["spec"]["outputs"]["converged"] = {"number": 1}
    with pytest.raises(ValueError):
        registry.register(spec)


def test_valid_but_wrong_candidate_cannot_register(spec):
    spec["outputs"]["delta_kj_mol"] = {"number": 0}
    assert not validate_candidate(spec)["passed"]
    with pytest.raises(ValueError, match="Independent"):
        Registry().register(spec)


@pytest.mark.parametrize("node", [
    {"op": "exec", "args": ["bad"]}, {"field": "__class__"},
    {"op": "retry", "args": []}, {"path": "../evaluator.py"},
    {"number": True}, {"number": float("inf")},
])
def test_forbidden_capabilities_rejected(spec, node):
    spec["outputs"]["energy_kj_mol"] = node
    assert not execute(spec, fixtures()[0])["ok"]


def test_depth_and_size_limits(spec):
    node = {"field": "energy"}
    for _ in range(12):
        node = {"op": "add", "args": [node, {"number": 0}]}
    spec["outputs"]["energy_kj_mol"] = node
    assert not execute(spec, fixtures()[0])["ok"]
    spec["id"] = "a" * 20000
    assert not execute(spec, fixtures()[0])["ok"]


@pytest.mark.parametrize("field,value", [("energy", float("nan")), ("max_force", -1),
    ("energy", True), ("converged", 1), ("energy_unit", "eV"),
    ("force_unit", "kJ/mol"), ("force_unit", "kJ/mol/A"), ("energy", 1e101)])
def test_bad_input_and_out_of_scope_units(spec, field, value):
    payload = dict(fixtures()[0], **{field: value})
    assert not execute(spec, payload)["ok"]


def test_known_values_and_unit_equivalence(spec):
    a, b = fixtures()[:2]
    left, right = execute(spec, a)["output"], execute(spec, b)["output"]
    assert left == pytest.approx(right)
    assert left["delta_kj_mol"] == pytest.approx(-12.552)
    assert left["max_force_kj_mol_nm"] == pytest.approx(83.68)


def test_energy_origin_and_cell_scaling(spec):
    payload = fixtures()[0]
    original = execute(spec, payload)["output"]
    shifted = dict(payload, initial_energy=1010, energy=1007)
    scaled = dict(payload, initial_energy=30, energy=21)
    assert execute(spec, shifted)["output"]["delta_kj_mol"] == original["delta_kj_mol"]
    result = execute(spec, scaled)["output"]
    assert result["delta_kj_mol"] == pytest.approx(3 * original["delta_kj_mol"])
    assert result["max_force_kj_mol_nm"] == original["max_force_kj_mol_nm"]


def test_decreasing_energy_does_not_imply_convergence(spec):
    payload = dict(fixtures()[0], converged=False)
    assert execute(spec, payload)["output"]["converged"] is False


def test_pair_freezes_inputs_and_removal(spec):
    seen = []
    def factory():
        agent = scripted_agent(spec)
        def decide(stage, context):
            seen.append((stage, copy.deepcopy(context)))
            return agent(stage, context)
        return decide
    rows = paired(fixtures()[0], spec, factory)
    assert all(row["passed"] for row in rows)
    assert len({row["input_hash"] for row in rows}) == 1
    assert [row["generated_or_manual_tool_uses"] for row in rows] == [0, 1, 0]
    assert [row["cost"]["tool_calls"] for row in rows] == [4, 1, 4]
    assert seen[0] == seen[4]
    assert all("expect" not in context and "reference" not in context for _, context in seen)


def test_unknown_usage_stops_and_preserves_cost(spec):
    calls = []
    def decide(stage, context):
        calls.append(stage)
        return dict(response=[], calls=1, tokens=None, wall_seconds=0.1)
    row = episode(fixtures()[0], spec, "fixed", decide, Ledger())
    assert calls == ["plan"]
    assert row["cost"]["tokens"] is None
    assert row["cost"]["calls"] == 1
    assert not row["passed"]


def test_removed_tool_cannot_run(spec):
    def decide(stage, context):
        return dict(response=[{"tool": "uff_audit", "arguments": {}}] if stage == "plan" else {},
                    calls=0, tokens=0, wall_seconds=0)
    row = episode(fixtures()[0], spec, "removed", decide, Ledger())
    assert row["errors"] == ["Tool unavailable"]
    assert not row["passed"]


def test_cost_includes_failed_authoring_and_validation():
    ledger = Ledger()
    ledger.add("generation", calls=1, tokens=10, status="error")
    ledger.add("debugging", calls=1, tokens=20)
    ledger.add("validation", wall_seconds=1)
    ledger.add("science", scientific_steps=1)
    assert ledger.total() == dict(calls=2, tokens=30, wall_seconds=1, scientific_steps=1, tool_calls=0)
    ledger.add("maintenance", tokens=None, wall_seconds=None, status="unknown")
    assert ledger.total()["tokens"] is None


def test_real_public_events_and_provenance(spec):
    data = json.loads((ROOT / "benchmarks/q2/public-events.json").read_text())
    assert len({t["source_family"] for t in data["tasks"]}) == 2
    for task in data["tasks"]:
        assert digest(task["event"]) == task["event_sha256"]
        evidence = from_uff_event(task["event"])
        assert execute(spec, evidence["payload"])["ok"]
        bad = copy.deepcopy(task["event"])
        bad["result"]["ok"] = False
        with pytest.raises(ValueError):
            from_uff_event(bad)


def test_no_model_budget_is_approved():
    config = json.loads((ROOT / "configs/q2_tool_pilot.json").read_text())
    assert all(value == 0 for value in config["approved_new_experiment_budget"].values())
    assert config["proposed_budget"]["research_calls"] == 6 * 4 * 2


def test_loader_rejects_large_document_before_parse(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(" " * 16385)
    with pytest.raises(ValueError, match="byte limit"):
        load_spec(path)


def test_candidate_cannot_replace_broker_input(spec):
    def decide(stage, context):
        return dict(response=[{"tool": "uff_audit", "arguments": {"converged": True}}]
                    if stage == "plan" else {}, calls=0, tokens=0, wall_seconds=0)
    row = episode(fixtures()[0], spec, "augmented", decide, Ledger())
    assert "bound to broker" in row["errors"][0]
    assert not row["passed"]


@pytest.mark.parametrize("response", [None, {"response": []},
    {"response": [], "calls": 1, "tokens": -1, "wall_seconds": 0}])
def test_malformed_backend_result_keeps_unknown_failed_cost(spec, response):
    row = episode(fixtures()[0], spec, "fixed", lambda *_: response, Ledger())
    assert not row["completed"]
    assert row["cost"]["calls"] == 1
    assert row["cost"]["tokens"] is None


def test_request_limit_does_not_retry(spec):
    stages = []
    def decide(stage, context):
        stages.append(stage)
        return dict(response=[{}] * 5, calls=1, tokens=7, wall_seconds=0)
    row = episode(fixtures()[0], spec, "fixed", decide, Ledger())
    assert stages == ["plan"]
    assert row["cost"]["tokens"] == 7
    assert row["cost"]["tool_calls"] == 0
    assert not row["completed"]

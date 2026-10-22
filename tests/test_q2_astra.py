"""No-network tests of pilot accounting, schemas, task boundaries and failure stops."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from rsi.q2_astra import science_recipes, make_tasks, summarize, Pilot
from rsi.q2_budget import RunBudget, BudgetStop
from rsi.q2_model_protocol import AUTHOR_SCHEMA, PLAN_SCHEMA, FINAL_SCHEMA
from rsi.tool_runtime import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def budget(tmp_path):
    config = json.loads((ROOT / "configs/q2_tool_astra.json").read_text())
    return RunBudget(tmp_path / "budget.json", config["budget"])


def finish(budget, row, tokens=10):
    budget.finish_model(row, dict(status="ok", usage=dict(input_tokens=tokens, output_tokens=0), duration_seconds=1))


def test_reservation_persisted_before_call(budget):
    row = budget.reserve_model("author", "a")
    assert json.loads(budget.path.read_text())["entries"][0]["status"] == "reserved"
    with pytest.raises(BudgetStop, match="Unresolved"):
        budget.reserve_model("research", "r")
    finish(budget, row)
    assert json.loads(budget.path.read_text())["entries"][0]["tokens"] == 10


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1}, {"input_tokens": -1, "output_tokens": 1}])
def test_unknown_usage_stops_every_stage(budget, usage):
    row = budget.reserve_model("author", "a")
    budget.finish_model(row, dict(status="error", usage=usage))
    with pytest.raises(BudgetStop):
        budget.reserve_model("research", "r")
    assert row["tokens"] is None


def test_soft_limit_stops_next_call(budget):
    row = budget.reserve_model("research", "r")
    finish(budget, row, 25001)
    with pytest.raises(BudgetStop):
        budget.reserve_model("author", "a")


def test_author_and_research_limits_are_separate(budget):
    for i in range(3):
        finish(budget, budget.reserve_model("author", str(i)))
    with pytest.raises(BudgetStop):
        budget.reserve_model("author", "extra")
    finish(budget, budget.reserve_model("research", "allowed"))


def test_scientific_failures_consume_budget(budget):
    for i in range(4):
        row = budget.reserve_science(str(i), 1)
        row["status"] = "error"
        budget.save()
    with pytest.raises(BudgetStop):
        budget.reserve_science("fifth", 0)


def test_schemas_accept_reference_without_supplying_it_to_author():
    spec = load_spec(ROOT / "tools/q2/manual-audit-v1.json")
    spec["origin"] = "outer_generated"
    jsonschema.validate({"hypothesis": "test", "spec": spec}, AUTHOR_SCHEMA)
    jsonschema.validate({"requests": [{"tool": "uff_audit", "arguments": {}}]}, PLAN_SCHEMA)
    for schema in (AUTHOR_SCHEMA, PLAN_SCHEMA, FINAL_SCHEMA):
        jsonschema.Draft202012Validator.check_schema(schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"requests": [{"tool": "expression", "arguments": {"exec": "anything"}}]}, PLAN_SCHEMA)


def test_science_recipes_preserve_mil88b_restriction():
    recipes = science_recipes()
    assert len(recipes) == 4
    assert sum(a.get("max_steps", 0) for r in recipes for a in r["actions"]) == 3
    blocked = recipes[-1]
    assert blocked["predeclared_exclusion"]
    assert all(a["verb"] != "optimize" for a in blocked["actions"])
    assert {r["source_family"] for r in recipes if r["split"] == "development"}.isdisjoint(
        {r["source_family"] for r in recipes if r["split"] == "evaluation"})


def test_unit_variants_keep_material_split_and_unavailable_denominator():
    payload = dict(initial_energy=10, energy=7, max_force=2, converged=False,
                   energy_unit="kcal/mol", force_unit="kcal/mol/A")
    records = [dict(task_id=r["task"]["id"], source_family=r["source_family"], split=r["split"],
                    eligible=not r["predeclared_exclusion"], reason=r["predeclared_exclusion"], evidence={},
                    payload=None if r["predeclared_exclusion"] else copy.deepcopy(payload)) for r in science_recipes()]
    tasks = make_tasks(records)
    assert len(tasks) == 6
    assert sum(t["eligible"] for t in tasks) == 4
    assert tasks[3]["payload"]["max_force"] == pytest.approx(83.68)
    assert tasks[2]["split"] == tasks[3]["split"] == "evaluation"
    assert records[2]["payload"] == payload


def test_summary_reports_missing_pairs_not_false_ties():
    rows = [dict(task_id="a", arm=arm, eligible=True, completed=False, passed=False, split="evaluation",
                 source_family="x", cost=dict(calls=0, tokens=0, wall_seconds=0), tool_trace=[])
            for arm in ("fixed", "outer_generated", "manual_control", "removed")]
    summary = summarize(rows)
    assert summary["paired_vs_fixed"]["outer_generated"] == dict(wins=0, losses=0, ties=0, missing=1)


def test_author_never_receives_evaluation_or_reference_tool(tmp_path):
    pilot = object.__new__(Pilot)
    pilot.output = tmp_path
    pilot.config = {"budget": {"generation_and_debug_calls": 1}}
    pilot.overhead = []
    spec = load_spec(ROOT / "tools/q2/manual-audit-v1.json")
    spec["origin"] = "outer_generated"
    prompts = []
    def model(stage, label, prompt, schema):
        prompts.append(prompt)
        return {"status": "ok", "response": {"spec": spec}}, 10
    pilot.model = model
    pilot.author([dict(source_family="dev", payload={"example": 1}, split="development", eligible=True),
                  dict(source_family="SECRET_EVALUATION", payload={"secret": 999}, split="evaluation", eligible=True)])
    assert "SECRET_EVALUATION" not in prompts[0]
    assert "manual-audit-v1.json" not in prompts[0]
    assert (tmp_path / "generated-tool.json").exists()

"""Controller tests; scientific invariance checks live in budgeted Q2B P0."""
import json

import pytest

from rsi.q2b.limits import Ledger
from rsi.q2b.protocol import AUTHOR_SCHEMA, PLAN_SCHEMA, REPORT_SCHEMA, research_prompt
from rsi.q2b.runtime import manual_program, validate_program
from rsi.q2b.run import LIMITS


def test_manual_contract():
    from jsonschema import validate
    program = manual_program()
    validate({"program": program}, PLAN_SCHEMA)
    validate({"hypothesis": "Reusable geometry", "program": program}, AUTHOR_SCHEMA)
    assert validate_program(program) == 9
    assert len(program["nodes"]) <= 16


def test_schema_validity():
    import jsonschema
    for schema in (AUTHOR_SCHEMA, PLAN_SCHEMA, REPORT_SCHEMA):
        jsonschema.Draft202012Validator.check_schema(schema)


def test_fixed_removed_identical_and_no_target_leak():
    pair = dict(case="A1", expected="different", source="E1", block="material_transfer",
                provenance={"secret": True}, A=dict(frac=[[]], elements=["C"], edges=[]),
                B=dict(frac=[[]], elements=["C"], edges=[]))
    prompt = research_prompt(pair, None)
    assert "secret" not in prompt and "material_transfer" not in prompt and "A1" not in prompt
    assert "Additional pure tool" not in prompt
    assert "Additional pure tool" in research_prompt(pair, manual_program())


def test_reference_budget_counts_two_checks(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    for i in range(24):
        row = ledger.reserve("reference", str(i), 20)
        row.update(status="ok", wall_seconds=.1)
        ledger.save()
    with pytest.raises(RuntimeError, match="Stage budget"):
        ledger.reserve("reference", "extra", 20)


def test_pending_reservation_fails_closed(tmp_path):
    path = tmp_path / "ledger.json"
    Ledger(path).reserve("small", "pending", 20)
    with pytest.raises(RuntimeError, match="unresolved"):
        Ledger(path).reserve("small", "later", 20)


def test_total_wall_reservation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.data["entries"] = [dict(stage="small", status="ok", wall_seconds=3590)]
    with pytest.raises(RuntimeError, match="wall-time"):
        ledger.reserve("research_tools", "episode", 20)


def test_config_limits_match():
    from rsi.q2b.limits import ROOT
    config = json.loads((ROOT / "configs/q2b.json").read_text())
    assert config["total_calls"] == LIMITS["total_model_calls"] == 195
    assert config["total_tokens_soft_limit"] == LIMITS["observed_token_soft_limit"]
    assert config["research_calls"] == 24 * 4 * 2
    assert json.loads((ROOT / "tools/q2b/manual-v2.json").read_text()) == manual_program()


def test_frozen_registered_name_mismatch_is_not_silently_repaired():
    program = dict(version=2, nodes=[dict(id="audit", op="registered", inputs=["A", "B"], config="{}")],
                   outputs=["audit.alignment"])
    assert validate_program(program, registered=manual_program()) == 9
    program["nodes"][0]["op"] = "periodic_pair_evidence"
    with pytest.raises(ValueError, match="Unknown primitive"):
        validate_program(program, registered=manual_program())

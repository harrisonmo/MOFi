"""Interface, freeze boundaries, and versioned transport regression checks."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import jsonschema
import pytest

from rsi.q2b_v21 import protocol, runtime, scoring, transport
from rsi.q2b_v21.budget import RunBudget, BudgetStop
from rsi.q2b_v21.run import LIMITS
from rsi.q2b_v21.limits import ROOT


def wrapper(op="periodic_pair_evidence"):
    return dict(version=2, nodes=[dict(id="audit", op=op, inputs=["A", "B"], config="{}")], outputs=["audit"])


def test_unique_name_in_schema_runtime_and_description():
    pair = dict(case="A0", A=dict(frac=[[]], elements=["C"], edges=[]), B=dict(frac=[[]], elements=["C"], edges=[]))
    program = wrapper()
    jsonschema.validate(dict(program=program), protocol.plan_schema(True))
    assert runtime.validate_program(program, registered=runtime.manual_program()) == 9
    assert "Additional pure tool periodic_pair_evidence(A,B)" in protocol.research_prompt(pair, runtime.manual_program())
    for invalid in ("registered", "audit", "eval"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(dict(program=wrapper(invalid)), protocol.plan_schema(True))
        with pytest.raises(ValueError):
            runtime.validate_program(wrapper(invalid), registered=runtime.manual_program())


def test_removed_tool_unavailable_and_author_cannot_recurse():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(program=wrapper()), protocol.plan_schema(False))
    with pytest.raises(ValueError):
        runtime.validate_program(wrapper())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(hypothesis="No recursion", program=wrapper()), protocol.AUTHOR_SCHEMA)


def test_scientific_score_and_inputs_frozen():
    from rsi.q2b import scoring as original, protocol as old_protocol
    assert inspect.getsource(scoring.score) == inspect.getsource(original.score)
    assert protocol.REPORT_SCHEMA == old_protocol.REPORT_SCHEMA
    assert protocol.QUESTIONS == old_protocol.QUESTIONS
    for name in ("reference.py", "fixtures.py"):
        assert (ROOT / "rsi/q2b" / name).read_bytes() == (ROOT / "rsi/q2b_v21" / name).read_bytes()
    assert json.loads((ROOT / "tools/q2b/manual-v2.json").read_text()) == runtime.manual_program()
    from rsi.q2b.run import Pilot as OldPilot
    from rsi.q2b_v21.run import Pilot
    expected = inspect.getsource(OldPilot.episode).replace('== "registered"', '== "periodic_pair_evidence"')
    expected = expected.replace('PLAN_SCHEMA)', 'plan_schema(registered is not None))')
    assert inspect.getsource(Pilot.episode) == expected


def test_probe_budget_and_unknown_usage_stop(tmp_path):
    ledger = RunBudget(tmp_path / "budget.json", LIMITS)
    for index in range(2):
        row = ledger.reserve_model("probe", str(index))
        ledger.finish_model(row, dict(status="ok", usage=dict(input_tokens=10, output_tokens=2)))
    with pytest.raises(BudgetStop):
        ledger.reserve_model("probe", "third")
    row = ledger.reserve_model("author", "author-1")
    ledger.finish_model(row, dict(status="timeout", usage=None))
    with pytest.raises(BudgetStop):
        ledger.reserve_model("research", "late")


def test_versioned_cli_flags_preserve_auth(monkeypatch, tmp_path):
    job, process = Mock(), Mock(pid=9123)
    monkeypatch.setattr(transport, "_WindowsJob", Mock(return_value=job))
    process.poll.return_value = 0
    process.communicate.return_value = (json.dumps(dict(type="turn.completed", usage=dict(input_tokens=10, output_tokens=2))), "")
    def spawn(argv, **kwargs):
        Path(argv[argv.index("--output-last-message") + 1]).write_text('{"ready":true}')
        return process
    factory = Mock(side_effect=spawn)
    monkeypatch.setattr(transport.subprocess, "Popen", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "never-expose")
    schema = protocol.obj(dict(ready={"type": "boolean"}))
    result = transport.CodexCLI(executable="codex", timeout=240).run("Return ready=true", schema, tmp_path)
    assert result["status"] == "ok"
    argv = factory.call_args.args[0]
    assert argv[1:3] == ["--no-daemon", "exec"]
    assert "skills.bundled.enabled=false" in argv
    assert "--strict-config" in argv and "unbounded_connection_retries" in argv
    assert "OPENAI_API_KEY" not in factory.call_args.kwargs["env"]
    assert "CODEX_HOME" not in inspect.getsource(transport.CodexCLI.run)

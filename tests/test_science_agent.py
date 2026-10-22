"""Agentic loop over the science hub with a scripted model: tool choice, errors, citations and limits."""

import json
from pathlib import Path

import jsonschema
import pytest

pytest.importorskip("ase")

from rsi.science.agent import ScienceAgent, isolated_scratch, step_schema, validate_agent_config  # noqa: E402
from rsi.science.hub import ScienceToolHub  # noqa: E402
from test_science_hub import emt_relax, make_config  # noqa: E402


class ScriptedModel:
    """Return queued replies through the backend contract and keep every prompt it was shown."""

    def __init__(self, replies, status="ok"):
        self.replies = list(replies)
        self.status = status
        self.prompts = []
        self.output_dirs = []

    def run(self, prompt, schema, output_dir):
        jsonschema.validators.validator_for(schema).check_schema(schema)
        self.prompts.append(prompt)
        self.output_dirs.append(Path(output_dir))
        (Path(output_dir) / "workspace").mkdir(parents=True)
        (Path(output_dir) / "prompt.txt").write_text(prompt, encoding="utf-8")
        if self.status != "ok":
            return {"status": self.status, "response": None, "usage": None}
        reply = self.replies.pop(0)
        jsonschema.validate(reply, schema)
        return {"status": "ok", "response": reply, "usage": {"input_tokens": 100, "output_tokens": 10},
                "actual_model": "scripted"}


def call(tool, arguments):
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"reasoning": "r", "action": "call_tool", "tool": tool, "arguments_json": text,
            "answer": None, "evidence_call_ids": None, "caveats": None}


def report(answer, cited):
    return {"reasoning": "r", "action": "final_report", "tool": None, "arguments_json": None,
            "answer": answer, "evidence_call_ids": cited, "caveats": ["EMT is not DFT"]}


@pytest.fixture
def hub(tmp_path):
    return ScienceToolHub(make_config(tmp_path), tmp_path / "session")


def test_tool_call_then_cited_report(hub, tmp_path):
    model = ScriptedModel([call("dft_calculation", emt_relax()), report("E = x eV", ["call-0001"])])
    record = ScienceAgent(model, hub, tmp_path / "run").run("Relax the Cu cluster with EMT.")
    assert record["status"] == "completed"
    assert record["steps"][0]["observation"]["ok"] and record["steps"][0]["tool"] == "dft_calculation"
    assert record["final"]["unsupported_citations"] == []
    assert record["usage"] == {"model_calls": 2, "tokens": 220}
    assert record["hub_budget"]["calls_used"] == 1
    assert "dft_calculation" in model.prompts[0] and "(no actions yet)" in model.prompts[0]
    assert "call-0001" in model.prompts[1] and "Relax the Cu cluster" in model.prompts[1]
    saved = json.loads((tmp_path / "run" / "agent-run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completed"


def test_rejected_actions_are_shown_and_never_launch(hub, tmp_path):
    model = ScriptedModel([call("dft_calculation", "{not json"),
                           call("dft_calculation", emt_relax(max_steps=10**6)),
                           call("dft_calculation", "[1, 2]"),
                           report("nothing ran", [])])
    record = ScienceAgent(model, hub, tmp_path / "run").run("Relax the Cu cluster.")
    errors = [s["observation"]["error"]["message"] for s in record["steps"][:3]]
    assert "not valid JSON" in errors[0] and "must lie in" in errors[1] and "JSON object" in errors[2]
    assert "must lie in" in model.prompts[2]
    assert record["hub_budget"]["calls_used"] == 0 and not list((tmp_path / "session").glob("call-*"))
    assert record["status"] == "invalid_report"  # no evidence cited


def test_citation_must_be_successful_call(hub, tmp_path):
    model = ScriptedModel([call("science_budget_status", {}), report("made up", ["call-0007"])])
    record = ScienceAgent(model, hub, tmp_path / "run").run("Anything.")
    assert record["status"] == "invalid_report"
    assert record["final"]["unsupported_citations"] == ["call-0007"]


def test_step_and_token_limits(hub, tmp_path):
    model = ScriptedModel([call("science_budget_status", {})] * 3)
    record = ScienceAgent(model, hub, tmp_path / "a", max_steps=3).run("Loop.")
    assert record["status"] == "step_limit" and record["usage"]["model_calls"] == 3
    assert "return action final_report now" in model.prompts[-1]
    model = ScriptedModel([call("science_budget_status", {})] * 3)
    record = ScienceAgent(model, hub, tmp_path / "b", max_steps=3, max_total_tokens=200).run("Loop.")
    assert record["status"] == "token_limit" and record["usage"]["model_calls"] == 2


def test_model_error_stops_and_run_is_not_reused(hub, tmp_path):
    record = ScienceAgent(ScriptedModel([], status="timeout"), hub, tmp_path / "run").run("x")
    assert record["status"] == "model_error" and record["steps"][0]["model_status"] == "timeout"
    with pytest.raises(ValueError, match="already exists"):
        ScienceAgent(ScriptedModel([]), hub, tmp_path / "run").run("x")


def test_observation_is_truncated_for_the_model(hub, tmp_path):
    model = ScriptedModel([call("science_budget_status", {}), report("ok", ["x"])])
    ScienceAgent(model, hub, tmp_path / "run", max_observation_chars=50).run("x")
    assert "[truncated" in model.prompts[1]


def test_schema_and_config_validation():
    schema = step_schema(["dft_calculation"])
    assert schema["properties"]["tool"]["enum"] == ["dft_calculation", None]
    jsonschema.validators.validator_for(schema).check_schema(schema)
    validate_agent_config({"schema_version": 1, "backend": "codex_cli", "model": "m", "max_steps": 3})
    for bad in ({"schema_version": 1, "backend": "codex_cli", "model": "m", "max_steps": 0},
                {"schema_version": 1, "backend": "codex_cli", "model": "m", "max_steps": 3, "extra": 1}):
        with pytest.raises(ValueError):
            validate_agent_config(bad)


def test_model_calls_run_outside_any_repository(hub, tmp_path):
    model = ScriptedModel([call("science_budget_status", {}), report("ok", ["x"])])
    agent = ScienceAgent(model, hub, tmp_path / "run", model_scratch=tmp_path / "scratch-root")
    record = agent.run("x")
    repository = Path(__file__).resolve().parents[1]
    for index, output in enumerate(model.output_dirs, start=1):
        workspace = (output / "workspace").resolve()
        assert repository not in workspace.parents
        assert not any((folder / name).exists() for folder in workspace.parents for name in ("AGENTS.md", ".git"))
        assert not output.exists()  # scratch removed after the call
        kept = tmp_path / "run" / "model" / f"step-{index:02d}"
        assert (kept / "prompt.txt").read_text(encoding="utf-8") == model.prompts[index - 1]
    assert record["model_scratch"] == str((tmp_path / "scratch-root").resolve())


def test_scratch_inside_a_repository_is_refused(hub, tmp_path):
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="not isolated"):
        isolated_scratch(repository / "runs")
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "AGENTS.md").write_text("Answer in Chinese.", encoding="utf-8")
    with pytest.raises(ValueError, match="AGENTS.md"):
        ScienceAgent(ScriptedModel([]), hub, tmp_path / "run", model_scratch=tmp_path / "project" / "scratch")

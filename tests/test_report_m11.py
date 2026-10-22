"""Synthetic replay integrity tests; never invoke a model or scientific worker."""

import copy
import json

import pytest

from rsi import m11
from rsi.experiment import read_json, write_json
from scripts import report_m11 as audit


def save_call(directory, config, response, prompt="admission", schema=None):
    directory.mkdir(parents=True, exist_ok=True)
    schema = schema or {"type": "object"}
    usage = {"input_tokens": 3, "output_tokens": 2}
    result = {"status": "ok", "exit_code": 0, "requested_model": config["model"],
              "actual_model": None, "model_evidence": [], "unexpected_tool_types": [],
              "usage": usage, "response": response, "duration_seconds": 0.01}
    write_json(directory / "result.json", result)
    write_json(directory / "final.json", response)
    write_json(directory / "schema.json", schema)
    write_json(directory / "command.json", {
        "requested_model": config["model"], "reasoning": config["reasoning"],
        "timeout_seconds": config["cli_timeout_seconds"],
        "authentication": "existing_cli_login", "prompt_transport": "stdin",
    })
    (directory / "prompt.txt").write_text(audit.DECISION_INSTRUCTIONS + prompt, encoding="utf-8")
    (directory / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": usage}) + "\n", encoding="utf-8")
    return result


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    """Build a zero-action synthetic comparison to exercise all replay plumbing."""
    monkeypatch.setattr(m11, "frozen_hashes", lambda: ({}, {"synthetic_fixture": True}))
    monkeypatch.setattr(m11, "validate_science_admission", lambda path, *args: read_json(path))
    from rsi.backends.codex_cli import CodexCLI
    monkeypatch.setattr(CodexCLI, "run", lambda *args, **kwargs: pytest.fail("Unexpected inference"))
    monkeypatch.setattr(m11.ThreeArmExperiment, "science", lambda *args, **kwargs: pytest.fail("Unexpected scientific worker"))
    run = tmp_path / "source"
    run.mkdir()
    cli = {"path": "/synthetic/codex", "sha256": "synthetic-sha", "version": "test"}
    admission_dir = run / "admission-evidence"
    result = save_call(admission_dir / "probe", {**m11.CONFIG, "cli_timeout_seconds": 120},
                       {"ready": True}, m11.ADMISSION_PROMPT, m11.ADMISSION_SCHEMA)
    admitted = {"admitted": True, "usage_known": True, "status": "ok", "tokens": 5,
                "actual_model": None, "duration_seconds": 0.01}
    write_json(admission_dir / "summary.json", admitted)
    write_json(admission_dir / "manifest.json", {"cli": cli, "max_calls": 1,
                                               "max_observed_tokens": 10000, "science_units": 0})
    write_json(admission_dir / "ledger.json", [{"label": "admission", "status": "known",
                                              "usage": result["usage"], "result_status": "ok"}])
    write_json(run / "admission-summary.json", admitted)
    admission_sources = {path.relative_to(admission_dir).as_posix(): {"path": str(path), "sha256": m11.digest(path)}
                         for path in admission_dir.rglob("*") if path.is_file()}
    science_dir = run / "science-admission-evidence"
    science_summary = {"admitted": True, "synthetic_fixture": True}
    write_json(science_dir / "summary.json", science_summary)
    science_source = {"summary.json": {"path": str(science_dir / "summary.json"),
                                       "sha256": m11.digest(science_dir / "summary.json")}}
    pools = read_json(m11.ROOT / "benchmarks/m7/pools.json")["pools"]
    data = m11.feedback()
    candidate_policy = {**m11.baseline_policy("m8"), "shortlist_size": 8}
    proposal = {"workflow": candidate_policy, "hypothesis": "Synthetic test only; no performance claim."}
    records = []

    class Recorder:
        def __init__(self, config, answers):
            self.config, self.answers = config, list(answers)

        def run(self, prompt, schema, directory):
            value = save_call(directory, self.config, self.answers.pop(0), prompt, schema)
            records.append({"label": str(directory), "status": "known", "tokens": 5})
            return value

    outer_config = dict(m11.CONFIG, max_llm_calls=1, max_observed_tokens=100000)
    outer = m11.ThreeArmExperiment(outer_config, run / "evolution", Recorder(outer_config, [proposal]))
    outer.model(m11.proposal_prompt(data), audit.PROPOSAL_SCHEMA, run / "evolution/proposal")
    rows = []
    schedule = []
    for i, pool in enumerate(pools):
        report = {"summary": "Synthetic unobserved candidates.", "candidates": [
            {"candidate_id": item["id"], "status": "unresolved", "diagnostic_codes": [],
             "evidence_actions": [], "summary": "Unobserved.", "scientific_status": "not_assessed"}
            for item in pool["candidates"]]}
        for arm in m11.ARMS[i % 3:] + m11.ARMS[:i % 3]:
            schedule.append({"pool": pool["id"], "arm": arm})
            output = run / pool["id"] / arm
            backend = Recorder(m11.CONFIG, [{"actions": [], "rationale": "Synthetic test stops immediately."}, report])
            exp = m11.WorkflowScreeningExperiment(m11.CONFIG, output, backend)
            pol = candidate_policy if arm == "candidate" else m11.baseline_policy(arm)
            row = exp.episode(pool, arm, pol)
            row["cost"] = {"calls": exp.calls, "tokens": exp.observed_tokens, "units": 0,
                           "requested_steps": 0, "unknown_usage": 0, "wall_seconds": 0.01}
            write_json(output / "graded.json", row)
            rows.append(row)
    manifest = {
        "root": {"commit": "synthetic", "status": ""}, "crystal_builder": {"commit": m11.CRYSTAL_COMMIT, "status": ""},
        "cli": cli, "config": m11.CONFIG, "packages": {}, "source_and_input_hashes": {},
        "freeze_audit": {"synthetic_fixture": True}, "baseline_policies": {arm: m11.baseline_policy(arm) for arm in ["batch", "m8"]},
        "historical_feedback_sources": data["provenance"], "schedule": schedule,
        "admission": {"path": str(admission_dir / "summary.json"), "sha256": m11.digest(admission_dir / "summary.json"),
                      "summary": admitted, "cli": cli, "sources": admission_sources},
        "science_admission": {"path": str(science_dir / "summary.json"), "sha256": m11.digest(science_dir / "summary.json"),
                              "summary": science_summary, "sources": science_source},
    }
    write_json(run / "feedback.json", data)
    manifest["feedback_sha256"] = m11.digest(run / "feedback.json")
    write_json(run / "manifest.json", manifest)
    write_json(run / "rows.json", rows)
    write_json(run / "pools.json", pools)
    write_json(run / "candidate.json", {**proposal, "sha256": audit.policy_hash(candidate_policy)})
    write_json(run / "summary.json", m11.summarize(rows))
    write_json(run / "selection.json", m11.selection(rows))
    parent = m11.baseline_policy("m8")
    write_json(run / "accepted.json", {"workflow": parent, "sha256": audit.policy_hash(parent),
                                      "source": "m8", "scope": "M7 public development"})
    write_json(run / "stage-ledger.json", records)
    return run


def test_completed_synthetic_comparison_replays_without_model_or_worker(completed_run, tmp_path):
    result = audit.audit(completed_run, tmp_path / "replay")
    assert result["audit"]["episodes_replayed"] == 18
    assert result["audit"]["decision_prompts_checked"] == 36
    assert result["audit"]["outer_prompts_checked"] == 1
    assert result["total_cost"]["calls"] == 37
    assert result["total_cost"]["tokens"] == 185
    assert result["admission"]["tokens"] == 5
    assert result["selection"]["promoted"] is False


def test_incomplete_comparison_fails_before_replay_or_provenance(tmp_path, monkeypatch):
    run = tmp_path / "run"
    write_json(run / "manifest.json", {"schedule": []})
    write_json(run / "rows.json", [])
    monkeypatch.setattr(audit, "check_provenance", lambda *args: pytest.fail("Read incomplete run evidence"))
    with pytest.raises(audit.AuditError, match="Incomplete"):
        audit.audit(run, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


@pytest.mark.parametrize("patch", [{"input_tokens": None}, {"input_tokens": True}, {"input_tokens": -1}])
def test_unknown_or_invalid_usage_is_not_counted_as_zero(patch):
    with pytest.raises(audit.AuditError, match="usage"):
        audit.known_tokens({"input_tokens": 2, "output_tokens": 3, **patch})


def test_raw_event_usage_is_checked_against_claimed_result(tmp_path):
    directory = tmp_path / "call"
    save_call(directory, m11.CONFIG, {"ok": True})
    (directory / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 2}}))
    with pytest.raises(audit.AuditError, match="Raw CLI usage"):
        audit.check_saved_call(directory, m11.CONFIG)


def test_explicit_actual_model_mismatch_is_rejected(tmp_path):
    directory = tmp_path / "call"
    result = save_call(directory, m11.CONFIG, {"ok": True})
    result.update(actual_model="other-model", model_evidence=[{"event_type": "session.started", "model": "other-model"}])
    write_json(directory / "result.json", result)
    (directory / "events.jsonl").write_text(json.dumps({"type": "session.started", "model": "other-model"}) + "\n"
        + json.dumps({"type": "turn.completed", "usage": result["usage"]}))
    with pytest.raises(audit.AuditError, match="Explicit actual model"):
        audit.check_saved_call(directory, m11.CONFIG)


def test_equal_aggregate_tokens_do_not_hide_swapped_stage_usage():
    records = [{"label": "one", "status": "known", "tokens": 3},
               {"label": "two", "status": "known", "tokens": 7}]
    stage = copy.deepcopy(records)
    stage[0]["tokens"], stage[1]["tokens"] = 7, 3
    with pytest.raises(audit.AuditError, match="ordered saved"):
        audit.check_stage(stage, records)


def test_admission_copy_mutation_is_detected_before_revalidation(completed_run, tmp_path):
    path = completed_run / "admission-evidence/probe/result.json"
    record = read_json(path)
    record["duration_seconds"] = 123
    write_json(path, record)
    with pytest.raises(audit.AuditError, match="copy hash"):
        audit.audit(completed_run, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_saved_prompt_mutation_cannot_produce_an_audited_report(completed_run, tmp_path):
    prompt = completed_run / "m7-dev-01/batch/decision-00/prompt.txt"
    prompt.write_text(prompt.read_text() + "changed")
    with pytest.raises(audit.AuditError, match="Prompt replay differs"):
        audit.audit(completed_run, tmp_path / "replay")


def test_existing_report_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("original")
    monkeypatch.setattr(audit, "audit", lambda *args: pytest.fail("Audit started before destination check"))
    with pytest.raises(audit.AuditError, match="refusing to overwrite"):
        audit.main(["--run", str(tmp_path / "run"), "--replay-output", str(tmp_path / "replay"), "--report-output", str(output)])
    assert output.read_text() == "original"


def test_all_output_arguments_are_required():
    with pytest.raises(SystemExit) as error:
        audit.main([])
    assert error.value.code == 2

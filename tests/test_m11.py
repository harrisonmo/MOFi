"""Offline checks for M11 admission, evidence provenance and frozen selection."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi import m10, m11
from rsi.experiment import read_json, write_json
from rsi.backends.codex_cli import DECISION_INSTRUCTIONS


FAKE_CLI = {"path": "fake", "sha256": "a" * 64, "version": "test"}


def admitted_summary(directory):
    summary = directory / "admission.json"
    write_json(summary, {"admitted": True, "usage_known": True, "status": "ok", "tokens": 10})
    write_json(directory / "probe/result.json", {"status": "ok", "exit_code": 0,
        "usage": {"input_tokens": 7, "output_tokens": 3}, "unexpected_tool_types": [],
        "requested_model": "gpt-6-astra", "response": {"ready": True}, "actual_model": None, "model_evidence": []})
    write_json(directory / "probe/command.json", {"requested_model": "gpt-6-astra", "reasoning": "high",
        "timeout_seconds": 120, "authentication": "existing_cli_login", "prompt_transport": "stdin"})
    write_json(directory / "manifest.json", {"cli": FAKE_CLI, "max_calls": 1, "max_observed_tokens": 10000, "science_units": 0})
    write_json(directory / "probe/schema.json", m11.ADMISSION_SCHEMA)
    write_json(directory / "probe/final.json", {"ready": True})
    (directory / "probe/prompt.txt").write_text(DECISION_INSTRUCTIONS + m11.ADMISSION_PROMPT, encoding="utf-8")
    (directory / "probe/events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}) + "\n", encoding="utf-8")
    write_json(directory / "ledger.json", [{"label": "admission", "status": "known", "usage": {"input_tokens": 7, "output_tokens": 3}, "result_status": "ok"}])
    return summary


def test_selection_and_budget_configuration_are_reused_exactly():
    assert m11.selection is m10.selection
    assert m11.summarize is m10.summarize
    assert m11.CONFIG is m10.CONFIG
    assert m11.ARMS is m10.ARMS
    assert m11.Budget is m10.Budget


@pytest.mark.parametrize("mutation", ["false_ready", "missing_events", "extra_turn", "usage_mismatch",
                                      "tool_event", "wrong_prompt", "wrong_schema", "unknown_ledger"])
def test_probe_corruption_stops_before_model_construction(tmp_path, monkeypatch, mutation):
    summary = admitted_summary(tmp_path)
    events = tmp_path / "probe/events.jsonl"
    if mutation == "false_ready":
        result = read_json(tmp_path / "probe/result.json")
        result["response"] = {"ready": False}
        write_json(tmp_path / "probe/result.json", result)
        write_json(tmp_path / "probe/final.json", {"ready": False})
    elif mutation == "missing_events":
        events.unlink()
    elif mutation == "extra_turn":
        events.write_text(events.read_text() * 2, encoding="utf-8")
    elif mutation == "usage_mismatch":
        events.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 8, "output_tokens": 3}}), encoding="utf-8")
    elif mutation == "tool_event":
        events.write_text(events.read_text() + json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}), encoding="utf-8")
    elif mutation == "wrong_prompt":
        (tmp_path / "probe/prompt.txt").write_text(DECISION_INSTRUCTIONS + "Different task", encoding="utf-8")
    elif mutation == "wrong_schema":
        write_json(tmp_path / "probe/schema.json", {"type": "object"})
    else:
        write_json(tmp_path / "ledger.json", [{"status": "unknown"}])
    monkeypatch.setattr(m11, "ThreeArmExperiment", lambda *a, **k: pytest.fail("Model constructed before evidence validation"))
    with pytest.raises((ValueError, FileNotFoundError)):
        m11.run(tmp_path / "run", summary, tmp_path / "science/summary.json")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("patch", [
    {"admitted": False}, {"usage_known": False}, {"status": "process_error"},
    {"tokens": -1}, {"tokens": 10001}, {"tokens": True}, {"admitted": 1},
    {"usage_known": 1}, {"tokens": None},
])
def test_rejected_admission_never_constructs_a_model(tmp_path, monkeypatch, patch):
    summary = admitted_summary(tmp_path)
    write_json(summary, {"admitted": True, "usage_known": True, "status": "ok", "tokens": 10, **patch})
    monkeypatch.setattr(m11, "ThreeArmExperiment", lambda *args, **kwargs: pytest.fail("Model constructed before admission"))
    with pytest.raises(ValueError, match="admission"):
        m11.run(tmp_path / "run", summary, tmp_path / "science/summary.json")
    assert not (tmp_path / "run").exists()


def test_missing_admission_fails_before_model_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(m11, "ThreeArmExperiment", lambda *args, **kwargs: pytest.fail("Unexpected model construction"))
    with pytest.raises(FileNotFoundError):
        m11.run(tmp_path / "run", tmp_path / "missing.json", tmp_path / "science/summary.json")


def test_clean_committed_root_required_before_model_calls(tmp_path, monkeypatch):
    summary = admitted_summary(tmp_path)
    monkeypatch.setattr(m11, "git_state", lambda path: {"commit": "abc", "status": "?? pending.py"})
    monkeypatch.setattr(m11, "ThreeArmExperiment", lambda *args, **kwargs: pytest.fail("Unexpected model construction"))
    with pytest.raises(ValueError, match="clean M11"):
        m11.run(tmp_path / "run", summary, tmp_path / "science/summary.json")


@pytest.mark.parametrize("file,patch", [
    ("probe/result.json", {"status": "auth_error"}),
    ("probe/result.json", {"usage": {"input_tokens": 7}}),
    ("probe/result.json", {"usage": {"input_tokens": True, "output_tokens": 9}}),
    ("probe/result.json", {"usage": {"input_tokens": -1, "output_tokens": 11}}),
    ("probe/result.json", {"usage": {"input_tokens": 8, "output_tokens": 3}}),
    ("probe/result.json", {"unexpected_tool_types": ["command_execution"]}),
    ("probe/result.json", {"requested_model": "other"}),
    ("probe/result.json", {"actual_model": "other"}),
    ("probe/command.json", {"requested_model": "other"}),
    ("probe/command.json", {"reasoning": "low"}),
    ("manifest.json", {"cli": {}}),
])
def test_raw_admission_evidence_is_checked(tmp_path, file, patch):
    summary = admitted_summary(tmp_path)
    target = tmp_path / file
    write_json(target, {**read_json(target), **patch})
    with pytest.raises(ValueError, match="raw evidence"):
        m11.validate_admission(summary)


def test_admission_requires_raw_files_and_freezes_them(tmp_path):
    summary = admitted_summary(tmp_path)
    assert m11.validate_admission(summary)["tokens"] == 10
    sources = m11.admission_sources(summary)
    m11.verify_admission_sources(sources)
    (tmp_path / "probe/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        m11.validate_admission(summary)
    write_json(tmp_path / "probe/result.json", {"status": "different"})
    with pytest.raises(RuntimeError, match="admission evidence changed"):
        m11.verify_admission_sources(sources)


def test_cli_drift_rejects_before_proposal(tmp_path, monkeypatch):
    summary = admitted_summary(tmp_path)
    monkeypatch.setattr(m11, "git_state", lambda path: {"commit": "clean", "status": ""})
    monkeypatch.setattr(m11, "frozen_hashes", lambda: ({}, {}))
    monkeypatch.setattr(m11, "validate_science_admission", lambda *args: {})
    monkeypatch.setattr(m11, "science_admission_sources", lambda path: {})
    monkeypatch.setattr(m11.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(m11, "cli_identity", lambda executable: {**FAKE_CLI, "sha256": "b" * 64})
    monkeypatch.setattr(m11, "ThreeArmExperiment", lambda *args, **kwargs: SimpleNamespace(
        backend=SimpleNamespace(executable="fake"),
        model=lambda *args: pytest.fail("CLI drift must prevent model invocation")))
    with pytest.raises(RuntimeError, match="admitted executable"):
        m11.run(tmp_path / "run", summary, summary)
    assert read_json(tmp_path / "run/stage-ledger.json") == []


@pytest.fixture
def frozen_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(m11, "ROOT", tmp_path)
    monkeypatch.setattr(m11, "RUNNER_FILES", ("rsi/m11.py",))
    monkeypatch.setattr(m11, "HISTORY_FILES", ("docs/reports/history.json",))
    monkeypatch.setattr(m11, "git_state", lambda path: {"commit": m11.CRYSTAL_COMMIT, "status": ""})
    (tmp_path / "rsi").mkdir()
    (tmp_path / "rsi/frozen.py").write_bytes(b"first\nsecond\n")
    (tmp_path / "rsi/m11.py").write_bytes(b"controller\n")
    fixture = tmp_path / "benchmarks/m7/fixtures/example.xtalproj"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"exact fixture\n")
    pools = [{"id": "m7-dev-01", "split": "development"}]
    write_json(tmp_path / "benchmarks/m7/pools.json", {"pools": pools})
    write_json(tmp_path / "docs/reports/M7_RESULT.json", {"manifest": {
        "source_hashes": {"rsi/frozen.py": hashlib.sha256(b"first\r\nsecond\r\n").hexdigest()},
        "fixture_hashes": {"example.xtalproj": hashlib.sha256(fixture.read_bytes()).hexdigest()},
        "pools_sha256": hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest(),
    }})
    write_json(tmp_path / "docs/reports/history.json", {"aggregate": True})
    return tmp_path


def test_legacy_newlines_are_explicit_but_current_raw_bytes_are_frozen(frozen_tree):
    hashes, audit = m11.frozen_hashes()
    assert audit["legacy_newline_matches"] == ["rsi/frozen.py"]
    assert audit["historical_raw_runs_available"] is False
    assert hashes["rsi/frozen.py"] == hashlib.sha256(b"first\nsecond\n").hexdigest()
    m11.verify_frozen(hashes)
    (frozen_tree / "rsi/frozen.py").write_bytes(b"first\r\nsecond\r\n")
    with pytest.raises(RuntimeError, match="Frozen experiment content changed"):
        m11.verify_frozen(hashes)


@pytest.mark.parametrize("name,content,match", [
    ("rsi/frozen.py", b"changed\nsecond\n", "source changed"),
    ("benchmarks/m7/fixtures/example.xtalproj", b"exact fixture\r\n", "fixture changed"),
    ("benchmarks/m7/pools.json", b'{"pools": []}', "pools changed"),
])
def test_scientific_changes_are_rejected(frozen_tree, name, content, match):
    (frozen_tree / name).write_bytes(content)
    with pytest.raises(ValueError, match=match):
        m11.frozen_hashes()


@pytest.mark.parametrize("state", [
    {"commit": "other", "status": ""},
    {"commit": m11.CRYSTAL_COMMIT, "status": " M xtal/core.py"},
])
def test_science_checkout_must_match_and_be_clean(frozen_tree, monkeypatch, state):
    monkeypatch.setattr(m11, "git_state", lambda path: state)
    with pytest.raises(ValueError, match="Crystal-Builder"):
        m11.frozen_hashes()


@pytest.mark.parametrize("name", ["rsi/m11.py", "docs/reports/history.json"])
def test_new_controller_and_versioned_feedback_are_frozen(frozen_tree, name):
    hashes, _ = m11.frozen_hashes()
    (frozen_tree / name).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen experiment content changed"):
        m11.verify_frozen(hashes)


def test_feedback_has_hash_provenance_and_excludes_material_records():
    data = m11.feedback()
    assert data["historical_raw_runs_available"] is False
    assert data["historical_selection"]["promoted"] is False
    assert data["historical_summary"]["metrics"]["candidate"]["Y"] == 9
    assert data["provenance"] == {name: m11.digest(m11.ROOT / name) for name in m11.HISTORY_FILES}
    serialized = json.dumps(data)
    for forbidden in ['"candidate_id"', '"framework_id"', '"canonical"', '"reference_actions"', '"qualified"']:
        assert forbidden not in serialized
    assert "runs/" not in serialized


@pytest.fixture
def science_evidence(tmp_path, monkeypatch):
    """Build saved worker records without running any scientific operation."""
    root, directory = tmp_path / "repo", tmp_path / "science"
    monkeypatch.setattr(m11, "ROOT", root)
    sources = ["rsi/runtime.py", "Crystal-Builder/xtal/core.py", "scripts/check_m11_science.py"]
    sources += [f"benchmarks/m7/fixtures/example-{variant}.xtalproj" for variant in ["base", "duplicate", "compound"]]
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged source")
    hashes = {name: m11.digest(root / name) for name in sources}
    packages = {name: "test" for name in m11.SCIENCE_PACKAGES}
    limits = {"prepare_actions": 3, "science_units": 4, "requested_uff_steps": 0, "model_calls": 0}
    manifest = {"source_and_input_hashes": {**hashes, "docs/report.md": "old-doc-hash"},
                "packages": packages, "limits": limits, "family": "example",
                "crystal_builder": {"commit": m11.CRYSTAL_COMMIT, "status": ""},
                "config": {"science_timeout_seconds": 120, "science_seed": 0}}
    write_json(directory / "manifest.json", manifest)
    rows, ledger = [], []
    for variant, cost in [("base", 1), ("duplicate", 1), ("compound", 2)]:
        filename = "example-" + variant + ".xtalproj"
        action = {"verb": "prepare", "steps": ["duplicates", "deuterium"] if variant == "compound" else ["duplicates"]}
        task = {"id": "m11-preparation-" + variant,
                "initialization": {"kind": "open", "root": "m7_fixture", "path": filename}}
        history = []
        for stage, actions in [("initial", []), ("prepare", [action])]:
            target = directory / variant / stage
            target.mkdir(parents=True)
            artifact = target / "structure.xtalproj"
            artifact.write_bytes((variant + stage).encode())
            endpoint = {"path": str(artifact), "sha256": m11.digest(artifact)}
            history.append({**endpoint, "action_count": len(actions)})
            raw = {"task_id": task["id"], "errors": [], "attempted_actions": actions, "stopped": False,
                   "artifact": endpoint, "checkpoint_history": list(history),
                   "events": [{"action": action, "result": {"ok": True}}] if actions else [],
                   "final": {"diagnostics": []}}
            if actions:
                initial = read_json(directory / variant / "initial/result.json")
                write_json(target / "previous.json", initial)
                raw["resumed_from"] = initial["artifact"]
            write_json(target / "task.json", task)
            write_json(target / "actions.json", actions)
            write_json(target / "result.json", raw)
        rows.append({"fixture": filename, "checks": {key: True for key in m11.SCIENCE_CHECKS},
                     "passed": True, "science_units": cost, "current_codes": []})
        ledger.append({"fixture": filename, "action": action, "science_units": cost,
                       "status": "completed", "passed": True})
    write_json(directory / "rows.json", rows)
    write_json(directory / "ledger.json", ledger)
    write_json(directory / "summary.json", {"admitted": True, **limits,
               "checks_passed": 27, "rows": rows})
    return directory / "summary.json", hashes, packages


def test_scientific_admission_revalidates_saved_evidence(science_evidence):
    summary, hashes, packages = science_evidence
    assert m11.validate_science_admission(summary, hashes, packages)["admitted"] is True
    sources = m11.science_admission_sources(summary)
    assert "base/initial/structure.xtalproj" in sources
    assert "compound/prepare/result.json" in sources
    m11.verify_admission_sources(sources)
    # Documentation and tests can change after the science check without changing its evidence.
    assert m11.validate_science_admission(summary, {**hashes, "docs/report.md": "new-doc-hash",
                                                   "tests/test_m11.py": "new-test-hash"}, packages)["admitted"]


def noop_result(**patch):
    outcome = {"ok": False, "atoms_before": 120, "atoms_after": 120,
               "diagnostics": [{"code": "NOTHING_TO_DO", "level": "info"}], **patch}
    return {"errors": [], "events": [{"action": {"verb": "prepare", "steps": ["duplicates"]},
                                      "result": outcome}]}


def test_explicit_base_noop_is_an_expected_preparation_contract(science_evidence):
    summary, hashes, packages = science_evidence
    target = summary.parent / "base/prepare/result.json"
    raw = read_json(target)
    raw["events"][0]["result"] = noop_result()["events"][0]["result"]
    write_json(target, raw)
    assert m11.preparation_contract_ok(raw["events"][0]["result"], "base")
    assert m11.validate_science_admission(summary, hashes, packages)["admitted"]


@pytest.mark.parametrize("patch,variant", [
    ({"diagnostics": [{"code": "PREPARE_STEP", "level": "info"}]}, "base"),
    ({"atoms_after": 119}, "base"),
    ({"atoms_before": 0, "atoms_after": 0}, "base"),
    ({"atoms_before": True, "atoms_after": True}, "base"),
    ({"atoms_before": None, "atoms_after": None}, "base"),
    ({"diagnostics": [{"code": "NOTHING_TO_DO", "level": "error"}]}, "base"),
    ({"diagnostics": [{"code": "NOTHING_TO_DO"}]}, "base"),
    ({"ok": None}, "base"),
    ({}, "duplicate"),
    ({}, "compound"),
])
def test_noop_exception_rejects_failed_or_mutated_preparation(patch, variant):
    assert not m11.preparation_contract_ok(noop_result(**patch)["events"][0]["result"], variant)


def test_successful_preparation_contract_is_accepted_for_each_variant():
    assert all(m11.preparation_contract_ok({"ok": True}, variant)
               for variant in ["base", "duplicate", "compound"])


@pytest.mark.parametrize("file,patch", [
    ("summary.json", {"admitted": False}),
    ("summary.json", {"prepare_actions": 2}),
    ("summary.json", {"science_units": 3}),
    ("summary.json", {"requested_uff_steps": 1}),
    ("summary.json", {"model_calls": 1}),
    ("summary.json", {"checks_passed": 26}),
    ("manifest.json", {"source_and_input_hashes": {}}),
    ("manifest.json", {"packages": {}}),
    ("base/prepare/result.json", {"errors": [{"message": "failure"}]}),
    ("base/prepare/result.json", {"events": []}),
    ("base/prepare/result.json", {"checkpoint_history": []}),
])
def test_science_admission_rejects_incomplete_or_changed_records(science_evidence, file, patch):
    summary, hashes, packages = science_evidence
    target = summary.parent / file
    write_json(target, {**read_json(target), **patch})
    with pytest.raises(ValueError, match="scientific admission"):
        m11.validate_science_admission(summary, hashes, packages)


def test_science_admission_rejects_reserved_cost_and_false_case(science_evidence):
    summary, hashes, packages = science_evidence
    target = summary.parent / "ledger.json"
    ledger = read_json(target)
    ledger[1]["status"] = "reserved"
    write_json(target, ledger)
    with pytest.raises(ValueError, match="ledger or case checks"):
        m11.validate_science_admission(summary, hashes, packages)
    ledger[1]["status"] = "completed"
    ledger[2]["science_units"] = 1
    write_json(target, ledger)
    with pytest.raises(ValueError, match="ledger or case checks"):
        m11.validate_science_admission(summary, hashes, packages)


def test_science_admission_rejects_raw_artifact_changes(science_evidence):
    summary, hashes, packages = science_evidence
    (summary.parent / "duplicate/prepare/structure.xtalproj").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw worker evidence"):
        m11.validate_science_admission(summary, hashes, packages)


def test_science_admission_rejects_runtime_source_changes(science_evidence):
    summary, hashes, packages = science_evidence
    (m11.ROOT / "rsi/runtime.py").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen experiment content changed"):
        m11.validate_science_admission(summary, hashes, packages)


def test_unknown_stage_usage_stops_after_first_episode(tmp_path, monkeypatch):
    """Exercise orchestration with fake decisions; no CLI or worker is invoked."""
    summary = admitted_summary(tmp_path)
    monkeypatch.setattr(m11, "git_state", lambda path: {"commit": "clean", "status": ""})
    monkeypatch.setattr(m11, "frozen_hashes", lambda: ({}, {}))
    monkeypatch.setattr(m11, "validate_science_admission", lambda *args: {})
    monkeypatch.setattr(m11, "science_admission_sources", lambda path: {})
    monkeypatch.setattr(m11.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(m11, "cli_identity", lambda executable: FAKE_CLI)
    candidate = {**m11.baseline_policy("m8"), "shortlist_size": 1}
    calls = []

    class Proposer:
        def __init__(self, config, output, stage):
            self.stage = stage
            self.observed_tokens = 2
            self.backend = type("Backend", (), {"executable": "fake"})()

        def model(self, prompt, schema, directory):
            calls.append("proposal")
            entry = self.stage.reserve("proposal")
            entry.update(status="known", tokens=2)
            self.stage.save()
            return {"workflow": candidate, "hypothesis": "Offline test proposal"}

    class Episode(Proposer):
        def __init__(self, config, output, stage):
            super().__init__(config, output, stage)
            self.output = output
            self.calls = 1
            self.observed_tokens = 0
            self.ledger = [{"status": "reserved"}]

        def episode(self, pool, arm, policy):
            calls.append((pool["id"], arm))
            entry = self.stage.reserve("episode")
            entry.update(status="unknown", tokens=0)
            self.stage.save()
            raise RuntimeError("Simulated unknown CLI usage")

    monkeypatch.setattr(m11, "ThreeArmExperiment", Proposer)
    monkeypatch.setattr(m11, "WorkflowScreeningExperiment", Episode)
    output = tmp_path / "run"
    m11.run(output, summary, summary)
    assert len(calls) == 2
    assert calls[0] == "proposal"
    assert read_json(output / "selection.json")["promoted"] is False
    assert read_json(output / "accepted.json")["source"] == "m8"
    assert read_json(output / "rows.json")[0]["cost"]["unknown_usage"] == 1
    assert read_json(output / "stage-ledger.json")[-1]["status"] == "unknown"


@pytest.mark.parametrize("protocol,limit,valid", [
    ("original", 10000, True), ("local-v2", 30000, True),
    ("original", 30000, False), ("local-v2", 10000, False),
    ("arbitrary", 30000, False),
])
def test_admission_allocation_is_versioned(protocol, limit, valid):
    manifest = {"admission_protocol": protocol, "max_observed_tokens": limit}
    if valid:
        assert m11.admission_limit(manifest) == limit
    else:
        with pytest.raises(ValueError):
            m11.admission_limit(manifest)


@pytest.mark.parametrize("tokens,admitted", [(19825, True), (30000, True), (30001, False)])
def test_local_v2_admission_budget_boundary(tmp_path, tokens, admitted):
    summary = admitted_summary(tmp_path)
    manifest = read_json(tmp_path / "manifest.json")
    manifest.update(admission_protocol="local-v2", max_observed_tokens=30000)
    write_json(tmp_path / "manifest.json", manifest)
    record = read_json(summary)
    record["tokens"] = tokens
    write_json(summary, record)
    usage = {"input_tokens": tokens - 3, "output_tokens": 3}
    result = read_json(tmp_path / "probe/result.json")
    result["usage"] = usage
    write_json(tmp_path / "probe/result.json", result)
    (tmp_path / "probe/events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": usage}) + "\n", encoding="utf-8")
    write_json(tmp_path / "ledger.json", [{"label": "admission", "status": "known", "usage": usage, "result_status": "ok"}])
    if admitted:
        assert m11.validate_admission(summary)["tokens"] == tokens
    else:
        with pytest.raises(ValueError):
            m11.validate_admission(summary)

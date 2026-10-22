"""Checks for independent selection and strict model call accounting."""

import pytest

from rsi.experiment import Experiment, load_parent, normalize_actions, promotion, public_observation, write_json


def test_no_promotion_for_ties_missing_results_or_regression():
    assert not promotion({"a": True}, {"a": True})["promoted"]
    assert not promotion({"a": False}, {})["promoted"]
    assert not promotion({"a": True, "b": False, "c": False}, {"a": False, "b": True, "c": True})["promoted"]
    assert promotion({"a": False}, {"a": True})["promoted"]


def test_private_task_fields_not_sent_as_observations():
    assert public_observation({"expect": "secret", "reference_actions": [1], "diagnostic_codes": ["X"]}) == {"diagnostic_codes": ["X"], "artifact": {"saved": False}}


def test_observation_includes_evidence_of_saved_artifact_without_private_path():
    observation = public_observation({"artifact": {"path": "private/path", "sha256": "abc"}})
    assert observation["artifact"] == {"saved": True, "sha256": "abc"}


def test_optimization_is_bounded():
    with pytest.raises(ValueError):
        normalize_actions([{"verb": "optimize", "max_steps": 0, "steps": []}])


def test_failed_calls_consume_budget(tmp_path):
    class Backend:
        def run(self, *args):
            return {"status": "process_error", "usage": {}}

    experiment = Experiment({"max_llm_calls": 1, "max_observed_tokens": 100}, tmp_path / "run", Backend())
    with pytest.raises(RuntimeError, match="CLI failure"):
        experiment.model("test", {}, experiment.output / "call")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        experiment.model("test", {}, experiment.output / "second")
    assert experiment.calls == 1


def test_parent_prompt_cannot_be_changed_without_updating_version_hash(tmp_path):
    import hashlib

    path = tmp_path / "accepted.json"
    write_json(path, {"prompt": "original", "sha256": hashlib.sha256(b"original").hexdigest()})
    assert load_parent(path)["prompt"] == "original"
    write_json(path, {"prompt": "changed", "sha256": hashlib.sha256(b"original").hexdigest()})
    with pytest.raises(ValueError, match="hash"):
        load_parent(path)


def test_cli_identity_uses_selected_backend_not_default_path(tmp_path, monkeypatch):
    import hashlib
    from rsi.experiment import cli_identity

    selected = tmp_path / "selected-cli.exe"
    selected.write_bytes(b"selected binary")
    seen = []

    def version(argv, **kwargs):
        seen.append(argv)
        return "selected-version\n"

    monkeypatch.setattr("rsi.experiment.subprocess.check_output", version)
    identity = cli_identity(str(selected))
    assert seen == [[str(selected.resolve()), "--version"]]
    assert identity["sha256"] == hashlib.sha256(b"selected binary").hexdigest()
    assert identity["version"] == "selected-version"

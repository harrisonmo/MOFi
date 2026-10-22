"""Admission, loss-mask, immutability and harness transport regression tests."""

import json
from pathlib import Path

import pytest

from rsi.llm import core
from rsi.llm.worker import encode, checkpoint_identity
from rsi.llm.transport import LocalQwen


def rows():
    return [{"id": split, "group_id": split, "split": split, "source": "synthetic unit fixture",
             "verification": "Engineering fixture only; no scientific label",
             "messages": [{"role": "user", "content": "Input " + split},
                          {"role": "assistant", "content": '{"stop": true}'}]}
            for split in ("train", "development")]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "repository", lambda path: {"commit": "a" * 40, "dirty": False})
    config = core.read_json(core.ROOT / "configs/llm-qwen3-8b-qlora.json")
    core.write_json(tmp_path / "config.json", config)
    (tmp_path / "data.jsonl").write_text("\n".join(map(json.dumps, rows())))
    run = tmp_path / "run"
    core.prepare(tmp_path / "config.json", tmp_path / "data.jsonl", run)
    return run


@pytest.mark.parametrize("mutation", [
    lambda r: r[1].update(split="final_holdout"),
    lambda r: r[1].update(group_id="train"),
    lambda r: r[1].update(id="train"),
    lambda r: r[0].update(verification=""),
    lambda r: r[1]["messages"][0].update(content="Input train"),
    lambda r: r[0]["messages"][-1].update(content="<think>invented</think>answer"),
])
def test_reject_leakage_and_unverified_targets(mutation):
    data = rows()
    mutation(data)
    with pytest.raises(ValueError):
        core.admit_rows(data)


def test_admission_freezes_files_and_rejects_reuse(bundle):
    _, manifest = core.verify_bundle(bundle)
    assert manifest["split_counts"] == {"train": 1, "development": 1}
    assert manifest["promotion_status"] == "not_evaluated"
    with pytest.raises(FileExistsError):
        core.prepare(bundle / "config.json", bundle / "data.jsonl", bundle)
    (bundle / "data.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Changed data"):
        core.verify_bundle(bundle)


class Tokenizer:
    unk_token_id = -1
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 2, 3]
    def encode(self, text, **kwargs):
        return [4, 5]
    def convert_tokens_to_ids(self, token):
        return 6


def test_completion_loss_masks_prompt_and_retains_end_token():
    item = encode(Tokenizer(), rows()[0], 6)
    assert item["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert item["labels"] == [-100, -100, -100, 4, 5, 6]
    with pytest.raises(ValueError, match="truncation"):
        encode(Tokenizer(), rows()[0], 5)


class FakeScheduler:
    def __init__(self, status="completed", text='{"stop": true}', cap=False):
        self.status, self.text, self.cap = status, text, cap
        self.specs = []
    def run(self, spec, backend):
        self.specs.append(spec)
        path = Path(spec["cwd"])
        if "rsi.llm.inference" in spec["argv"]:
            core.write_json(path / "generation.json", {"text": self.text, "cap_hit": self.cap,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}})
        else:
            (path / "adapter").mkdir()
            (path / "adapter/adapter.safetensors").write_bytes(b"fake-test-only")
            core.write_json(path / "training_summary.json", {"adapter_files": core.files_manifest(path / "adapter")})
        return {"id": "job-0001", "status": self.status, "exit_code": 0}


def test_train_uses_scheduler_and_does_not_promote(bundle):
    scheduler = FakeScheduler()
    result = core.execute(bundle, bundle / "model", scheduler)
    assert result["status"] == "trained"
    assert result["promotion_status"] == "not_evaluated"
    assert scheduler.specs[0]["resources"]["wall_seconds"] == 3600
    with pytest.raises(FileExistsError):
        core.execute(bundle, bundle / "model", scheduler)


def test_failed_training_is_not_a_candidate(bundle):
    result = core.execute(bundle, bundle / "model", FakeScheduler("timeout"))
    assert result["status"] == "timeout"


@pytest.mark.parametrize("status,text,cap,expected", [
    ("completed", '{"stop": true}', False, "ok"),
    ("completed", '{"stop": "yes"}', False, "invalid_output"),
    ("completed", "not JSON", False, "invalid_output"),
    ("completed", '{"stop": true}', True, "output_limit"),
    ("timeout", "", False, "timeout"),
])
def test_transport_preserves_schema_failures_and_unknown_usage(bundle, status, text, cap, expected):
    scheduler = FakeScheduler(status, text, cap)
    transport = LocalQwen(bundle / "model", bundle / "config.json", scheduler)
    schema = {"type": "object", "properties": {"stop": {"type": "boolean"}}, "required": ["stop"]}
    result = transport.run("Stop", schema, bundle / "decision")
    assert result["status"] == expected
    if status == "timeout":
        assert result["usage"] is None
    assert scheduler.specs[0]["resources"]["wall_seconds"] == 180


def test_changed_adapter_refused_before_inference(bundle):
    core.execute(bundle, bundle / "model", FakeScheduler())
    (bundle / "adapter/adapter.safetensors").write_bytes(b"changed")
    transport = LocalQwen(bundle / "model", bundle / "config.json", FakeScheduler(), candidate=bundle)
    with pytest.raises(ValueError, match="adapter changed"):
        transport.run("Stop", {"type": "object"}, bundle / "decision")


@pytest.mark.parametrize("key,value", [("max_steps", 0), ("revision", "main"), ("learning_rate", float("nan")),
                                      ("seed", True), ("model", "Qwen/Qwen3-4B")])
def test_invalid_config(bundle, key, value):
    config = core.read_json(bundle / "config.json")
    config[key] = value
    with pytest.raises(ValueError):
        core.validate_config(config)


def test_wrong_base_revision_rejected(bundle):
    model = bundle / "model"
    model.mkdir()
    core.write_json(model / "model_source.json", {"id": "Qwen/Qwen3-8B", "sha": "0" * 40})
    with pytest.raises(ValueError, match="pinned"):
        checkpoint_identity(model, core.read_json(bundle / "config.json"))


def test_experiment_accepts_qwen_transport_and_accounts_tokens(bundle):
    from rsi.experiment import Experiment
    client = LocalQwen(bundle / "model", bundle / "config.json", FakeScheduler())
    experiment = Experiment({"max_llm_calls": 1, "max_observed_tokens": 100},
                            bundle / "experiment", backend=client)
    result = experiment.model("Stop", {"type": "object"}, experiment.output / "call")
    assert result == {"stop": True}
    assert experiment.observed_tokens == 15
    assert client.identity()["model"] == "Qwen/Qwen3-8B"
    with pytest.raises(RuntimeError, match="budget"):
        experiment.model("Stop", {"type": "object"}, experiment.output / "call-2")

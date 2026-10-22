"""Offline contract tests for the CLI decision boundary."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rsi.backends import codex_cli


SCHEMA = {
    "type": "object", "properties": {"choice": {"type": "string"}},
    "required": ["choice"], "additionalProperties": False,
}

REAL_WINDOWS_JOB = codex_cli._WindowsJob


@pytest.fixture(autouse=True)
def owned_job(monkeypatch):
    job = Mock()
    monkeypatch.setattr(codex_cli, "_WindowsJob", Mock(return_value=job))
    if codex_cli.os.name != "nt":
        monkeypatch.setattr(codex_cli.os, "killpg", Mock())
    return job


def install_process(monkeypatch, output='{"choice":"A"}', events=None, code=0,
                    stderr="", timeout=False):
    process = Mock(pid=91827)
    process.poll.return_value = code
    if events is None:
        events = [{"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}]
    serialized = "\n".join(json.dumps(event) for event in events)

    def spawn(argv, **kwargs):
        if output is not None:
            Path(argv[argv.index("--output-last-message") + 1]).write_text(output, encoding="utf-8")
        process.communicate.side_effect = (
            [subprocess.TimeoutExpired(argv, 1), (serialized, stderr)]
            if timeout else None
        )
        process.communicate.return_value = (serialized, stderr)
        return process

    factory = Mock(side_effect=spawn)
    monkeypatch.setattr(codex_cli.subprocess, "Popen", factory)
    return process, factory


def test_success_preserves_auth_boundary_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    monkeypatch.setenv("OTHER_API_KEY", "another-test-secret")
    process, factory = install_process(monkeypatch)
    result = codex_cli.CodexCLI(executable="codex.exe").run("private prompt", SCHEMA, tmp_path)
    assert result["status"] == "ok"
    assert result["response"] == {"choice": "A"}
    assert result["usage"]["input_tokens"] == 7
    assert result["actual_model"] is None
    argv = factory.call_args.args[0]
    kwargs = factory.call_args.kwargs
    assert "private prompt" not in " ".join(argv)
    assert argv[-1] == "-"
    assert {"--ephemeral", "--ignore-user-config", "--output-schema"} <= set(argv)
    assert argv[argv.index("-s") + 1] == "read-only"
    assert kwargs["shell"] is False
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "OTHER_API_KEY" not in kwargs["env"]
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert "private prompt" in process.communicate.call_args.args[0]
    assert "private prompt" not in (tmp_path / "command.json").read_text()
    assert not list(tmp_path.glob("*.writing"))


@pytest.mark.parametrize("output", ['{"choice":3}', 'not json', '{"choice":"A","extra":1}'])
def test_invalid_output(monkeypatch, tmp_path, output):
    install_process(monkeypatch, output=output)
    result = codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "invalid_output"
    assert result["response"] is None


def test_missing_final_is_invalid_even_for_nullable_schema(monkeypatch, tmp_path):
    install_process(monkeypatch, output=None)
    assert codex_cli.CodexCLI().run("prompt", {}, tmp_path)["status"] == "invalid_output"


@pytest.mark.parametrize("tool_type", sorted(codex_cli.FORBIDDEN_ITEMS))
def test_forbidden_tools_fail_even_on_success(monkeypatch, tmp_path, tool_type):
    install_process(monkeypatch, events=[{"type": "item.started", "item": {"type": tool_type}}])
    result = codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "unexpected_tool_use"
    assert result["unexpected_tool_types"] == [tool_type]
    assert result["response"] is None


@pytest.mark.parametrize(("message", "status"), [
    ("401 unauthorized", "auth_error"),
    ("model_not_found", "model_unavailable"),
    ("transport failed", "process_error"),
])
def test_failed_event_overrides_zero_exit(monkeypatch, tmp_path, message, status):
    install_process(monkeypatch, events=[{"type": "turn.failed", "error": {"message": message}}])
    assert codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)["status"] == status


def test_timeout_cleans_up_and_retains_logs(monkeypatch, tmp_path, owned_job):
    process, _ = install_process(monkeypatch, timeout=True)
    cleanup = Mock()
    monkeypatch.setattr(codex_cli, "_stop_process_tree", cleanup)
    result = codex_cli.CodexCLI(timeout=1).run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "timeout"
    assert result["response"] is None
    cleanup.assert_called_once_with(process, owned_job if codex_cli.os.name == "nt" else None)
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "stderr.log").exists()
    assert result["duration_seconds"] >= 0


def test_model_evidence_is_not_inferred_from_request(monkeypatch, tmp_path):
    install_process(monkeypatch, events=[{"type": "session.started", "session": {"model": "reported-model"}}])
    result = codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)
    assert result["actual_model"] == "reported-model"
    assert result["requested_model"] == "gpt-6-astra"
    assert result["model_evidence"]


def test_missing_executable_is_a_recorded_process_error(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_cli.subprocess, "Popen", Mock(side_effect=FileNotFoundError("missing executable")))
    result = codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "process_error"
    assert result["exit_code"] is None


def test_nonzero_exit_never_accepts_valid_final(monkeypatch, tmp_path):
    install_process(monkeypatch, code=1, stderr="login required")
    result = codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "auth_error"
    assert result["response"] is None
    assert result["exit_code"] == 1


def test_malformed_event_stream_is_rejected(monkeypatch, tmp_path):
    process, _ = install_process(monkeypatch)
    original_popen = codex_cli.subprocess.Popen

    def spawn(*args, **kwargs):
        original_popen(*args, **kwargs)
        process.communicate.return_value = ("not a JSON event", "")
        return process

    monkeypatch.setattr(codex_cli.subprocess, "Popen", spawn)
    assert codex_cli.CodexCLI().run("prompt", SCHEMA, tmp_path)["status"] == "invalid_output"


def test_invocation_directory_cannot_be_reused(monkeypatch, tmp_path):
    install_process(monkeypatch)
    backend = codex_cli.CodexCLI()
    backend.run("prompt", SCHEMA, tmp_path)
    with pytest.raises(FileExistsError):
        backend.run("second prompt", SCHEMA, tmp_path)


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment")
def test_cleanup_without_job_only_targets_live_owned_root(monkeypatch):
    process = Mock(pid=91827)
    process.poll.return_value = 0
    runner = Mock()
    monkeypatch.setattr(codex_cli.subprocess, "run", runner)
    codex_cli._stop_process_tree(process)
    runner.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment")
def test_windows_cleanup_closes_job_even_after_root_exit():
    process = Mock(pid=91827)
    process.poll.return_value = 0
    job = Mock()
    codex_cli._stop_process_tree(process, job)
    job.close.assert_called_once_with()
    process.kill.assert_not_called()


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment")
def test_parent_exit_with_inherited_pipe_timeout_closes_job(monkeypatch, tmp_path, owned_job):
    process, _ = install_process(monkeypatch, timeout=True, code=0)
    result = codex_cli.CodexCLI(timeout=1).run("prompt", SCHEMA, tmp_path)
    assert result["status"] == "timeout"
    owned_job.assign.assert_called_once_with(process)
    owned_job.close.assert_called_once_with()
    process.kill.assert_not_called()


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment")
def test_job_assignment_failure_never_sends_prompt(monkeypatch, tmp_path, owned_job):
    process, _ = install_process(monkeypatch)
    process.poll.side_effect = [None, 0, 0]
    owned_job.assign.side_effect = OSError("assignment denied")
    result = codex_cli.CodexCLI().run("private prompt", SCHEMA, tmp_path)
    assert result["status"] == "process_error"
    assert result["response"] is None
    process.communicate.assert_not_called()
    process.kill.assert_called_once_with()
    owned_job.close.assert_called_once_with()


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment")
def test_job_creation_failure_does_not_launch_process(monkeypatch, tmp_path):
    _, factory = install_process(monkeypatch)
    monkeypatch.setattr(codex_cli, "_WindowsJob", Mock(side_effect=OSError("containment unavailable")))
    result = codex_cli.CodexCLI().run("private prompt", SCHEMA, tmp_path)
    assert result["status"] == "process_error"
    factory.assert_not_called()


def test_posix_cleanup_targets_owned_group_after_leader_exit(monkeypatch):
    killpg = Mock()
    monkeypatch.setattr(codex_cli, "os", SimpleNamespace(name="posix", killpg=killpg))
    monkeypatch.setattr(codex_cli, "signal", SimpleNamespace(SIGKILL=9))
    process = Mock(pid=91827)
    process.poll.return_value = 0
    codex_cli._stop_process_tree(process)
    killpg.assert_called_once_with(91827, codex_cli.signal.SIGKILL)


@pytest.mark.skipif(codex_cli.os.name != "nt", reason="Windows process containment integration")
def test_real_windows_job_kills_orphan_with_inherited_pipe():
    """Exercise OS containment with local Python processes; never launch Codex."""
    import psutil

    job = REAL_WINDOWS_JOB()
    script = (
        "import subprocess,sys; sys.stdin.readline(); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(child.pid,flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    child = None
    try:
        job.assign(process)
        process.stdin.write("start\n")
        process.stdin.flush()
        child = psutil.Process(int(process.stdout.readline().strip()))
        process.wait(timeout=5)
        assert process.poll() == 0
        assert child.is_running()
        codex_cli._stop_process_tree(process, job)
        process.communicate(timeout=5)
        child.wait(timeout=5)
    finally:
        job.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()

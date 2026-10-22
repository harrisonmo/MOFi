"""Offline transport configuration and credential isolation regression."""
import inspect
import json
from pathlib import Path
from unittest.mock import Mock
from rsi import q2b_http_transport as transport
from rsi.q2b_v21 import protocol

def test_http_transport_keeps_auth_and_no_proxy(monkeypatch, tmp_path):
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

    assert 'model_provider="q2b_openai_http"' in argv
    assert "model_providers.q2b_openai_http.supports_websockets=false" in argv
    assert "model_providers.q2b_openai_http.request_max_retries=0" in argv
    assert "model_providers.q2b_openai_http.stream_max_retries=0" in argv
    assert not any("base_url" in arg or "env_key" in arg for arg in argv)

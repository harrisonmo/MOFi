"""Offline contract tests for the OpenAI and Anthropic API decision backends."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")
import httpx2

from rsi.backends.factory import backend_identity, create_backend
from rsi.backends.api import AnthropicAPI, OpenAIAPI


SCHEMA = {
    "type": "object", "properties": {"choice": {"type": "string"}},
    "required": ["choice"], "additionalProperties": False,
}


def anthropic_client(text='{"choice":"A"}', stop_reason="end_turn", content=None, error=None):
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "stop_reason": stop_reason,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "usage": {"input_tokens": 11, "output_tokens": 4,
                  "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3},
    }
    create = Mock(side_effect=error) if error else Mock(return_value=message)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def responses_client(text='{"choice":"B"}', status="completed", output=None):
    response = {
        "id": "resp_1", "model": "gpt-6-astra-snapshot", "status": status,
        "output": output if output is not None else [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 6},
    }
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))


def chat_client(text='{"choice":"C"}', finish_reason="stop", message=None):
    completion = {
        "id": "chat_1", "model": "qwen3-8b",
        "choices": [{"finish_reason": finish_reason,
                     "message": message or {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 8},
    }
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion))))


def status_error(cls, code):
    request = httpx2.Request("POST", "https://api.example.invalid/v1")
    return cls("failure", response=httpx2.Response(code, request=request), body=None)


def test_anthropic_native_schema_effort_and_usage(tmp_path):
    client = anthropic_client()
    backend = AnthropicAPI(reasoning="high", client=client)
    result = backend.run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == "ok" and result["response"] == {"choice": "A"}
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"] == {"effort": "high",
                                       "format": {"type": "json_schema", "schema": SCHEMA}}
    assert "thinking" not in kwargs and "fallbacks" not in kwargs
    # Cached prompt tokens count as input, matching the CLI ledger total.
    assert result["usage"]["input_tokens"] == 16 and result["usage"]["output_tokens"] == 4
    assert result["actual_model"] == "claude-opus-5-5"
    for name in ("prompt.txt", "schema.json", "request.json", "response.json", "final.json", "result.json"):
        assert (tmp_path / "call" / name).exists()
    assert json.loads((tmp_path / "call" / "final.json").read_text()) == {"choice": "A"}


def test_api_key_is_never_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    backend = AnthropicAPI()
    backend._client = anthropic_client()
    backend.run("Pick one.", SCHEMA, tmp_path / "call")
    request = json.loads((tmp_path / "call" / "request.json").read_text())
    assert request["authentication"] == "environment_variable:ANTHROPIC_API_KEY"
    for path in (tmp_path / "call").iterdir():
        assert "sk-ant-secret-value" not in path.read_text(encoding="utf-8")


def test_missing_key_fails_without_request(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = OpenAIAPI().run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == "auth_error" and "OPENAI_API_KEY" in result["error"]
    assert json.loads((tmp_path / "call" / "final.json").read_text()) is None


@pytest.mark.parametrize("stop_reason, expected", [("refusal", "refusal"), ("max_tokens", "invalid_output")])
def test_anthropic_stop_reasons_are_not_ok(tmp_path, stop_reason, expected):
    backend = AnthropicAPI(client=anthropic_client(stop_reason=stop_reason))
    result = backend.run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == expected and result["response"] is None
    assert result["stop_reason"] == stop_reason


def test_anthropic_tool_blocks_are_rejected(tmp_path):
    content = [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]
    backend = AnthropicAPI(client=anthropic_client(content=content, stop_reason="tool_use"))
    assert backend.run("Pick one.", SCHEMA, tmp_path / "call")["status"] == "unexpected_tool_use"


@pytest.mark.parametrize("error, expected", [
    (lambda: status_error(anthropic.AuthenticationError, 401), "auth_error"),
    (lambda: status_error(anthropic.NotFoundError, 404), "model_unavailable"),
    (lambda: status_error(anthropic.RateLimitError, 429), "rate_limited"),
    (lambda: status_error(anthropic.InternalServerError, 500), "api_error"),
    (lambda: anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x.invalid")), "timeout"),
])
def test_anthropic_errors_map_to_statuses(tmp_path, error, expected):
    backend = AnthropicAPI(client=anthropic_client(error=error()))
    result = backend.run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == expected and (tmp_path / "call" / "result.json").exists()


def test_schema_violation_is_invalid_output(tmp_path):
    backend = AnthropicAPI(client=anthropic_client(text='{"choice": 3}'))
    assert backend.run("Pick one.", SCHEMA, tmp_path / "call")["status"] == "invalid_output"


def test_openai_responses_structured_output(tmp_path):
    client = responses_client()
    backend = OpenAIAPI(reasoning="high", client=client)
    result = backend.run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == "ok" and result["response"] == {"choice": "B"}
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["schema"] == SCHEMA
    assert result["usage"]["input_tokens"] == 20 and result["actual_model"] == "gpt-6-astra-snapshot"


def test_openai_responses_incomplete_and_tool_calls(tmp_path):
    incomplete = OpenAIAPI(client=responses_client(status="incomplete"))
    assert incomplete.run("x", SCHEMA, tmp_path / "a")["status"] == "invalid_output"
    tools = OpenAIAPI(client=responses_client(output=[{"type": "function_call", "name": "f"}]))
    assert tools.run("x", SCHEMA, tmp_path / "b")["status"] == "unexpected_tool_use"


def test_openai_compatible_chat_without_key_in_prompt_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = chat_client(text='```json\n{"choice":"C"}\n```')
    backend = OpenAIAPI(model="qwen3-8b", endpoint="chat_completions", api_key_env=None,
                        base_url="http://127.0.0.1:8000/v1", reasoning=None,
                        structured_output="prompt", client=client)
    result = backend.run("Pick one.", SCHEMA, tmp_path / "call")
    assert result["status"] == "ok" and result["response"] == {"choice": "C"}
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs and "reasoning_effort" not in kwargs
    assert '"required": [' in kwargs["messages"][0]["content"]
    assert result["usage"] == {"input_tokens": 30, "output_tokens": 8,
                               "provider_usage": {"prompt_tokens": 30, "completion_tokens": 8}}
    request = json.loads((tmp_path / "call" / "request.json").read_text())
    assert request["authentication"] == "none" and request["base_url"] == "http://127.0.0.1:8000/v1"


def test_openai_chat_refusal_and_length(tmp_path):
    refused = OpenAIAPI(endpoint="chat_completions",
                        client=chat_client(message={"role": "assistant", "content": None, "refusal": "no"}))
    assert refused.run("x", SCHEMA, tmp_path / "a")["status"] == "refusal"
    truncated = OpenAIAPI(endpoint="chat_completions", client=chat_client(finish_reason="length"))
    assert truncated.run("x", SCHEMA, tmp_path / "b")["status"] == "invalid_output"


def test_output_directory_is_single_use(tmp_path):
    backend = AnthropicAPI(client=anthropic_client())
    backend.run("x", SCHEMA, tmp_path / "call")
    with pytest.raises(FileExistsError):
        backend.run("x", SCHEMA, tmp_path / "call")


def test_factory_and_identity():
    cli = create_backend({"model": "gpt-6-astra", "reasoning": "high", "cli_timeout_seconds": 60})
    assert type(cli).__name__ == "CodexCLI"
    api = create_backend({"backend": "anthropic_api", "model": "claude-opus-5-5", "reasoning": "high",
                          "cli_timeout_seconds": 90, "api": {"max_output_tokens": 8000}})
    assert isinstance(api, AnthropicAPI) and api.timeout == 90 and api.max_output_tokens == 8000
    identity = backend_identity(api)
    assert identity["kind"] == "api" and identity["provider"] == "anthropic"
    assert identity["version"].startswith("anthropic ")
    compatible = create_backend({"backend": "openai_api", "model": "qwen3-8b", "reasoning": None,
                                 "api": {"endpoint": "chat_completions", "api_key_env": None,
                                         "base_url": "http://127.0.0.1:8000/v1"}})
    assert compatible.endpoint == "chat_completions" and compatible.api_key_env is None
    with pytest.raises(ValueError):
        create_backend({"backend": "unknown", "model": "m"})


def test_experiment_ledger_accepts_api_backend(tmp_path):
    from rsi.experiment import Experiment

    config = {"backend": "openai_api", "model": "gpt-6-astra", "reasoning": "high",
              "cli_timeout_seconds": 60, "max_llm_calls": 2, "max_observed_tokens": 10_000}
    exp = Experiment(config, tmp_path / "run", backend=create_backend(config))
    assert isinstance(exp.backend, OpenAIAPI)
    exp.backend._client = responses_client()
    assert exp.model("Pick one.", SCHEMA, exp.output / "call-1") == {"choice": "B"}
    assert exp.calls == 1 and exp.observed_tokens == 26

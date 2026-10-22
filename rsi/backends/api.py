"""Run independent, structured decisions through official provider APIs.

These backends implement the same ``run(prompt, schema, output_dir)`` contract
as :class:`rsi.backends.codex_cli.CodexCLI`, so controllers can switch between
the authenticated CLI and a documented provider API without other changes.

API keys are read from a named environment variable at call time. They are
never written to disk; persisted metadata records only the variable name.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

import jsonschema

from rsi.backends.codex_cli import DECISION_INSTRUCTIONS, _atomic_text


STRUCTURED_MODES = {"native", "prompt"}
OPENAI_ENDPOINTS = {"responses", "chat_completions"}
SCHEMA_NAME = "decision"


def _schema_prompt(schema: dict) -> str:
    """Embed the schema for servers without native JSON-schema constraints."""
    return ("\n\nReturn exactly one JSON value, with no surrounding prose or code fences, "
            "that conforms to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2))


def _parse_json_text(text: str) -> Any:
    stripped = text.strip()
    # Tolerate a single fenced block in prompt mode; schema validation still applies.
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in stripped else ""
    return json.loads(stripped)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class _APIBackend:
    """Shared persistence, validation and status mapping for provider APIs."""

    provider = ""
    sdk_package = ""
    default_key_env = ""

    def __init__(self, model: str, reasoning: str | None = "high", timeout: float = 180,
                 base_url: str | None = None, api_key_env: str | None = None,
                 max_output_tokens: int = 16000, max_retries: int = 0,
                 structured_output: str = "native", strict_schema: bool = False,
                 request_options: dict | None = None, client: Any = None):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if structured_output not in STRUCTURED_MODES:
            raise ValueError(f"structured_output must be one of {sorted(STRUCTURED_MODES)}")
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self.base_url = base_url
        # An explicit None means the server needs no key, such as a local endpoint.
        self.api_key_env = self.default_key_env if api_key_env == "" else api_key_env
        self.max_output_tokens = max_output_tokens
        # SDK retries are hidden extra requests, so the default keeps one request per call.
        self.max_retries = max_retries
        self.structured_output = structured_output
        self.strict_schema = strict_schema
        self.request_options = dict(request_options or {})
        self._client = client

    def identity(self) -> dict[str, Any]:
        """Describe the backend for manifests without exposing credentials."""
        try:
            version = importlib.metadata.version(self.sdk_package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {"kind": "api", "provider": self.provider, "endpoint": self.endpoint,
                "base_url": self.base_url, "sdk": self.sdk_package,
                "version": f"{self.sdk_package} {version}", "requested_model": self.model,
                "reasoning": self.reasoning, "structured_output": self.structured_output,
                "max_retries": self.max_retries}

    @property
    def endpoint(self) -> str:
        raise NotImplementedError

    def _api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env) or None

    def client(self):
        raise NotImplementedError

    def _request(self, text: str, schema: dict) -> dict[str, Any]:
        """Return raw, text, usage, actual_model and an optional non-ok status."""
        raise NotImplementedError

    def _error_status(self, exc: Exception) -> str:
        raise NotImplementedError

    def run(self, prompt: str, schema: dict, output_dir: Path) -> dict[str, Any]:
        """Return validated output and persist an auditable, credential-free call log.

        Each output directory belongs to one invocation and must be unused.
        """
        started = time.monotonic()
        jsonschema.validators.validator_for(schema).check_schema(schema)
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "request.json").exists():
            raise FileExistsError(f"API output directory already used: {directory}")
        text = DECISION_INSTRUCTIONS + prompt
        if self.structured_output == "prompt":
            text += _schema_prompt(schema)
        _atomic_text(directory / "schema.json", json.dumps(schema, ensure_ascii=False, indent=2))
        _atomic_text(directory / "prompt.txt", text)
        metadata = {**self.identity(), "timeout_seconds": self.timeout,
                    "max_output_tokens": self.max_output_tokens,
                    "request_options": self.request_options,
                    "authentication": (f"environment_variable:{self.api_key_env}"
                                       if self.api_key_env else "none")}
        _atomic_text(directory / "request.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        result: dict[str, Any] = {
            "status": "api_error", "response": None, "usage": None,
            "duration_seconds": 0.0, "exit_code": None,
            "requested_model": self.model, "actual_model": None,
            "model_evidence": [], "unexpected_tool_types": [],
            "provider": self.provider, "stop_reason": None, "error": None,
        }
        reply: dict[str, Any] | None = None
        if self.api_key_env is not None and self._api_key() is None and self._client is None:
            result["status"] = "auth_error"
            result["error"] = f"Environment variable {self.api_key_env} is not set"
        else:
            try:
                reply = self._request(text, schema)
            except Exception as exc:  # Every failure is persisted, never raised past the ledger.
                result["status"] = self._error_status(exc)
                result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        if reply is not None:
            _atomic_text(directory / "response.json",
                         json.dumps(reply["raw"], ensure_ascii=False, indent=2, default=str))
            result["usage"] = reply["usage"]
            result["stop_reason"] = reply.get("stop_reason")
            if reply.get("actual_model"):
                result["actual_model"] = reply["actual_model"]
                result["model_evidence"].append({"source": "api_response", "model": reply["actual_model"]})
            if reply.get("status"):
                result["status"] = reply["status"]
            else:
                try:
                    response = _parse_json_text(reply["text"])
                    jsonschema.validate(response, schema)
                    result["response"] = response
                    result["status"] = "ok"
                except (ValueError, jsonschema.ValidationError) as exc:
                    result["status"] = "invalid_output"
                    result["error"] = str(exc)[:2000]
        final = result["response"] if result["status"] == "ok" else None
        _atomic_text(directory / "final.json", json.dumps(final, ensure_ascii=False, indent=2) + "\n")
        result["duration_seconds"] = time.monotonic() - started
        if result["duration_seconds"] >= self.timeout:
            result["status"] = "timeout"
            result["response"] = None
        _atomic_text(directory / "result.json", json.dumps(result, ensure_ascii=False, indent=2))
        return result


class AnthropicAPI(_APIBackend):
    """Claude through the Messages API with ``output_config`` structured output.

    Server-side refusal fallbacks are deliberately not enabled: a silent model
    switch would break the requested-versus-actual model record. A refusal is
    reported as status ``refusal`` instead.
    """

    provider = "anthropic"
    sdk_package = "anthropic"
    default_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-opus-5-5", **kwargs):
        kwargs.setdefault("api_key_env", self.default_key_env)
        super().__init__(model, **kwargs)

    @property
    def endpoint(self) -> str:
        return "messages"

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key(), base_url=self.base_url,
                timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _request(self, text: str, schema: dict) -> dict[str, Any]:
        output_config: dict[str, Any] = {}
        if self.reasoning:
            output_config["effort"] = self.reasoning
        if self.structured_output == "native":
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": text}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        kwargs.update(self.request_options)
        message = self.client().messages.create(**kwargs)
        raw = _dump(message)
        usage = raw.get("usage") or {}
        # Count cached prompt tokens as input, matching the CLI's input_tokens total.
        input_tokens = sum(usage.get(key) or 0 for key in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        reply = {
            "raw": raw, "actual_model": raw.get("model"), "stop_reason": raw.get("stop_reason"),
            "usage": {"input_tokens": input_tokens, "output_tokens": usage.get("output_tokens") or 0,
                      "provider_usage": usage},
            "text": "".join(block.get("text", "") for block in raw.get("content") or []
                            if block.get("type") == "text"),
        }
        tool_types = sorted({block.get("type") for block in raw.get("content") or []
                             if block.get("type") not in {"text", "thinking", "redacted_thinking"}})
        if raw.get("stop_reason") == "refusal":
            reply["status"] = "refusal"
        elif tool_types:
            reply["status"] = "unexpected_tool_use"
        elif raw.get("stop_reason") == "max_tokens":
            reply["status"] = "invalid_output"
        return reply

    def _error_status(self, exc: Exception) -> str:
        import anthropic

        if isinstance(exc, anthropic.APITimeoutError):
            return "timeout"
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return "auth_error"
        if isinstance(exc, anthropic.NotFoundError):
            return "model_unavailable"
        if isinstance(exc, anthropic.RateLimitError):
            return "rate_limited"
        if isinstance(exc, anthropic.APIConnectionError):
            return "connection_error"
        return "api_error"


class OpenAIAPI(_APIBackend):
    """OpenAI, or any OpenAI-compatible server, through Responses or Chat Completions.

    Set ``base_url`` for compatible servers (for example a local vLLM endpoint)
    and ``api_key_env=None`` when that server requires no key. Many compatible
    servers implement only ``chat_completions``.
    """

    provider = "openai"
    sdk_package = "openai"
    default_key_env = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-6-astra", endpoint: str = "responses", **kwargs):
        if endpoint not in OPENAI_ENDPOINTS:
            raise ValueError(f"endpoint must be one of {sorted(OPENAI_ENDPOINTS)}")
        kwargs.setdefault("api_key_env", self.default_key_env)
        self._endpoint = endpoint
        super().__init__(model, **kwargs)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self._api_key() or "not-required", base_url=self.base_url,
                timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _request(self, text: str, schema: dict) -> dict[str, Any]:
        if self._endpoint == "responses":
            return self._responses(text, schema)
        return self._chat(text, schema)

    def _responses(self, text: str, schema: dict) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "input": text,
                                  "max_output_tokens": self.max_output_tokens}
        if self.reasoning:
            kwargs["reasoning"] = {"effort": self.reasoning}
        if self.structured_output == "native":
            kwargs["text"] = {"format": {"type": "json_schema", "name": SCHEMA_NAME,
                                         "schema": schema, "strict": self.strict_schema}}
        kwargs.update(self.request_options)
        response = self.client().responses.create(**kwargs)
        raw = _dump(response)
        usage = raw.get("usage") or {}
        parts, tool_types, refused = [], set(), False
        for item in raw.get("output") or []:
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if content.get("type") == "output_text":
                        parts.append(content.get("text", ""))
                    elif content.get("type") == "refusal":
                        refused = True
            elif item.get("type") != "reasoning":
                tool_types.add(item.get("type"))
        reply = {
            "raw": raw, "actual_model": raw.get("model"),
            "stop_reason": raw.get("status"),
            "usage": {"input_tokens": usage.get("input_tokens") or 0,
                      "output_tokens": usage.get("output_tokens") or 0, "provider_usage": usage},
            "text": "".join(parts),
        }
        if refused:
            reply["status"] = "refusal"
        elif tool_types:
            reply["status"] = "unexpected_tool_use"
        elif raw.get("status") not in (None, "completed"):
            reply["status"] = "invalid_output"
        return reply

    def _chat(self, text: str, schema: dict) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model,
                                  "messages": [{"role": "user", "content": text}],
                                  "max_completion_tokens": self.max_output_tokens}
        if self.reasoning:
            kwargs["reasoning_effort"] = self.reasoning
        if self.structured_output == "native":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": SCHEMA_NAME, "schema": schema, "strict": self.strict_schema}}
        kwargs.update(self.request_options)
        completion = self.client().chat.completions.create(**kwargs)
        raw = _dump(completion)
        usage = raw.get("usage") or {}
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        reply = {
            "raw": raw, "actual_model": raw.get("model"), "stop_reason": choice.get("finish_reason"),
            "usage": {"input_tokens": usage.get("prompt_tokens") or 0,
                      "output_tokens": usage.get("completion_tokens") or 0, "provider_usage": usage},
            "text": message.get("content") or "",
        }
        if message.get("refusal"):
            reply["status"] = "refusal"
        elif message.get("tool_calls") or message.get("function_call"):
            reply["status"] = "unexpected_tool_use"
        elif choice.get("finish_reason") not in (None, "stop"):
            reply["status"] = "invalid_output"
        return reply

    def _error_status(self, exc: Exception) -> str:
        import openai

        if isinstance(exc, openai.APITimeoutError):
            return "timeout"
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return "auth_error"
        if isinstance(exc, openai.NotFoundError):
            return "model_unavailable"
        if isinstance(exc, openai.RateLimitError):
            return "rate_limited"
        if isinstance(exc, openai.APIConnectionError):
            return "connection_error"
        return "api_error"

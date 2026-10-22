# Provider API backends

The authenticated Codex CLI remains the default decision
backend. Two opt-in backends call documented provider APIs through the official
Python SDKs:

| `backend` | Class | Provider surface | Structured output |
|---|---|---|---|
| `codex_cli` (default) | `rsi.backends.codex_cli.CodexCLI` | `codex exec` with existing login | `--output-schema` |
| `openai_api` | `rsi.backends.api.OpenAIAPI` | OpenAI Responses or Chat Completions; any OpenAI-compatible server via `base_url` | `text.format` / `response_format` JSON schema |
| `anthropic_api` | `rsi.backends.api.AnthropicAPI` | Claude Messages API | `output_config.format` JSON schema |

All three implement `run(prompt, schema, output_dir)` and return the same result
fields (`status`, `response`, `usage.input_tokens`, `usage.output_tokens`,
`requested_model`, `actual_model`, `model_evidence`, `duration_seconds`, ...), so
`rsi.experiment.Experiment.model` and its budget ledger accept any of them. Every
response is validated locally with `jsonschema`, whatever the provider guarantees.

`rsi/experiment.py` and `rsi/backends/__init__.py` are hash-frozen M7 sources, so
backend selection lives in `rsi/backends/factory.py` and is passed in explicitly:

```python
from rsi.backends.factory import backend_identity, create_backend
from rsi.experiment import Experiment

backend = create_backend(config)
experiment = Experiment(config, output, backend=backend)
manifest["backend"] = backend_identity(backend)  # credential-free description
```

## Installation

```powershell
uv sync --extra api
```

The SDKs are optional. The CLI backend and all other tests do not need them;
`tests/test_api_backends.py` skips when they are absent.

## Configuration

`rsi.backends.factory.create_backend(config)` reads `backend`, `model`, `reasoning` and
`cli_timeout_seconds` (the per-call limit for every backend), plus optional API
settings under `api`:

| `api` key | Default | Meaning |
|---|---|---|
| `api_key_env` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Environment variable holding the key; `null` for servers that need none |
| `base_url` | provider default | Alternative endpoint, such as a local vLLM server |
| `endpoint` (OpenAI only) | `responses` | `responses` or `chat_completions` |
| `max_output_tokens` | `16000` | Output cap, including reasoning tokens |
| `max_retries` | `0` | SDK retries; non-zero values hide extra requests from the ledger |
| `structured_output` | `native` | `native` uses the provider schema constraint; `prompt` embeds the schema in the prompt for servers without one |
| `strict_schema` (OpenAI only) | `false` | OpenAI strict mode; requires every property listed in `required` and `additionalProperties: false` |
| `request_options` | `{}` | Extra request fields passed through verbatim and recorded |

`reasoning` maps to `output_config.effort` (Anthropic), `reasoning.effort`
(Responses) or `reasoning_effort` (Chat Completions). Set it to `null` for
models or servers without a reasoning control.

Examples:

```json
{"backend": "anthropic_api", "model": "claude-opus-5-5", "reasoning": "high",
 "cli_timeout_seconds": 300}
```

```json
{"backend": "openai_api", "model": "gpt-6-astra", "reasoning": "high",
 "cli_timeout_seconds": 300}
```

```json
{"backend": "openai_api", "model": "Qwen/Qwen3-8B", "reasoning": null,
 "cli_timeout_seconds": 300,
 "api": {"base_url": "http://127.0.0.1:8000/v1", "endpoint": "chat_completions",
         "api_key_env": null}}
```

The last form is the intended route for the planned strong/Qwen3-8B experiments
once a local server is admitted.

## Audit records

Each call writes `prompt.txt`, `schema.json`, `request.json` (backend identity,
limits and the key's variable name, never its value), `response.json` (the raw
provider response), `final.json` and `result.json`. The actual model comes from
the provider response and is recorded separately from the requested model.
`backend_identity(backend)` returns the provider, endpoint, base URL, SDK
version, requested model and reasoning setting for run manifests.

## Status values

`ok`, `invalid_output` (schema violation, truncation or an incomplete
response), `refusal`, `unexpected_tool_use`, `auth_error` (including a missing
key, detected before any request), `model_unavailable`, `rate_limited`,
`timeout`, `connection_error` and `api_error`. Every status other than `ok`
fails the call in `Experiment.model` after the ledger entry is written.

## Deliberate limits

- Claude refusal fallbacks (`fallbacks`) are not enabled: a server-side model
  switch would break the requested-versus-actual model record. Enable them only
  through `request_options` and a recorded protocol decision.
- Usage counts cached prompt tokens as input, matching the CLI's
  `input_tokens`; the raw provider usage is kept in `usage.provider_usage`.
- Existing runners are unchanged. `Experiment.run` and the frozen protocol
  runners (M6–M11, Q2/Q2B/Q2C, Blind-G1 file exchange) fingerprint a CLI
  executable and stay CLI-bound; new controllers opt in through the factory.
- API calls are billed to the key's account. Record the actual backend and
  model for every experiment, as with the CLI.

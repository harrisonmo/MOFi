"""Select a decision backend from configuration.

Kept outside ``rsi/backends/__init__.py`` and ``rsi/experiment.py`` because
both files are hash-frozen M7 sources.
"""

from __future__ import annotations

BACKENDS = ("codex_cli", "openai_api", "anthropic_api")


def _create_backend(config: dict):
    """Build the decision backend named by ``config["backend"]`` (default: Codex CLI).

    API backends read optional settings from ``config["api"]``; see
    docs/API_BACKENDS.md. Timeouts reuse ``cli_timeout_seconds`` so a config can
    switch backends without changing its per-call limit.
    """
    kind = config.get("backend", "codex_cli")
    timeout = config.get("cli_timeout_seconds", 180)
    if kind == "codex_cli":
        from rsi.backends.codex_cli import CodexCLI

        return CodexCLI(model=config["model"], reasoning=config["reasoning"], timeout=timeout,
                        executable=config.get("cli_executable"))
    options = dict(config.get("api") or {})
    if kind == "openai_api":
        from rsi.backends.api import OpenAIAPI

        return OpenAIAPI(model=config["model"], reasoning=config.get("reasoning"), timeout=timeout, **options)
    if kind == "anthropic_api":
        from rsi.backends.api import AnthropicAPI

        return AnthropicAPI(model=config["model"], reasoning=config.get("reasoning"), timeout=timeout, **options)
    raise ValueError(f"Unknown backend {kind!r}; expected one of {BACKENDS}")


def create_backend(config: dict):
    """Build the selected backend with optional frozen Valency literature context.

    ``literature`` requires evidence_path and evidence_sha256; no query or OAuth
    happens during a model decision. Existing configs retain their behavior.
    """
    backend = _create_backend(config)
    literature = config.get("literature")
    if literature is not None:
        if (not isinstance(literature, dict)
                or {"evidence_path", "evidence_sha256"} - set(literature)
                or set(literature) - {"evidence_path", "evidence_sha256", "max_context_bytes"}):
            raise ValueError("literature requires evidence_path, evidence_sha256 and optional max_context_bytes")
        from rsi.literature.grounding import GroundedBackend
        return GroundedBackend(backend, **literature)
    return backend


def backend_identity(backend) -> dict:
    """Fingerprint a CLI executable, or describe an API backend without credentials."""
    identity = getattr(backend, "identity", None)
    if callable(identity):
        return identity()
    from rsi.experiment import cli_identity

    return cli_identity(backend.executable)

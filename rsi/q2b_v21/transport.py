"""Run independent, structured decisions through the authenticated Codex CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

import jsonschema


FORBIDDEN_ITEMS = {"command_execution", "mcp_tool_call", "web_search", "file_change"}
DECISION_INSTRUCTIONS = (
    "Act only as a structured decision engine. Do not use any tools, commands, "
    "connectors, web search, or file access. Do not inspect the workspace. "
    "Return only JSON conforming to the supplied output schema. The external "
    "controller performs all scientific and filesystem operations.\n\n"
)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class _WindowsJob:
    """Own a kill-on-close job; pywin32 is required only on Windows."""

    def __init__(self):
        self.handle = None
        try:
            import win32api
            import win32job

            self.api = win32api
            self.jobs = win32job
            self.handle = win32job.CreateJobObject(None, "")
            limits = win32job.QueryInformationJobObject(
                self.handle, win32job.JobObjectExtendedLimitInformation,
            )
            limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(
                self.handle, win32job.JobObjectExtendedLimitInformation, limits,
            )
        except Exception as exc:
            self.close()
            raise OSError(f"Unable to create Windows process containment job: {exc}") from exc

    def assign(self, process: subprocess.Popen) -> None:
        try:
            # Use the owned process handle, never reopen a potentially reused PID.
            self.jobs.AssignProcessToJobObject(self.handle, int(process._handle))
        except Exception as exc:
            raise OSError(f"Unable to assign CLI to Windows process containment job: {exc}") from exc

    def close(self) -> None:
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _stop_process_tree(process: subprocess.Popen, job: _WindowsJob | None = None) -> None:
    """Terminate the invocation's owned container, including orphan descendants."""
    if os.name == "nt":
        if job is not None:
            job.close()
        # Assignment failure leaves only the newly spawned, unprompted root.
        if process.poll() is None:
            process.kill()
    else:
        try:
            # Popen started a new session, so its PID identifies our process group.
            # The group can still contain descendants after the leader has exited.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _error_status(message: str) -> str:
    lowered = message.lower()
    if any(term in lowered for term in (
        "unauthorized", "authentication", "not logged in", "login required",
        "invalid_api_key", "401", "please log in",
    )):
        return "auth_error"
    if any(term in lowered for term in (
        "model_not_found", "model not found", "model is not supported",
        "unsupported model", "model unavailable", "does not exist",
        "not supported when using codex with a chatgpt account",
    )):
        return "model_unavailable"
    return "process_error"


class CodexCLI:
    """Use existing CLI login without exposing or provisioning credentials."""

    def __init__(self, model: str = "gpt-6-astra", reasoning: str = "high",
                 timeout: float = 180, executable: str | None = None):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self.executable = executable or shutil.which("codex") or "codex"

    def run(self, prompt: str, schema: dict, output_dir: Path) -> dict[str, Any]:
        """Return validated output and persist an auditable, prompt-free command log.

        Each output directory belongs to one invocation and must be unused.
        Duration includes startup, cleanup, log persistence and validation.
        """
        started = time.monotonic()
        jsonschema.validators.validator_for(schema).check_schema(schema)
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        cwd = directory / "workspace"
        cwd.mkdir(exist_ok=False)
        schema_path = directory / "schema.json"
        temporary_final = directory / ".final.pending.json"
        _atomic_text(schema_path, json.dumps(schema, ensure_ascii=False, indent=2))
        _atomic_text(directory / "prompt.txt", DECISION_INSTRUCTIONS + prompt)
        command = [
            self.executable, "--no-daemon", "exec", "--strict-config", "--model", self.model, "--json",
            "--output-schema", str(schema_path),
            "--output-last-message", str(temporary_final),
            "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
            "-c", "skills.bundled.enabled=false",
            "--disable", "shell_snapshot", "--disable", "unbounded_connection_retries",
            "-s", "read-only", "-C", str(cwd),
            "-c", f'model_reasoning_effort={json.dumps(self.reasoning)}', "-",
        ]
        metadata = {
            "argv": command, "cwd": str(cwd), "prompt_transport": "stdin",
            "requested_model": self.model, "reasoning": self.reasoning,
            "timeout_seconds": self.timeout, "authentication": "existing_cli_login",
        }
        _atomic_text(directory / "command.json", json.dumps(metadata, indent=2))
        environment = {
            key: value for key, value in os.environ.items()
            if not key.upper().endswith("API_KEY")
            and key.upper() not in {"OPENAI_BASE_URL", "OPENAI_ACCESS_TOKEN"}
        }
        environment["PYTHONUTF8"] = "1"
        result: dict[str, Any] = {
            "status": "process_error", "response": None, "usage": None,
            "duration_seconds": 0.0, "exit_code": None,
            "requested_model": self.model, "actual_model": None,
            "model_evidence": [], "unexpected_tool_types": [],
        }
        stdout, stderr = "", ""
        process = None
        job = None
        cleaned_up = False
        process_failed = False
        timed_out = False
        try:
            if os.name == "nt":
                job = _WindowsJob()
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=str(cwd), env=environment,
                text=True, encoding="utf-8", errors="replace", shell=False,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            if job is not None:
                job.assign(process)
            remaining = max(0.001, self.timeout - (time.monotonic() - started))
            stdout, stderr = process.communicate(DECISION_INSTRUCTIONS + prompt, timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_tree(process, job)
            cleaned_up = True
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.output or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
        except OSError as exc:
            process_failed = True
            stderr = str(exc)
        finally:
            if process is not None:
                if not cleaned_up:
                    _stop_process_tree(process, job)
                # Reap the owned root after cleanup without sending any prompt.
                if process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process_failed = True
                        stderr += "\nCLI root did not exit after process containment cleanup"
            elif job is not None:
                job.close()
        if process is not None:
            result["exit_code"] = process.poll()
        _atomic_text(directory / "events.jsonl", stdout)
        _atomic_text(directory / "stderr.log", stderr)
        failures: list[str] = []
        malformed_events = False
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("event must be an object")
            except (ValueError, TypeError):
                malformed_events = True
                continue
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") in FORBIDDEN_ITEMS:
                result["unexpected_tool_types"].append(item["type"])
            if event.get("type") == "turn.completed":
                result["usage"] = event.get("usage")
            if event.get("type") in {"error", "turn.failed"}:
                failures.append(json.dumps(event, ensure_ascii=False))
            for source in (event, event.get("session"), event.get("response")):
                if isinstance(source, dict) and isinstance(source.get("model"), str):
                    result["actual_model"] = source["model"]
                    result["model_evidence"].append({"event_type": event.get("type"), "model": source["model"]})
        final_exists = temporary_final.exists()
        if final_exists:
            temporary_final.replace(directory / "final.json")
        else:
            _atomic_text(directory / "final.json", "null\n")
        if timed_out:
            result["status"] = "timeout"
        elif result["unexpected_tool_types"]:
            result["status"] = "unexpected_tool_use"
        elif process_failed or result["exit_code"] != 0 or failures:
            result["status"] = _error_status(stderr + "\n" + "\n".join(failures))
        elif malformed_events or not final_exists:
            result["status"] = "invalid_output"
        else:
            try:
                response = json.loads((directory / "final.json").read_text(encoding="utf-8"))
                jsonschema.validate(response, schema)
                result["response"] = response
                result["status"] = "ok"
            except (ValueError, jsonschema.ValidationError):
                result["status"] = "invalid_output"
        result["duration_seconds"] = time.monotonic() - started
        if result["duration_seconds"] >= self.timeout:
            result["status"] = "timeout"
            result["response"] = None
        _atomic_text(directory / "result.json", json.dumps(result, ensure_ascii=False, indent=2))
        return result

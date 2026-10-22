"""Drop-in structured-decision transport for frozen Qwen and trained candidates.

The harness still executes scientific actions and grades results independently.
Each invocation reloads weights in a fresh, scheduler-contained GPU process.
"""

import json
from pathlib import Path
import time

import jsonschema

from rsi.hpc.core import ROOT, integer, read_json, write_json
from .core import files_manifest, validate_config, verify_bundle


class LocalQwen:
    def __init__(self, model_path, config, scheduler, candidate=None, backend="local",
                 max_new_tokens=512, timeout=180):
        self.model_path = str(Path(model_path).resolve())
        self.config = validate_config(read_json(config))
        self.scheduler, self.backend = scheduler, backend
        self.candidate = str(Path(candidate).resolve()) if candidate else None
        integer(max_new_tokens, "max_new_tokens", 1)
        integer(timeout, "timeout", 1)
        if max_new_tokens >= self.config["max_length"]:
            raise ValueError("Generation must leave room for the prompt")
        self.max_new_tokens, self.timeout = max_new_tokens, timeout

    def identity(self):
        return {"backend": "transformers_nf4", "model": self.config["model"],
                "revision": self.config["revision"], "candidate": self.candidate,
                "enable_thinking": False, "max_length": self.config["max_length"],
                "max_new_tokens": self.max_new_tokens, "timeout_seconds": self.timeout}

    def run(self, prompt, schema, output_dir):
        jsonschema.validators.validator_for(schema).check_schema(schema)
        if self.candidate:
            config, manifest = verify_bundle(self.candidate)
            if config != self.config or manifest["status"] != "trained":
                raise ValueError("Candidate must be trained and match the inference config")
            summary = read_json(Path(self.candidate) / "training_summary.json")
            if files_manifest(Path(self.candidate) / "adapter") != summary["adapter_files"]:
                raise ValueError("Candidate adapter changed")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        request = {"prompt": prompt, "schema": schema, "config": self.config,
                   "model_path": self.model_path, "candidate": self.candidate,
                   "max_new_tokens": self.max_new_tokens}
        write_json(output / "request.json", request)
        resources = dict(self.config["resources"], wall_seconds=self.timeout)
        spec = {"name": "qwen3-decision", "argv": ["{python}", "-m", "rsi.llm.inference", str(output)],
                "cwd": str(output), "log": "worker.log", "resources": resources,
                "env": {"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}}
        started = time.monotonic()
        result = {"status": "process_error", "response": None, "usage": None,
                  "requested_model": self.config["model"], "backend": "transformers_nf4",
                  "candidate": self.candidate}
        try:
            job = self.scheduler.run(spec, self.backend)
            write_json(output / "hpc-job.json", job)
            if job["status"] != "completed":
                result["status"] = "timeout" if job["status"] in {"timeout", "queue_timeout"} else "process_error"
            else:
                raw = read_json(output / "generation.json")
                result["usage"] = raw["usage"]
                result["status"] = "output_limit" if raw["cap_hit"] else "invalid_output"
                if not raw["cap_hit"]:
                    response = json.loads(raw["text"])
                    jsonschema.validate(response, schema)
                    result.update(status="ok", response=response)
        except (ValueError, jsonschema.ValidationError) as exc:
            result["error"] = str(exc)
        finally:
            result["duration_seconds"] = time.monotonic() - started
            write_json(output / "result.json", result)
        return result

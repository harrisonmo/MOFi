"""CPU-only admission and scheduler integration; GPU dependencies are worker-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from rsi.hpc.core import ROOT, exact_keys, integer, positive, read_json, write_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files_manifest(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if not files:
        raise ValueError(f"Empty checkpoint: {directory}")
    return {p.relative_to(directory).as_posix(): digest(p) for p in files}


def repository(path):
    path = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
            text=True, stderr=subprocess.DEVNULL).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def validate_config(config):
    keys = {"schema_version", "candidate_id", "parent_id", "scientific_target", "model", "revision",
            "seed", "max_length", "max_steps", "gradient_accumulation", "learning_rate",
            "rank", "alpha", "dropout", "resources"}
    exact_keys(config, keys, keys, "QLoRA config")
    if config["schema_version"] != 1 or config["model"] != "Qwen/Qwen3-8B":
        raise ValueError("Only schema 1 and Qwen/Qwen3-8B are supported")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin an immutable Hugging Face revision")
    for key in ("candidate_id", "parent_id", "scientific_target"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be nonempty")
    for key in ("max_length", "max_steps", "gradient_accumulation", "rank", "alpha"):
        integer(config[key], key, 1)
    integer(config["seed"], "seed")
    positive(config["learning_rate"], "learning_rate")
    if not isinstance(config["dropout"], (float, int)) or not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    resources = config["resources"]
    exact_keys(resources, {"cpus", "gpus", "memory_mb", "wall_seconds"},
               {"cpus", "gpus", "memory_mb", "wall_seconds"}, "resources")
    for key in resources:
        integer(resources[key], key, 1)
    if resources["gpus"] != 1:
        raise ValueError("This trainer requires exactly one GPU")
    return config


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def admit_rows(rows):
    """Accept explicitly curated targets; isolate families and duplicate prompts across splits.

    Verification is an input assertion, not an independent scientific judgment.
    Only train/development rows enter the trainer. Held-out labels stay outside this bundle.
    """
    if not rows:
        raise ValueError("Empty training bundle")
    identifiers, groups, prompts = set(), {}, {}
    splits = set()
    for row in rows:
        keys = {"id", "group_id", "split", "source", "verification", "messages"}
        exact_keys(row, keys, keys, "training row")
        for key in ("id", "group_id", "source", "verification"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
        if row["id"] in identifiers:
            raise ValueError("Duplicate example id")
        identifiers.add(row["id"])
        split = row["split"]
        if split not in {"train", "development"}:
            raise ValueError("Promotion/final-held-out data must not enter training")
        splits.add(split)
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("A prompt and final assistant completion are required")
        for message in messages:
            exact_keys(message, {"role", "content"}, {"role", "content"}, "message")
            if message["role"] not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Unsupported message role")
            if not isinstance(message["content"], str) or not message["content"].strip():
                raise ValueError("Empty message")
        if messages[-1]["role"] != "assistant" or messages[-2]["role"] not in {"user", "tool"}:
            raise ValueError("Supervise only a final assistant response after user/tool input")
        # Preserve reasoning/control tokens in the template, never inject them through targets.
        if any(token in messages[-1]["content"] for token in ("<|im_", "<think>", "</think>")):
            raise ValueError("Targets must contain answers, not chat control or thinking tokens")
        prompt = json.dumps(messages[:-1], sort_keys=True)
        for key, index in ((row["group_id"], groups), (prompt, prompts)):
            if key in index and index[key] != split:
                raise ValueError("Cross-split group or prompt leakage")
            index[key] = split
    if splits != {"train", "development"}:
        raise ValueError("Both train and development splits are required")
    return rows


def prepare(config_path, data_path, output):
    config = validate_config(read_json(config_path))
    rows = admit_rows(read_rows(data_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    (output / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"schema_version": 1, "status": "prepared", "promotion_status": "not_evaluated",
                "repositories": {"root": repository(ROOT), "crystal_builder": repository(ROOT / "Crystal-Builder")},
                "config_sha256": digest(output / "config.json"),
                "data_sha256": digest(output / "data.jsonl"),
                "source_data_sha256": digest(data_path),
                "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "development")},
                "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_bundle(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    for name in ("config", "data"):
        path = run / ("config.json" if name == "config" else "data.jsonl")
        if digest(path) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Changed {name} after admission")
    current = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    if current != manifest["implementation"]:
        raise ValueError("Trainer implementation changed after preparation")
    return validate_config(read_json(run / "config.json")), manifest


def execute(run, model_path, scheduler, backend=None):
    """One immutable attempt, charged by the existing local/Slurm scheduler."""
    run, model_path = Path(run).resolve(), Path(model_path).resolve()
    config, manifest = verify_bundle(run)
    # Exclusive marker prevents concurrent execution and implicit restart after a failed job.
    with (run / "execution.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "reserved", "model_path": str(model_path)}, stream)
    spec = {"name": "qwen3-qlora", "argv": ["{python}", "-m", "rsi.llm.worker", str(run), str(model_path)],
            "cwd": str(ROOT), "resources": config["resources"]}
    # Logs are relative to cwd; run bundles may also live outside the checkout.
    spec["cwd"], spec["log"] = str(run), "worker.log"
    spec["env"] = {"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        job = scheduler.run(spec, backend)
        write_json(run / "execution.json", job)
        manifest["status"] = job["status"]
        manifest["hpc_job_id"] = job["id"]
        if job["status"] == "completed":
            if not (run / "training_summary.json").is_file():
                raise RuntimeError("Worker exited without a training summary")
            summary = read_json(run / "training_summary.json")
            if files_manifest(run / "adapter") != summary["adapter_files"]:
                raise ValueError("Adapter hashes do not match worker summary")
            manifest["status"] = "trained"
        write_json(run / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(run / "manifest.json", manifest)
        raise

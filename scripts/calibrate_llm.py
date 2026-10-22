"""Bounded engineering smoke: one QLoRA update and fresh-process baseline/adapter calls.

Synthetic targets exercise plumbing only. This is not a scientific improvement trial.
Run with the existing CUDA Python environment; no downloads or installations occur.
"""

import argparse
import json
from pathlib import Path

from rsi.hpc import Scheduler
from rsi.hpc.core import ROOT, read_json, write_json
from rsi.llm import prepare, execute
from rsi.llm.transport import LocalQwen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reload-candidate", help="Reuse a trained engineering candidate; do not train again")
    parser.add_argument("--inference-timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.inference_timeout <= 600:
        parser.error("inference-timeout must be between 1 and 600 seconds")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT / "configs/llm-qwen3-8b-qlora.json")
    config.update(candidate_id="qwen3-8b-engineering-smoke", max_steps=1, gradient_accumulation=1,
                  max_length=256, scientific_target="Synthetic transport calibration only; no scientific gain claim.")
    config["resources"].update(cpus=4, wall_seconds=600)
    candidate_path = Path(args.reload_candidate).resolve() if args.reload_candidate else output / "candidate"
    if args.reload_candidate:
        config = read_json(candidate_path / "config.json")
    write_json(output / "config.json", config)
    schema = {"type": "object", "properties": {"stop": {"type": "boolean"}},
              "required": ["stop"], "additionalProperties": False}
    rows = [{"id": s, "group_id": s, "split": s, "source": "synthetic calibration",
             "verification": "Hand-specified engineering output, not a scientific observation",
             "messages": [{"role": "system", "content": "Return JSON with a boolean stop field."},
                          {"role": "user", "content": prompt},
                          {"role": "assistant", "content": '{"stop": true}'}]}
            for s, prompt in (("train", "The job is finished. Stop."),
                              ("development", "No calculation budget remains. Stop."))]
    (output / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hpc = {"schema_version": 1,
           "backends": {"local": {"type": "local", "max_cpus": 4, "gpu_ids": ["0"], "max_memory_mb": 24000}},
           "session_budget": {"max_jobs": 2 if args.reload_candidate else 3,
                              "max_cpu_seconds": 4 * (2 * args.inference_timeout + (0 if args.reload_candidate else 600)),
                              "max_gpu_seconds": 2 * args.inference_timeout + (0 if args.reload_candidate else 600),
                              "max_wall_seconds": 2 * args.inference_timeout + (0 if args.reload_candidate else 600)}}
    write_json(output / "hpc.json", hpc)
    scheduler = Scheduler(output / "hpc.json", output / "session")
    if not args.reload_candidate:
        prepare(output / "config.json", output / "data.jsonl", candidate_path)
        result = execute(candidate_path, args.model_path, scheduler)
        if result["status"] != "trained":
            raise RuntimeError(f"Training failed: {result['status']}")
    responses = {}
    for name, candidate in (("baseline", None), ("adapter", candidate_path)):
        client = LocalQwen(args.model_path, output / "config.json", scheduler,
                           candidate=candidate, max_new_tokens=32, timeout=args.inference_timeout)
        responses[name] = client.run("Return exactly a JSON object with stop set to true.", schema, output / name)
    write_json(output / "calibration.json", {"responses": responses, "ledger": scheduler.status(),
               "scientific_claim": "None; synthetic one-update engineering smoke only."})
    if any(r["status"] != "ok" for r in responses.values()):
        raise RuntimeError("Structured decision calibration failed; inspect calibration.json")
    print(json.dumps({"status": "passed", "output": str(output)}))


if __name__ == "__main__":
    main()

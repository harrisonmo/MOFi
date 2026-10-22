"""Run a public scripted integration check, with no model or new calculation."""

import argparse
import json
import time
from pathlib import Path

from rsi.experiment import git_state
from rsi.tool_evidence import from_uff_event
from rsi.tool_runtime import Registry, digest, load_spec
from rsi.tool_trial import paired

ROOT = Path(__file__).resolve().parents[1]


def scripted_agent(spec):
    """An explicit harness fixture, never an Agent capability measurement."""
    def decide(stage, context):
        if stage == "plan":
            if spec["id"] in context["tools"]:
                response = [{"tool": spec["id"], "arguments": {}}]
            else:
                response = [{"tool": "expression", "arguments": expr}
                            for expr in spec["outputs"].values()]
        elif spec["id"] in context["tools"]:
            response = context["tool_results"][0]["answer"]["output"]
        else:
            response = {key: row["answer"]["output"] for key, row in
                        zip(spec["outputs"], context["tool_results"])}
        return {"response": response, "calls": 0, "tokens": 0, "wall_seconds": 0}
    return decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = load_spec(ROOT / "tools/q2/manual-audit-v1.json")
    fixtures = json.loads((ROOT / "benchmarks/q2/public-events.json").read_text())
    registry = Registry()
    started = time.perf_counter()
    record = registry.register(spec)
    registration_seconds = time.perf_counter() - started
    rows = []
    for task in fixtures["tasks"]:
        payload = from_uff_event(task["event"])["payload"]
        for row in paired(payload, spec, lambda: scripted_agent(spec)):
            row.update(task_id=task["id"], source_family=task["source_family"])
            rows.append(row)
    report = {
        "kind": "public_scripted_engineering_check",
        "root_git": git_state(ROOT), "crystal_builder_git": git_state(ROOT / "Crystal-Builder"),
        "frozen_files": {str(p): digest((ROOT / p).read_text()) for p in (
            "rsi/tool_runtime.py", "rsi/tool_trial.py", "rsi/tool_validation.py",
            "rsi/tool_evidence.py", "rsi/tool_smoke.py", "tools/q2/manual-audit-v1.json",
            "benchmarks/q2/public-events.json", "configs/q2_tool_pilot.json")},
        "registration": record, "rows": rows,
        "summary": {arm: {
            "episodes": sum(r["arm"] == arm for r in rows),
            "completed": sum(r["completed"] for r in rows if r["arm"] == arm),
            "correct": sum(r["passed"] for r in rows if r["arm"] == arm),
            "incorrect_reports": sum(r["completed"] and not r["passed"] for r in rows if r["arm"] == arm),
            "tool_errors": sum(len(r["errors"]) for r in rows if r["arm"] == arm),
            "tool_uses": sum(r["generated_or_manual_tool_uses"] for r in rows if r["arm"] == arm),
            "reuse_source_families": sorted({r["source_family"] for r in rows
                                           if r["arm"] == arm and r["generated_or_manual_tool_uses"]}),
            "tool_calls": sum(r["cost"]["tool_calls"] for r in rows if r["arm"] == arm),
        } for arm in ("fixed", "augmented", "removed")},
        "cost": {"new_model_calls": 0, "new_model_tokens": 0, "new_uff_steps": 0,
                 "registration_seconds": registration_seconds,
                 "replay_validation_and_use_seconds": sum(row["cost"]["wall_seconds"] for row in rows),
                 "authoring_and_debugging_tokens": None, "total_lifecycle_cost": None,
                 "historical_science_cost": "Reused M6 calibration, not charged as new science; see fixture provenance.",
                 "note": "Interactive authoring tokens and historical per-event costs unavailable. Not a zero-cost tool."},
        "conclusion": "Harness equivalence only. No outer-generated candidate or generalization evidence."
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "registry.json").write_text(json.dumps(registry.manifest(), indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if not all(row["passed"] for row in rows):
        raise SystemExit("Scripted integration failed")


if __name__ == "__main__":
    main()

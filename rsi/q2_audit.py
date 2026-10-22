"""Read-only post-run audit of frozen evidence, prompt parity and cost receipts."""

import argparse
import hashlib
import json
from pathlib import Path

from rsi.experiment import ROOT, read_json, git_state
from rsi.tool_runtime import digest
from rsi.tool_validation import reference, matches


def audit(directory):
    directory = Path(directory)
    manifest, tasks, rows, budget = [read_json(directory / name) for name in
                                    ("manifest.json", "tasks.json", "rows.json", "budget.json")]
    checks = {}
    checks["frozen_code"] = all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
                                for path, sha in manifest["frozen_files"].items())
    checks["frozen_scientific_sources"] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
                                              for path, sha in manifest["source_files"].items())
    checks["scientific_repository_unchanged"] = git_state(ROOT / "Crystal-Builder") == manifest["crystal_builder_git"]
    checks["frozen_tasks"] = digest(tasks) == manifest["task_manifest_sha256"]
    checks["planned_denominator"] = len(rows) == len(tasks) * 4 == 24
    checks["unique_pairs"] = len({(r["task_id"], r["arm"]) for r in rows}) == len(rows)
    checks["complete_budget"] = not budget["blocked"] and all(r["status"] != "reserved" for r in budget["entries"])
    models = [r for r in budget["entries"] if r["kind"] == "model"]
    checks["all_model_receipts"] = True
    checks["all_model_usage_events"] = True
    checks["no_forbidden_model_tools"] = True
    for row in models:
        result = read_json(directory / "model" / row["label"] / "result.json")
        checks["all_model_receipts"] &= result["status"] == row["status"] == "ok"
        events = [json.loads(line) for line in
                  (directory / "model" / row["label"] / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        usages = [e["usage"] for e in events if e.get("type") == "turn.completed"]
        checks["all_model_usage_events"] &= (len(usages) == 1 and usages[0] == row["usage"] and
                                              row["tokens"] == usages[0]["input_tokens"] + usages[0]["output_tokens"])
        checks["no_forbidden_model_tools"] &= not result["unexpected_tool_types"]
    checks["budget_caps"] = (len(models) <= 51 and sum(r["tokens"] for r in models) <= 1275000
                             and all(r["tokens"] <= 25000 for r in models)
                             and sum(r["stage"] == "author" for r in models) <= 3
                             and sum(r["stage"] == "research" for r in models) <= 48)
    checks["artifact_integrity"] = True
    checks["scientific_events_bound"] = True
    checks["payload_bound_to_scientific_values"] = True
    science = read_json(directory / "science-records.json")
    for record in science:
        result = read_json(directory / "science" / record["task_id"] / "result.json")
        artifact = result["artifact"]
        checks["artifact_integrity"] &= hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
        checks["scientific_events_bound"] &= digest(result) == record["result_sha256"]
        if record["eligible"]:
            events = [e for e in result["events"] if e["action"]["verb"] == "optimize" and e["result"].get("ok")]
            checks["scientific_events_bound"] &= digest(events[-1]) == record["evidence"]["event_sha256"]
            data = events[-1]["result"]["data"]
            expected = {key: data[key] for key in ("initial_energy", "energy", "max_force", "converged")}
            expected.update(energy_unit="kcal/mol", force_unit="kcal/mol/A")
            checks["payload_bound_to_scientific_values"] &= record["payload"] == expected
            for task in tasks:
                if task["id"].startswith(record["task_id"] + "-"):
                    payload = dict(expected)
                    if task["unit_condition"] == "SI":
                        for key in ("initial_energy", "energy"):
                            payload[key] *= 4.184
                        payload["max_force"] *= 41.84
                        payload.update(energy_unit="kJ/mol", force_unit="kJ/mol/nm")
                    checks["payload_bound_to_scientific_values"] &= task["payload"] == payload
    checks["independent_rescore"] = True
    checks["same_input_per_arm"] = True
    checks["fixed_removed_prompt_parity"] = True
    checks["enhanced_prompt_parity"] = True
    checks["unavailable_not_success"] = True
    checks["no_audit_tool_in_removed"] = True
    for task in tasks:
        paired = [r for r in rows if r["task_id"] == task["id"]]
        if not task["eligible"]:
            checks["unavailable_not_success"] &= all(not r["passed"] and not r["completed"] and r["cost"]["calls"] == 0 for r in paired)
            continue
        checks["same_input_per_arm"] &= all(r["input_hash"] == digest(task["payload"]) for r in paired)
        checks["independent_rescore"] &= all(r["passed"] == matches(r["report"], reference(task["payload"])) for r in paired)
        def prompt(arm):
            return (directory / "model" / (task["id"] + "-" + arm + "-plan") / "prompt.txt").read_text(encoding="utf-8")
        checks["fixed_removed_prompt_parity"] &= prompt("fixed") == prompt("removed")
        checks["enhanced_prompt_parity"] &= prompt("outer_generated") == prompt("manual_control")
        checks["no_audit_tool_in_removed"] &= all(t["request"]["tool"] != "uff_audit" for r in paired
                                                  if r["arm"] == "removed" for t in r["tool_trace"])
    checks["author_evaluation_labels_absent"] = all(
        "MIL53" not in p.read_text(encoding="utf-8") and "MIL88B" not in p.read_text(encoding="utf-8")
        for p in (directory / "model").glob("author-*/prompt.txt"))
    checks["no_reference_fields_in_prompts"] = all(
        "reference_actions" not in p.read_text(encoding="utf-8") and "reference_report" not in p.read_text(encoding="utf-8")
        for p in (directory / "model").glob("*/prompt.txt"))
    science_rows = [r for r in budget["entries"] if r["kind"] == "science"]
    checks["science_budget"] = len(science_rows) <= 4 and sum(r["requested_steps"] for r in science_rows) <= 4
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = audit(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

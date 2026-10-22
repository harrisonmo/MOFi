"""Read-only audit and compact export; never changes frozen experiment files."""
from collections import Counter
import re
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/q2b_v21"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit():
    manifest, completion = read("manifest.json"), read("completion.json")
    models, science = read("model_budget.json"), read("science_budget.json")
    pairs, episodes, authors = read("pairs.json"), read("episodes.json"), read("authors.json")
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    check("frozen_sources", all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest["files"].items()))
    check("source_file_count", len(manifest["source_hashes"]) == 4)
    check("source_hashes", all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in manifest["source_hashes"].items()))
    check("scientific_gate", manifest["gate"]["science_scoring_unchanged"] and manifest["gate"]["registered_evaluations"] == 48)
    check("dataset_hash", digest(pairs) == manifest["gate"]["dataset_hash"])
    check("no_unresolved_model_reservations", not any(r["status"] == "reserved" for r in models["entries"]))
    check("no_unresolved_science_reservations", not any(r["status"] == "reserved" for r in science["entries"]))
    check("clean_start", not manifest["root_git"]["status"] and not manifest["crystal_builder_git"]["status"])
    check("call_limit", len(models["entries"]) <= 197)
    call_rows = []
    stderr_diagnostics = []
    for row in models["entries"]:
        result = read("model/" + row["label"] + "/result.json")
        stderr = (RUN / "model" / row["label"] / "stderr.log").read_text(encoding="utf-8")
        if stderr.strip():
            stderr_diagnostics.append(dict(label=row["label"], text=stderr[-3000:]))
        usage = result.get("usage")
        known = isinstance(usage, dict) and all(isinstance(usage.get(k), int) for k in ("input_tokens", "output_tokens"))
        check("usage_" + row["label"], row["tokens"] == (usage["input_tokens"] + usage["output_tokens"] if known else None))
        check("status_" + row["label"], row["status"] == result["status"])
        call_rows.append({k: row.get(k) for k in ("stage", "label", "status", "usage", "tokens", "wall_seconds", "actual_model", "requested_model")})
    index = {p["id"]: p for p in pairs}
    for row in episodes:
        label = row["id"] + "-" + row["arm"]
        plan_file = RUN / "model" / (label + "-plan") / "result.json"
        plan_result = json.loads(plan_file.read_text(encoding="utf-8"))
        if plan_result.get("response"):
            bad_ids = [n["id"] for n in plan_result["response"]["program"]["nodes"] if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", n["id"])]
            if bad_ids:
                row["audit_invalid_node_ids"] = bad_ids
        path = RUN / "evidence" / (label + ".json")
        if path.exists():
            output = json.loads(path.read_text())
            check("input_binding_" + label, output["input_hash"] == digest([index[row["id"]]["A"], index[row["id"]]["B"]]))
            plan = read("model/" + label + "-plan/result.json")["response"]["program"]
            check("program_binding_" + label, output["program_hash"] == digest(plan))
    arms, by_block, paired = {}, {}, {}
    names = ["fixed", "outer_generated", "manual_control", "removed"]
    for arm in names:
        rows = [r for r in episodes if r["arm"] == arm]
        arms[arm] = dict(attempted=len(rows), completed=sum(r["completed"] for r in rows),
            passed=sum(r["passed"] for r in rows), incorrect_completed=sum(r["completed"] and not r["passed"] for r in rows),
            certificate_valid=sum(r["certificate_valid"] for r in rows), calls=sum(r["calls"] for r in rows),
            tokens=sum(r["tokens"] or 0 for r in models["entries"] if r["stage"] == "research" and f"-{arm}-" in r["label"]),
            unknown_tokens=sum(r["tokens"] is None for r in rows), successful_scored_reports=sum(r.get("certificate_valid",False) for r in rows),
            requested_registered_uses=sum(r["registered_uses"] for r in rows), successful_registered_uses=sum(r["registered_uses"] for r in rows if r["operations"] > 0), observed_primitive_operations=sum(r["operations"] for r in rows),
            tool_errors=sum("tool_error" in r for r in rows),
            reused_sources=sorted({r["source"] for r in rows if r["registered_uses"] and r["operations"] > 0}),
            reused_families=sorted({r["case"][0] for r in rows if r["registered_uses"] and r["operations"] > 0}))
        by_block[arm] = {block: dict(passed=sum(r["passed"] for r in rows if r["block"] == block),
            attempted=sum(r["block"] == block for r in rows), planned=count)
            for block, count in [("development", 4), ("material_transfer", 4), ("task_transfer", 8), ("joint_transfer", 8)]}
    for block in ["development", "material_transfer", "task_transfer", "joint_transfer", "all"]:
        paired[block] = {}
        for arm in names[1:]:
            count = dict(wins=0, losses=0, ties=0, missing=0)
            for p in pairs:
                if block != "all" and p["block"] != block:
                    continue
                f = next((r for r in episodes if r["id"] == p["id"] and r["arm"] == "fixed"), None)
                other = next((r for r in episodes if r["id"] == p["id"] and r["arm"] == arm), None)
                key = "missing" if f is None or other is None else "wins" if other["passed"] > f["passed"] else "losses" if other["passed"] < f["passed"] else "ties"
                count[key] += 1
            paired[block][arm] = count
    author_tokens = sum(r["tokens"] or 0 for r in models["entries"] if r["stage"] == "author")
    stages = {}
    for stage in sorted({r["stage"] for r in science["entries"]}):
        rows = [r for r in science["entries"] if r["stage"] == stage]
        stages[stage] = dict(calls=len(rows), charged_seconds=sum(r.get("wall_seconds", 0) for r in rows),
                             errors=sum(r["status"] == "error" for r in rows))
    source_results = {arm: {s: dict(passed=sum(r["passed"] for r in episodes if r["arm"] == arm and r["source"] == s),
                                   attempted=sum(r["arm"] == arm and r["source"] == s for r in episodes))
                            for s in ("D1", "D2", "E1", "E2")} for arm in names}
    result = dict(experiment_version="q2b-v2.1", visibility="public_engineering_validation", manifest=manifest, completion=completion,
        arms=arms, blocks=by_block, paired_vs_fixed=paired, sources=source_results,
        probe_tokens=sum(r["tokens"] or 0 for r in models["entries"] if r["stage"] == "probe"), author_tokens=author_tokens, generated_amortized_tokens=arms["outer_generated"]["tokens"] + author_tokens,
        known_token_lower_bound=sum(r["tokens"] or 0 for r in models["entries"]), unknown_usage=sum(r["tokens"] is None for r in models["entries"]),
        model_wall_seconds=sum(r["wall_seconds"] or 0 for r in models["entries"]), science_stages=stages,
        science_charged_seconds=sum(r.get("wall_seconds", 0) for r in science["entries"]),
        estimated_science_seconds=sum(r.get("wall_seconds", 0) for r in science["entries"] if r.get("measured") is False),
        peak_memory_bytes=max(r.get("peak_process_memory_bytes", 0) for r in science["entries"]),
        stderr_diagnostics=stderr_diagnostics, authors=authors, candidate=read("candidate.json"), episodes=episodes, calls=call_rows,
        error_classes=dict(Counter(r.get("tool_error") or r.get("error") or "no_error" for r in episodes)),
        audit=dict(checks=len(checks), passed=all(c["passed"] for c in checks)),
        engineering_cost="Development-session tokens and engineering labor unmetered; not claimed zero.")
    target = ROOT / "benchmarks/q2b_v21/astra-summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("completion", "arms", "author_tokens", "known_token_lower_bound", "science_charged_seconds", "audit")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit()

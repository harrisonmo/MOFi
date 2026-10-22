"""Reconstruct M9 refusal decisions using only information visible at that time."""

import argparse
from collections import Counter
from pathlib import Path

from rsi.experiment import ROOT, read_json, write_json, public_observation
from rsi.m9 import digest
from rsi.m9_workflow import facts


def repair_opportunity(item, observation, remaining, actions_left, steps_left):
    info = facts(item, observation)
    cost = len(info["repairs"]) + 3
    if item["incomplete_record"] or info["blockers"]:
        category = "observed_blocker"
    elif not info["repairs"]:
        category = "no_observed_allowed_repair"
    elif remaining < cost or actions_left < 2 or steps_left < 1:
        category = "insufficient_completion_budget"
    else:
        category = "affordable_observed_repair"
    return {"category": category, "repairs": sorted(info["repairs"]),
            "observed_blockers": sorted(info["blockers"]), "completion_cost_lower_bound": cost,
            "remaining_units": remaining, "remaining_actions": actions_left,
            "remaining_candidate_optimization_steps": steps_left}


def analyze():
    report = read_json(ROOT / "docs/reports/M9_RECOVERY_RESULT.json")
    episodes, refusals, hashes = [], [], {}
    for row in report["episodes"]:
        run = ROOT / ("runs/m9-recovery-01" if row["pool"] == "m7-dev-06" else "runs/m9-workflow-01")
        folder = run / row["pool"] / row["arm"]
        saved = read_json(folder / "pool-state.json")
        pools = read_json(ROOT / "runs/m9-workflow-01/pools.json")
        pool = next(p for p in pools if p["id"] == row["pool"])
        catalog = [{"candidate_id": c["id"], **c["catalog"]} for c in pool["candidates"]]
        planning = []
        for path in sorted(folder.glob("decision-*/prompt.txt")):
            import json
            context = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            result = read_json(path.parent / "result.json")
            planning.append({"actions_used": context["actions_used"], "response": result["response"],
                             "planning_index": len(planning) + 1})
            for p in [path, path.parent / "result.json"]:
                hashes[p.relative_to(ROOT).as_posix()] = digest(p)
        observations, actions = {}, []
        for index, event in enumerate(saved["events"], 1):
            path = folder / f"action-{index:02d}/result/result.json"
            result = read_json(path)
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
            cid, action = event["candidate_id"], event["action"]
            before = observations.get(cid)
            observations[cid] = public_observation(result)
            ok = result["events"][-1]["result"].get("ok")
            actions.append({"candidate_id": cid, "action": action, "cost": event["cost"],
                            "cumulative_cost": event["cumulative_cost"], "ok": ok,
                            "current_codes": sorted(d["code"] for d in result["final"]["diagnostics"])})
            if action["verb"] != "optimize" or ok is not False:
                continue
            item = next(c for c in catalog if c["candidate_id"] == cid)
            used = sum(e["action"].get("max_steps", 0) for e in result["events"])
            immediate = [p for p in planning if p["actions_used"] == index]
            next_event = saved["events"][index] if index < len(saved["events"]) else None
            refusals.append({"pool": row["pool"], "arm": row["arm"], "action_index": index,
                "candidate_id": cid, "observed_before": before, "observed_after": observations[cid],
                "opportunity": repair_opportunity(item, observations[cid], 12 - event["cumulative_cost"], 12 - index, 3 - used),
                "replanned_immediately": bool(immediate), "subsequent_plan": immediate[0] if immediate else None,
                "next_executed_action": {k: next_event[k] for k in ["candidate_id", "action", "cost"]} if next_event else None})
        episodes.append({"pool": row["pool"], "arm": row["arm"], "catalog": catalog,
                         "actions": actions, "score": row["score"], "cost": row["cost"]})
    summary = {}
    for arm in ["batch", "m8", "candidate"]:
        windows = [r for r in refusals if r["arm"] == arm]
        summary[arm] = {"refusals": len(windows),
            "categories": dict(Counter(r["opportunity"]["category"] for r in windows)),
            "immediate_replanning": sum(r["replanned_immediately"] for r in windows),
            "next_action_is_repair": sum(bool(r["next_executed_action"] and r["next_executed_action"]["action"]["verb"] == "prepare") for r in windows)}
    return {"scope": "M9 executed public development evidence; opportunity categories use only contemporaneous observations. Affordable is not predicted success.",
            "summary": summary, "refusal_windows": refusals, "executed_episodes": episodes, "source_hashes": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze()
    write_json(args.output, data)
    print(data["summary"])

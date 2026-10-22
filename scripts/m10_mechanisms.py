"""Summarize actual M10 action costs and repair outcomes after replay audit."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.experiment import read_json, write_json


def summarize_mechanisms(report, run):
    assert report["audit"]["all_checks_passed"] and report["summary"]["complete"]
    result = {}
    for arm in ["batch", "m8", "candidate"]:
        rows = [r for r in report["episodes"] if r["arm"] == arm]
        repairs, refusals = [], []
        for row in rows:
            for index, event in enumerate(row["actions"], 1):
                cid = event["candidate_id"]
                saved = read_json(run / row["pool"] / arm / f"action-{index:02d}/result/result.json")
                if event["action"]["verb"] == "optimize" and saved["events"][-1]["result"].get("ok") is False:
                    refusals.append({"pool": row["pool"], "candidate_id": cid, "action_index": index, "cost": event["cost"]})
                if event["action"]["verb"] != "prepare":
                    continue
                later_calculation = any(e["candidate_id"] == cid and e["action"]["verb"] == "optimize" for e in row["actions"][index:])
                interrupted = any(e["reason"] == "after_prepare" and e.get("after_action") == index for e in row["control"])
                outcome = ("verified_after_repair" if cid in row["score"]["verified_ids"] else
                           "unqualified_after_calculation" if later_calculation else
                           "interrupted_for_new_blocker" if interrupted else "not_calculated")
                repairs.append({"pool": row["pool"], "candidate_id": cid, "action_index": index,
                                "outcome": outcome, "cost": event["cost"], "later_calculation": later_calculation,
                                "final_reasons": row["final_qualifications"][cid]["reasons"]})
        result[arm] = {"action_counts": dict(Counter(e["action"]["verb"] for r in rows for e in r["actions"])),
            "control_counts": dict(Counter(e["reason"] for r in rows for e in r["control"])),
            "repair_outcomes": dict(Counter(e["outcome"] for e in repairs)), "repairs": repairs,
            "refusals": refusals, "refusal_cost": sum(e["cost"] for e in refusals),
            "unused_science_units": 72 - sum(r["cost"]["units"] for r in rows),
            "planning_calls": sum(len(r["decisions"]) for r in rows),
            "feedback_plans": sum(max(0, len(r["decisions"]) - 1) for r in rows)}
    return {"scope": "Descriptive outcomes from replay-verified M10 traces; no counterfactual success claims.", "arms": result}


if __name__ == "__main__":
    report = read_json(ROOT / "docs/reports/M10_RESULT.json")
    result = summarize_mechanisms(report, ROOT / "runs/m10-workflow-01")
    write_json(ROOT / "docs/reports/M10_MECHANISMS.json", result)
    for arm, row in result["arms"].items():
        print(arm, {k: v for k, v in row.items() if k not in {"repairs", "refusals"}})

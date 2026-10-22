"""Replay saved model decisions without inference and audit M9 evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.backends.codex_cli import DECISION_INSTRUCTIONS
from rsi.experiment import read_json, write_json, public_observation
from rsi.m9 import ARMS, digest, selection, summarize
from rsi.m9_workflow import WorkflowScreeningExperiment, policy_hash, interruption, veto


class ReplayBackend:
    def __init__(self, source, destination):
        self.source, self.destination = source, destination

    def run(self, prompt, schema, directory):
        original = self.source / directory.relative_to(self.destination)
        assert DECISION_INSTRUCTIONS + prompt == (original / "prompt.txt").read_text(encoding="utf-8"), "Prompt replay differs"
        assert schema == read_json(original / "schema.json")
        result = read_json(original / "result.json")
        assert result["status"] == "ok" and result["unexpected_tool_types"] == []
        assert result["requested_model"] == "gpt-6-astra"
        for key in ['"qualified"', '"canonical"', '"framework_id"', '"reference_actions"']:
            assert key not in prompt, "Private evaluator field exposed"
        return result


def audit(run, destination):
    manifest, rows = read_json(run / "manifest.json"), read_json(run / "rows.json")
    summary = summarize(rows)
    assert not summary["complete"]
    all_rows = rows
    failed_rows = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    assert len(rows) == 15 and len(failed_rows) == 1
    assert {r["pool"] for r in rows} == {f"m7-dev-{i:02d}" for i in range(1, 6)}
    assert all(sum(r["pool"] == p and r["arm"] == a for r in rows) == 1
               for p in {r["pool"] for r in rows} for a in ARMS)
    assert all(digest(ROOT / p) == h for p, h in manifest["source_and_input_hashes"].items())
    assert digest(run / "feedback.json") == manifest["feedback_sha256"]
    assert digest(ROOT / "runs/m8-workflow-01/rows.json") == manifest["historical_feedback_sha256"]
    candidate = read_json(run / "candidate.json")
    assert policy_hash(candidate["workflow"]) == candidate["sha256"]
    pools = {p["id"]: p for p in read_json(run / "pools.json")}
    assert [{"pool": r["pool"], "arm": r["arm"]} for r in rows] == manifest["schedule"][:15]
    counts = {"episodes_replayed": 0, "action_artifacts": 0, "candidate_artifacts": 0, "decision_prompts_checked": 0}
    episodes = []
    for row in rows:
        source, target = run / row["pool"] / row["arm"], destination / row["pool"] / row["arm"]
        exp = WorkflowScreeningExperiment(manifest["config"], target, ReplayBackend(source, target))
        def science(task, actions, directory, previous=None):
            original = source / directory.relative_to(target)
            assert read_json(original / "task.json") == task
            assert read_json(original / "actions.json") == actions
            if previous is not None:
                assert read_json(original / "previous.json") == previous, "Checkpoint replay differs"
            return read_json(original / "result.json")
        exp.science = science
        expected_policy = candidate["workflow"] if row["arm"] == "candidate" else manifest["baseline_policies"][row["arm"]]
        assert row["workflow"] == expected_policy
        replay = exp.episode(pools[row["pool"]], row["arm"], expected_policy)
        assert all(row[k] == v for k, v in replay.items()), "Decision/control/report replay differs"
        saved = read_json(source / "pool-state.json")
        assert saved == read_json(target / "pool-state.json")
        assert exp.calls == row["cost"]["calls"] <= 13
        assert exp.observed_tokens == row["cost"]["tokens"] <= 325000
        assert saved["spent"] == row["cost"]["units"] <= 12
        assert sum(e["action"].get("max_steps", 0) for e in saved["events"]) == row["cost"]["requested_steps"] <= 4
        for cid, state in saved["states"].items():
            endpoint = state["checkpoint_history"][-1]
            assert all(state["artifact"][k] == endpoint[k] for k in ["path", "sha256"])
            assert all(digest(Path(h["path"])) == h["sha256"] for h in state["checkpoint_history"])
        candidates = {c["id"]: c for c in pools[row["pool"]]["candidates"]}
        for claim in row["report"]["candidates"]:
            if claim["status"] == "rejected":
                cid = claim["candidate_id"]
                state = saved["states"][cid]
                current = {d["code"] for d in state["final"]["diagnostics"]}
                assert (candidates[cid]["catalog"]["incomplete_record"]
                        or current & {"COINCIDENT_ATOMS", "DEUTERIUM", "OVERCOORDINATED", "MISSING_HYDROGENS", "CALCULATION_REFUSED"}
                        or "BONDING_WOULD_CHANGE" in state["diagnostic_codes"]), "Stale rejection evidence"
        counts["episodes_replayed"] += 1
        counts["action_artifacts"] += len(saved["events"])
        counts["candidate_artifacts"] += len(saved["states"])
        counts["decision_prompts_checked"] += exp.calls
        episodes.append({**row, "actions": saved["events"], "final_qualifications": saved["grades"]})
    outer = read_json(run / "evolution/proposal/result.json")
    assert outer["status"] == "ok" and not outer["unexpected_tool_types"]
    assert outer["response"]["workflow"] == candidate["workflow"]
    outer_tokens = outer["usage"]["input_tokens"] + outer["usage"]["output_tokens"]
    assert outer_tokens <= 100000
    stage = read_json(run / "stage-ledger.json")
    total_tokens = outer_tokens + sum(r["cost"]["tokens"] for r in rows)
    assert all(e["status"] == "known" for e in stage[:-1]) and stage[-1]["status"] == "unknown"
    failed = failed_rows[0]
    assert failed["pool"] == "m7-dev-06" and failed["arm"] == "candidate"
    failure_path = run / failed["pool"] / failed["arm"] / "decision-00"
    failure = read_json(failure_path / "result.json")
    assert failure["status"] == "process_error" and failure["usage"] is None
    assert "usage limit" in (failure_path / "events.jsonl").read_text(encoding="utf-8")
    assert failed["cost"]["unknown_usage"] == 1 and failed["cost"]["units"] == 0
    assert not (run / "m7-dev-06/batch").exists() and not (run / "m7-dev-06/m8").exists()
    assert len(stage) == 2 + sum(r["cost"]["calls"] for r in rows) <= 235
    assert sum(e["tokens"] for e in stage) == total_tokens <= 5950000
    decision = selection(all_rows)
    assert read_json(run / "selection.json") == decision
    accepted = read_json(run / "accepted.json")
    assert accepted["workflow"] == (candidate["workflow"] if decision["promoted"] else manifest["baseline_policies"]["m8"])
    assert policy_hash(accepted["workflow"]) == accepted["sha256"]
    return {"manifest": manifest, "candidate": candidate, "summary": summarize(rows), "raw_attempt_summary": summary, "failed_rows": failed_rows, "selection": decision,
            "mechanisms": {a: {"planning_calls": sum(len(r["decisions"]) for r in rows if r["arm"] == a),
                "feedback_plans": sum(max(0, len(r["decisions"])-1) for r in rows if r["arm"] == a),
                "control_events": [e for r in rows if r["arm"] == a for e in r["control"]],
                "action_verbs": [e["action"]["verb"] for r in episodes if r["arm"] == a for e in r["actions"]]} for a in ARMS},
            "accepted": accepted, "audit": {**counts, "all_checks_passed": True,
            "scope": "Only 15 completed episodes replayed. Stage incomplete; failed call usage unknown. No inference or science calls."},
            "total_cost": {"calls": len(stage), "known_tokens": total_tokens, "unknown_usage_calls": 1, "outer_tokens": outer_tokens,
                           "requested_steps": sum(r["cost"]["requested_steps"] for r in rows)}, "episodes": episodes}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.7, 1]})
    colors, labels = ["#1768ac", "#de8f05", "#008b73"], ["Batch", "M8 parent", "Evolved workflow"]
    x = np.arange(5)
    for i, arm in enumerate(ARMS):
        ys = [next(r["score"]["Y"] for r in report["episodes"] if r["pool"] == f"m7-dev-{p:02d}" and r["arm"] == arm) for p in range(1, 6)]
        axes[0].bar(x + (i-1)*.24, ys, width=.23, color=colors[i], label=labels[i])
    axes[0].axhline(3, color="#555555", linestyle="--", linewidth=1, label="Known feasible (not optimum)")
    axes[0].set(xticks=x, xticklabels=[f"{i:02d}" for i in range(1, 6)], xlabel="Frozen M7 pool", ylabel="Evidence-qualified selections", ylim=(0, 4.1), yticks=range(5))
    axes[0].legend(loc="upper center", fontsize=8, ncol=2)
    totals = [report["summary"]["metrics"][a]["tokens"] / 1000 for a in ARMS]
    axes[1].barh(labels, totals, color=colors)
    axes[1].set(xlabel="Research Loop tokens (thousands)", xlim=(0, max(totals)*1.25))
    axes[1].invert_yaxis()
    for y, v in enumerate(totals):
        axes[1].text(v+3, y, f"{v:.1f}", va="center", fontsize=9)
    fig.suptitle("M9: one frozen autonomous proposal; five completed public development pools; sixth interrupted")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/m9-workflow-01")
    parser.add_argument("--replay-output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run.resolve(), args.replay_output.resolve())
    write_json(ROOT / "docs/reports/M9_RESULT.json", report)
    plot(report, ROOT / "docs/reports/M9_COMPARISON.png")
    print(json.dumps({k: report[k] for k in ["audit", "summary", "selection", "total_cost"]}, indent=2))

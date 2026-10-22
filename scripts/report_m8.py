"""Replay saved model decisions without inference and audit M8 evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.backends.codex_cli import DECISION_INSTRUCTIONS
from rsi.experiment import read_json, write_json, public_observation
from rsi.m8 import ARMS, digest, selection, summarize
from rsi.m8_workflow import WorkflowScreeningExperiment, policy_hash, interruption, veto


def mechanism_probes(policy):
    """Check control decisions on existing real observations, without new science."""
    prior = ROOT / "runs/m7-pilot-01/m7-dev-01/stepwise"
    before_path = prior / "action-02/result/result.json"
    after_path = prior / "action-03/result/result.json"
    before, after = public_observation(read_json(before_path)), public_observation(read_json(after_path))
    pool = read_json(ROOT / "runs/m7-pilot-01/pools.json")[0]
    item = next({"candidate_id": c["id"], **c["catalog"]} for c in pool["candidates"] if c["id"] == "c07")
    event = interruption(item, before, after, {"verb": "prepare", "steps": ["duplicates"]}, policy)
    guard = veto({"catalog": [item], "observations": {"c07": before}, "remaining_units": 3},
                 "c07", {"verb": "inspect"}, policy)
    assert event == "after_prepare" and guard == "calculation_reserve"
    return {"repair_event": event, "reserve_event": guard,
            "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in [before_path, after_path]},
            "scope": "Offline probes on saved real M7 observations; not an M8 episode or evidence of performance gain"}


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
    assert summary["complete"] and not any(r.get("error") for r in rows)
    assert all(digest(ROOT / p) == h for p, h in manifest["source_and_input_hashes"].items())
    assert digest(run / "feedback.json") == manifest["feedback_sha256"]
    assert digest(ROOT / "runs/m7-pilot-01/rows.json") == manifest["historical_feedback_sha256"]
    candidate = read_json(run / "candidate.json")
    assert policy_hash(candidate["workflow"]) == candidate["sha256"]
    pools = {p["id"]: p for p in read_json(run / "pools.json")}
    assert [{"pool": r["pool"], "arm": r["arm"]} for r in rows] == manifest["schedule"]
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
    assert all(e["status"] == "known" for e in stage)
    assert len(stage) == 1 + sum(r["cost"]["calls"] for r in rows) <= 235
    assert sum(e["tokens"] for e in stage) == total_tokens <= 5950000
    decision = selection(rows)
    assert read_json(run / "selection.json") == decision
    accepted = read_json(run / "accepted.json")
    assert accepted["workflow"] == (candidate["workflow"] if decision["promoted"] else manifest["baseline_policies"]["m5"])
    assert policy_hash(accepted["workflow"]) == accepted["sha256"]
    return {"manifest": manifest, "candidate": candidate, "summary": summary, "selection": decision,
            "offline_mechanism_probes": mechanism_probes(candidate["workflow"]),
            "accepted": accepted, "audit": {**counts, "all_checks_passed": True,
            "scope": "Replay uses saved responses and artifacts only; no inference or scientific worker calls"},
            "total_cost": {"calls": len(stage), "tokens": total_tokens, "outer_tokens": outer_tokens,
                           "requested_steps": sum(r["cost"]["requested_steps"] for r in rows)}, "episodes": episodes}


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.7, 1]})
    colors, labels = ["#1768ac", "#de8f05", "#008b73"], ["Batch", "M5", "Evolved workflow"]
    x = np.arange(6)
    for i, arm in enumerate(ARMS):
        ys = [next(r["score"]["Y"] for r in report["episodes"] if r["pool"] == f"m7-dev-{p:02d}" and r["arm"] == arm) for p in range(1, 7)]
        axes[0].bar(x + (i-1)*.24, ys, width=.23, color=colors[i], label=labels[i])
    axes[0].axhline(3, color="#555555", linestyle="--", linewidth=1, label="Known feasible (not optimum)")
    axes[0].set(xticks=x, xticklabels=[f"{i:02d}" for i in range(1, 7)], xlabel="Frozen M7 pool", ylabel="Evidence-qualified selections", ylim=(0, 4.1), yticks=range(5))
    axes[0].legend(loc="upper center", fontsize=8, ncol=2)
    totals = [report["summary"]["metrics"][a]["tokens"] / 1000 for a in ARMS]
    axes[1].barh(labels, totals, color=colors)
    axes[1].set(xlabel="Research Loop tokens (thousands)", xlim=(0, max(totals)*1.25))
    axes[1].invert_yaxis()
    for y, v in enumerate(totals):
        axes[1].text(v+3, y, f"{v:.1f}", va="center", fontsize=9)
    fig.suptitle("M8: one frozen autonomous proposal; six public development pools")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/m8-workflow-01")
    parser.add_argument("--replay-output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run.resolve(), args.replay_output.resolve())
    write_json(ROOT / "docs/reports/M8_RESULT.json", report)
    plot(report, ROOT / "docs/reports/M8_COMPARISON.png")
    print(json.dumps({k: report[k] for k in ["audit", "summary", "selection", "total_cost"]}, indent=2))

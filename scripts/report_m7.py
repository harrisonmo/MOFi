"""Audit saved M7 evidence and plot yield without running science or models."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.m7 import ARMS, summarize
from rsi.m7_core import BLOCKERS, REVIEW_CODES, action_cost, grade_report, qualify


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    manifest, rows = read(run / "manifest.json"), read(run / "rows.json")
    summary = summarize(rows)
    assert summary["complete"] and summary["comparable"], "The full paired experiment is required"
    pools = {p["id"]: p for p in read(run / "pools.json")}
    assert manifest["pools_sha256"] == hashlib.sha256(json.dumps(list(pools.values()), sort_keys=True).encode()).hexdigest()
    assert all(digest(ROOT / p) == h for p, h in manifest["source_hashes"].items()), "Source changed"
    assert all(digest(ROOT / "benchmarks/m7/fixtures" / p) == h for p, h in manifest["fixture_hashes"].items())
    assert [{"pool": r["pool"], "arm": r["arm"]} for r in rows] == manifest["schedule"]
    episodes, counts = [], {"candidate_artifacts": 0, "action_artifacts": 0, "model_calls": 0, "tokens": 0, "regraded_episodes": 0}
    for row in rows:
        folder = run / row["pool"] / row["arm"]
        saved = read(folder / "pool-state.json")
        candidates = {c["id"]: c for c in pools[row["pool"]]["candidates"]}
        state = SimpleNamespace(candidates=candidates, **saved)
        score = grade_report(state, row["report"])
        assert all(row["score"][k] == v for k, v in score.items()), "Regrade mismatch"
        counts["regraded_episodes"] += 1
        assert sum(action_cost(e["action"]) for e in state.events) == state.spent <= 12
        assert len(state.events) <= 12
        cumulative = 0
        for index, event in enumerate(state.events, 1):
            cumulative += action_cost(event["action"])
            assert event["cumulative_cost"] == cumulative and event["status"] == "completed"
            result = read(folder / f"action-{index:02d}" / "result/result.json")
            assert qualify(candidates[event["candidate_id"]], result)["qualified"] == event["qualified"]
            counts["action_artifacts"] += 1
        for cid, result in state.states.items():
            counts["candidate_artifacts"] += 1
            assert qualify(candidates[cid], result) == state.grades[cid]
            endpoint = result["checkpoint_history"][-1]
            assert all(result["artifact"][k] == endpoint[k] for k in ("path", "sha256"))
            assert sum(e["action"].get("max_steps", 0) for e in result["events"]) <= 3
        # Independently exclude a cleared historical blocker as a rejection basis.
        for claim in row["report"]["candidates"]:
            if claim["status"] != "rejected":
                continue
            cid = claim["candidate_id"]
            result = state.states[cid]
            current = {d["code"] for d in result["final"]["diagnostics"]}
            assert (candidates[cid]["catalog"]["incomplete_record"] or current & BLOCKERS
                    or set(result["diagnostic_codes"]) & REVIEW_CODES
                    or "CALCULATION_REFUSED" in current), "Unsupported current rejection"
        ledger = read(folder / "ledger.json")
        tokens = sum(e["usage"]["input_tokens"] + e["usage"]["output_tokens"] for e in ledger)
        assert len(ledger) == row["cost"]["calls"] <= 13
        assert tokens == row["cost"]["tokens"] <= 325000
        assert row["cost"]["units"] == state.spent
        assert row["cost"]["requested_steps"] == sum(e["action"].get("max_steps", 0) for e in state.events) <= 4
        for entry in ledger:
            result = read(folder / entry["path"] / "result.json")
            assert result["status"] == "ok" and not result["unexpected_tool_types"]
            assert result["requested_model"] == "gpt-6-astra"
            prompt = (folder / entry["path"] / "prompt.txt").read_text(encoding="utf-8")
            assert not any(key in prompt for key in ('"qualified"', '"canonical"', '"reference_actions"', '"framework_id"'))
        counts["model_calls"] += len(ledger)
        counts["tokens"] += tokens
        episodes.append({**row, "actions": state.events,
                         "final_qualifications": state.grades,
                         "frameworks_selected": [candidates[c]["framework_id"] for c in score["verified_ids"]]})
    stage = read(run / "stage-ledger.json")
    assert all(e["status"] == "known" for e in stage)
    assert len(stage) == counts["model_calls"] <= 234
    assert sum(e["tokens"] for e in stage) == counts["tokens"] <= 5850000
    return {"run": run.relative_to(ROOT).as_posix(), "manifest": manifest, "summary": summary,
            "audit": {**counts, "all_checks_passed": True,
                      "scope": "Independent rereading and regrading of saved artifacts; no model or UFF calls"},
            "episodes": episodes}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "m7-yield-v1"
    import matplotlib.pyplot as plt
    colors = {"batch": "#1768ac", "m5": "#de8f05", "stepwise": "#008b73"}
    labels = {"batch": "Batch", "m5": "M5 workflow", "stepwise": "Stepwise"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat, 1):
        for arm in ARMS:
            row = next(r for r in report["episodes"] if r["pool"] == f"m7-dev-{index:02d}" and r["arm"] == arm)
            first = {}
            for event in row["actions"]:
                if event.get("qualified") and event["candidate_id"] in row["score"]["verified_ids"]:
                    first.setdefault(event["candidate_id"], event["cumulative_cost"])
            xs = list(range(13))
            ys = [sum(cost <= x for cost in first.values()) for x in xs]
            ax.step(xs, ys, where="post", color=colors[arm], label=labels[arm],
                    linestyle={"batch": "-", "m5": "--", "stepwise": ":"}[arm], linewidth=2,
                    marker={"batch": "o", "m5": "s", "stepwise": "^"}[arm],
                    markevery=[-1], markersize=5, markerfacecolor="white")
        ax.axhline(3, color="#777777", linewidth=1, linestyle="-.", label="Known feasible (not optimum)")
        ax.set(title=f"Pool {index:02d}", xlim=(0, 12.35), ylim=(-0.1, 4.1), xticks=[0, 3, 6, 9, 12], yticks=range(5))
        ax.grid(alpha=0.18)
    fig.supxlabel("Scientific cost units consumed (common ceiling: 12)")
    fig.supylabel("Final evidence-qualified selections")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.92))
    for suffix in ("png", "svg"):
        destination = output.with_suffix("." + suffix)
        fig.savefig(destination, dpi=180, **({"metadata": {"Date": None}} if suffix == "svg" else {}))
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/m7-pilot-01")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/M7_RESULT.json")
    args = parser.parse_args()
    report = audit(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(report, args.output.with_name("M7_YIELD"))
    print(json.dumps({"audit": report["audit"], "summary": report["summary"]}, indent=2))

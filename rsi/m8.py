"""Propose and compare one workflow on the immutable public M7 benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import ROOT, cli_identity, git_state, read_json, write_json
from rsi.m6 import ThreeArmExperiment
from rsi.m7_core import grade_report
from rsi.m8_workflow import WorkflowScreeningExperiment, PROPOSAL_SCHEMA, POLICY_SCHEMA, SEMANTICS, baseline_policy, policy_hash

ARMS = ("batch", "m5", "candidate")
CONFIG = {"model": "gpt-6-astra", "reasoning": "high", "cli_timeout_seconds": 180,
          "science_timeout_seconds": 120, "science_seed": 0, "max_llm_calls": 13, "max_observed_tokens": 325000}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Budget:
    def __init__(self, path):
        self.path, self.entries = path, []
        self.save()

    def reserve(self, label):
        if (len(self.entries) >= 235 or sum(e.get("tokens", 0) for e in self.entries) >= 5950000
                or any(e["status"] != "known" for e in self.entries)):
            raise RuntimeError("M8 budget exhausted or unknown usage")
        entry = {"label": label, "status": "reserved"}
        self.entries.append(entry)
        self.save()
        return entry

    def save(self):
        write_json(self.path, self.entries)


def frozen_hashes():
    prior = read_json(ROOT / "docs/reports/M7_RESULT.json")["manifest"]
    for path, sha in prior["source_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError("Frozen M7 source changed: " + path)
    for path, sha in prior["fixture_hashes"].items():
        if digest(ROOT / "benchmarks/m7/fixtures" / path) != sha:
            raise ValueError("Frozen M7 fixture changed")
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    if hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest() != prior["pools_sha256"]:
        raise ValueError("Frozen M7 pools changed")
    files = [p for base in [ROOT / "rsi", ROOT / "Crystal-Builder/xtal", ROOT / "benchmarks/m7"] for p in base.rglob("*")
             if p.is_file() and (p.suffix in {".py", ".json", ".xtalproj"})]
    files.append(ROOT / "workflows/m5-accepted.json")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def feedback():
    """Expose executed public development evidence, without reference identities."""
    source = ROOT / "runs/m7-pilot-01"
    data = []
    for row in read_json(source / "rows.json"):
        directory = source / row["pool"] / row["arm"]
        saved = read_json(directory / "pool-state.json")
        actions = []
        for index, event in enumerate(saved["events"], 1):
            result = read_json(directory / f"action-{index:02d}/result/result.json")
            actions.append({"candidate_id": event["candidate_id"], "action": event["action"],
                            "cost": event["cost"], "cumulative_cost": event["cumulative_cost"],
                            "ok": result["events"][-1]["result"].get("ok"),
                            "current_codes": sorted({d["code"] for d in result["final"]["diagnostics"]}),
                            "observed_codes": result["diagnostic_codes"]})
        data.append({"pool": row["pool"], "arm": row["arm"], "actions": actions,
                     "score": row["score"], "cost": row["cost"]})
    catalogs = [{"pool": p["id"], "catalog": [{"candidate_id": c["id"], **c["catalog"]} for c in p["candidates"]]}
                for p in read_json(source / "pools.json")]
    return {"catalogs": catalogs, "executed_episodes": data}


def summarize(rows):
    metrics = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        metrics[arm] = {"pools": len(selected), **{k: sum(r.get("score", {}).get(k, 0) for r in selected)
                          for k in ["Y", "F", "auc", "reliable", "false_rejections", "unsupported_rejections"]},
                        **{k: sum(r["cost"][k] for r in selected) for k in ["calls", "tokens", "units", "requested_steps"]}}
    keys = [(r["pool"], r["arm"]) for r in rows]
    return {"metrics": metrics, "complete": len(keys) == len(set(keys)) == 18}


def selection(rows):
    summary = summarize(rows)
    if not summary["complete"] or any(r.get("error") or r["cost"]["unknown_usage"] for r in rows):
        return {"promoted": False, "reason": "incomplete_or_unknown_usage"}
    by_key = {(r["pool"], r["arm"]): r for r in rows}
    expected = {f"m7-dev-{i:02d}" for i in range(1, 7)}
    if set(by_key) != {(p, a) for p in expected for a in ARMS}:
        return {"promoted": False, "reason": "wrong_comparison_pools"}
    if not all(r["score"]["reliable"] for r in rows):
        return {"promoted": False, "reason": "unreliable_comparison"}
    regressions = [{"pool": p, "against": a} for p in sorted(expected) for a in ["batch", "m5"]
                   if by_key[p, "candidate"]["score"]["Y"] < by_key[p, a]["score"]["Y"]]
    m = summary["metrics"]
    gain = m["candidate"]["Y"] > max(m[a]["Y"] for a in ["batch", "m5"])
    affordable = m["candidate"]["tokens"] <= 1.5 * m["m5"]["tokens"]
    equal = m["candidate"]["Y"] == m["batch"]["Y"] == m["m5"]["Y"]
    efficient = (m["candidate"]["tokens"] <= .9 * min(m[a]["tokens"] for a in ["batch", "m5"])
                 and m["candidate"]["calls"] <= min(m[a]["calls"] for a in ["batch", "m5"]))
    promoted = not regressions and ((gain and affordable) or (equal and efficient))
    return {"promoted": promoted, "reason": "qualified_development_gain" if promoted else "no_qualified_gain",
            "regressions": regressions, "yield_gain_over_both": gain, "within_token_tradeoff": affordable,
            "scope": "Public development only; not held-out generalization"}


def run(output):
    hashes = frozen_hashes()
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    if len(pools) != 6 or any(p["split"] != "development" for p in pools):
        raise ValueError("Only the six frozen development pools may run")
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output / "stage-ledger.json")
    proposer = ThreeArmExperiment(dict(CONFIG, max_llm_calls=1, max_observed_tokens=100000), output / "evolution", stage=budget)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "root": git_state(ROOT),
                "crystal_builder": git_state(ROOT / "Crystal-Builder"), "cli": cli_identity(proposer.backend.executable),
                "source_and_input_hashes": hashes, "config": CONFIG,
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder"]},
                "baseline_policies": {a: baseline_policy(a) for a in ["batch", "m5"]},
                "historical_feedback_sha256": digest(ROOT / "runs/m7-pilot-01/rows.json"),
                "scope": "One autonomous bounded-policy proposal, frozen M7 public development comparison"}
    data = feedback()
    write_json(output / "feedback.json", data)
    manifest["feedback_sha256"] = digest(output / "feedback.json")
    schedule = [(p, a) for i, p in enumerate(pools) for a in ARMS[i % 3:] + ARMS[:i % 3]]
    manifest["schedule"] = [{"pool": p["id"], "arm": a} for p, a in schedule]
    write_json(output / "manifest.json", manifest)
    write_json(output / "pools.json", pools)
    proposal = proposer.model("You are the Evolution Loop. Propose exactly one changed screening workflow and an English hypothesis. "
        "Only numeric, boolean, and enum workflow fields may change. Do not change scientific prompts, tools, tasks, grading, "
        "budgets, or models. No candidate IDs, material identity rules, free-form executable instructions or new tools. "
        "Your hypothesis is recorded but never sent to the Research Loop. Optimize candidate allocation, interruption after "
        "repair, and reserving enough units to complete useful calculations. More feedback is not inherently beneficial. "
        "The supplied outcomes are public development feedback, not hidden validation. The same six pools will be freshly run "
        "under batch, M5, and your frozen proposal with identical ceilings: 12 science units, 13 model calls, 325000 tokens. "
        "A development promotion requires reliable reports in all arms, no per-pool yield regression against either baseline, "
        "strictly greater total yield than both and at most 1.5 times M5 tokens; alternatively equal yield to both with at least "
        "10% fewer tokens and no more calls than either. You get one proposal, no tuning after comparison.\n" + SEMANTICS
        + "\n" + json.dumps({"parent": baseline_policy("m5"), "schema": POLICY_SCHEMA, "development": data}, ensure_ascii=False),
        PROPOSAL_SCHEMA, output / "evolution/proposal")
    sha = policy_hash(proposal["workflow"])
    if proposer.observed_tokens > 100000:
        raise RuntimeError("Outer proposal token ceiling exceeded; no comparison started")
    write_json(output / "candidate.json", {**proposal, "sha256": sha})
    if sha == policy_hash(baseline_policy("m5")):
        write_json(output / "selection.json", {"promoted": False, "reason": "unchanged_policy"})
        return
    rows = []
    for pool, arm in schedule:
        if any(digest(ROOT / p) != h for p, h in hashes.items()) or policy_hash(read_json(output / "candidate.json")["workflow"]) != sha:
            raise RuntimeError("Frozen experiment content changed")
        exp = WorkflowScreeningExperiment(CONFIG, output / pool["id"] / arm, stage=budget)
        if cli_identity(exp.backend.executable) != manifest["cli"]:
            raise RuntimeError("CLI changed")
        policy = proposal["workflow"] if arm == "candidate" else baseline_policy(arm)
        started = time.monotonic()
        try:
            row = exp.episode(pool, arm, policy)
        except Exception as exc:
            row = {"pool": pool["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
        path = exp.output / "pool-state.json"
        state = read_json(path) if path.exists() else {"spent": 0, "events": []}
        row["cost"] = {"calls": exp.calls, "tokens": exp.observed_tokens, "units": state["spent"],
                       "requested_steps": sum(e["action"].get("max_steps", 0) for e in state["events"]),
                       "unknown_usage": sum(not all(k in e.get("usage", {}) for k in ["input_tokens", "output_tokens"]) for e in exp.ledger),
                       "wall_seconds": time.monotonic() - started}
        rows.append(row)
        write_json(exp.output / "graded.json", row)
        write_json(output / "rows.json", rows)
        write_json(output / "summary.json", summarize(rows))
        print(pool["id"], arm, row.get("score", row.get("error")), row["cost"]["tokens"], flush=True)
        if row["cost"]["unknown_usage"] or any(e["status"] != "known" for e in budget.entries):
            break
    decision = selection(rows)
    write_json(output / "selection.json", decision)
    selected = proposal["workflow"] if decision["promoted"] else baseline_policy("m5")
    write_json(output / "accepted.json", {"workflow": selected, "sha256": policy_hash(selected),
               "source": "candidate" if decision["promoted"] else "m5", "scope": "M7 public development"})
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as exc:
        if args.output.exists():
            write_json(args.output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise

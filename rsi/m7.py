"""Run the frozen three-arm M7 development screening experiment."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import ROOT, PLAN_SCHEMA, cli_identity, git_state, normalize_actions, read_json, write_json
from rsi.m6 import ThreeArmExperiment, policy_for, ARMS
from rsi.m7_core import PoolState, grade_report
from rsi.science.adapter import input_path
from rsi.workflow import load_policy

INSTRUCTIONS = (
    "Find as many different framework candidates as possible with verified computational-preparation evidence. "
    "A verified candidate must have a saved artifact, preserved framework identity and cell, at least one successful "
    "finite UFF calculation, no unresolved missing-record flag, and no remaining COINCIDENT_ATOMS, DEUTERIUM, "
    "OVERCOORDINATED or MISSING_HYDROGENS diagnosis. Any BONDING_WOULD_CHANGE observed during this episode "
    "requires external bond review and prevents verification; the available tools cannot clear that review. "
    "ENGINE_NOTE typing uncertainties must be disclosed but do not alone prevent computational preparation; "
    "this does not establish force-field accuracy. Finite short UFF results do not establish convergence or stability. "
    "Allowed tools: inspect costs 1 unit; prepare allows only duplicates and deuterium, each substep costs 1; "
    "optimize(max_steps=1..3) costs 3 units per requested step, including refusal; stop ends the whole pool. "
    "Each candidate has an independent checkpoint and a maximum of three requested UFF steps. "
    "The entire pool has 12 units and at most 12 nonterminal actions. No refunds or budget borrowing. "
    "A first action includes the candidate's actual initialization diagnostics; you may optimize directly. "
    "Later actions resume the latest checkpoint. Do not reconstruct missing chemistry or change bonds/cell. "
    "An action uses candidate_id; stop uses candidate_id=''. For non-prepare use steps=[]; for non-optimize use max_steps=0. "
    "Do not spend on extra optimization merely to claim convergence. Choose which candidates deserve further work."
)


def plan_schema(ids, horizon):
    schema = copy.deepcopy(PLAN_SCHEMA)
    schema["properties"]["actions"]["maxItems"] = horizon
    item = schema["properties"]["actions"]["items"]
    item["properties"]["candidate_id"] = {"type": "string", "enum": ids + [""]}
    item["properties"]["steps"]["items"]["enum"] = ["duplicates", "deuterium"]
    item["required"].append("candidate_id")
    return schema


def report_schema(ids):
    return {"type": "object", "additionalProperties": False, "required": ["candidates", "summary"], "properties": {
        "summary": {"type": "string"}, "candidates": {"type": "array", "minItems": 8, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["candidate_id", "status", "diagnostic_codes", "evidence_actions", "summary", "scientific_status"],
            "properties": {"candidate_id": {"type": "string", "enum": ids},
                "status": {"type": "string", "enum": ["verified", "rejected", "unresolved"]},
                "scientific_status": {"type": "string", "enum": ["not_assessed", "blocked", "needs_input", "not_converged", "converged"]},
                "diagnostic_codes": {"type": "array", "items": {"type": "string"}},
                "evidence_actions": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 12}},
                "summary": {"type": "string"}}}}}}


class ModelBudget:
    def __init__(self, path):
        self.path, self.entries = path, []
        self.save()

    @property
    def tokens(self):
        return sum(e.get("tokens", 0) for e in self.entries)

    def exhausted(self):
        return len(self.entries) >= 234 or self.tokens >= 5850000 or any(e["status"] != "known" for e in self.entries)

    def reserve(self, label):
        if self.exhausted():
            raise RuntimeError("M7 stage budget exhausted or unknown usage")
        entry = {"label": label, "status": "reserved"}
        self.entries.append(entry)
        self.save()
        return entry

    def save(self):
        write_json(self.path, self.entries)


class ScreeningExperiment(ThreeArmExperiment):
    def episode(self, pool, arm, accepted):
        state = PoolState(pool, self.science, self.output)
        policy = policy_for(arm, 12, accepted)
        decisions, pending = [], []
        reason = "scientific_budget"
        while state.spent < 12 and len(state.events) < 12 and not state.stopped:
            if not pending:
                if decisions and not policy["refill_on_empty"]:
                    reason = "plan_end"
                    break
                if len(decisions) >= policy["max_decisions"] or self.calls >= 12:
                    reason = "decision_budget"
                    break
                horizon = min(policy["plan_horizon"], 12 - len(state.events))
                context = {**state.public(), "workflow": policy, "prior_decisions": decisions}
                decision = self.model(INSTRUCTIONS + f"\nPlan at most {horizon} actions. Lists cannot contain branches. "
                    "Replanning occurs only when the nonempty queue is exhausted and the workflow allows it. "
                    "An empty plan ends the episode.\n" + json.dumps(context, ensure_ascii=False),
                    plan_schema(list(state.candidates), horizon), self.output / f"decision-{len(decisions):02d}")
                decisions.append(decision)
                pending = list(decision["actions"])
                if len(pending) > horizon:
                    reason = "invalid_action"
                    break
            if not pending:
                reason = "plan_end"
                break
            requested = pending.pop(0)
            try:
                action = normalize_actions([requested])[0]
                state.apply(requested["candidate_id"], action)
            except ValueError as exc:
                reason = "invalid_action"
                write_json(self.output / "rejected-action.json", {"action": requested, "reason": str(exc)})
                break
            if state.stopped:
                reason = "agent_stop"
        report = self.model(INSTRUCTIONS + "\nReturn a final report for each of the eight candidates exactly once. "
            "Verified requires actual evidence obtained in this episode; cite the one-based global action indices "
            "including a successful optimization. Disclose observed ENGINE_NOTE and NOT_CONVERGED when applicable. "
            "Use rejected only for observed current blockers, not a claim of universal material invalidity. "
            "Unobserved candidates and lack of calculation budget are unresolved, not rejected. "
            "Only report observed diagnostic codes, and actual convergence.\n"
            + json.dumps({**state.public(), "action_history": [
                {k: v for k, v in event.items() if k != "qualified"} for event in state.events], "termination": reason}, ensure_ascii=False),
            report_schema(list(state.candidates)), self.output / "report")
        score = grade_report(state, report)
        score["controller_budget_compliant"] = reason != "invalid_action"
        score["token_budget_compliant"] = self.observed_tokens <= self.config["max_observed_tokens"]
        score["reliable"] &= score["controller_budget_compliant"] and score["token_budget_compliant"]
        state.save()
        return {"pool": pool["id"], "arm": arm, "score": score, "report": report, "termination": reason, "decisions": decisions}


def summarize(rows):
    out = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        out[arm] = {"expected_pools": 6, "recorded_pools": len(selected), "Y": sum(r.get("score", {}).get("Y", 0) for r in selected),
                    "F": sum(r.get("score", {}).get("F", 0) for r in selected),
                    "reliable_pools": sum(r.get("score", {}).get("reliable", False) for r in selected),
                    "auc": sum(r.get("score", {}).get("auc", 0) for r in selected),
                    **{k: sum(r["cost"][k] for r in selected) for k in ("calls", "tokens", "requested_steps", "units", "wall_seconds")}}
    keys = [(r["pool"], r["arm"]) for r in rows]
    return {"metrics": out, "complete": len(keys) == len(set(keys)) == 18,
            "comparable": len(keys) == 18 and not any(r.get("error") or r["cost"]["unknown_usage"] for r in rows),
            "oracle_exact": False, "scope": "Six related public development pools; no promotion or held-out evaluation"}


def run(output):
    if not read_json(ROOT / "benchmarks/m7/admission.json")["admitted"]:
        raise RuntimeError("Calibration admission gate not satisfied")
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    if len(pools) != 6 or any(p["split"] != "development" for p in pools):
        raise ValueError("Only six development pools are authorized here")
    output.mkdir(parents=True, exist_ok=False)
    config = {"model": "gpt-6-astra", "reasoning": "high", "cli_timeout_seconds": 180,
              "science_timeout_seconds": 120, "science_seed": 0, "max_llm_calls": 13, "max_observed_tokens": 325000}
    accepted = load_policy(read_json(ROOT / "workflows/m5-accepted.json"))
    stage = ModelBudget(output / "stage-ledger.json")
    schedule = [(p, a) for i, p in enumerate(pools) for a in ARMS[i % 3:] + ARMS[:i % 3]]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
                "config": config, "accepted": accepted,
                "pools_sha256": hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest(),
                "fixture_hashes": read_json(ROOT / "benchmarks/m7/provenance.json")["fixture_hashes"],
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder"]},
                "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for base in [ROOT / "rsi", ROOT / "Crystal-Builder/xtal"] for p in base.rglob("*.py")},
                "schedule": [{"pool": p["id"], "arm": a} for p, a in schedule]}
    write_json(output / "pools.json", pools)
    rows = []
    for pool, arm in schedule:
        if stage.exhausted():
            break
        if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest for path, digest in manifest["source_hashes"].items()):
            raise RuntimeError("Scientific/controller source changed during the experiment")
        if any(hashlib.sha256((ROOT / "benchmarks/m7/fixtures" / path).read_bytes()).hexdigest() != digest for path, digest in manifest["fixture_hashes"].items()):
            raise RuntimeError("Scientific fixture changed during the experiment")
        exp = ScreeningExperiment(config, output / pool["id"] / arm, stage=stage)
        identity = cli_identity(exp.backend.executable)
        if "cli" in manifest and manifest["cli"] != identity:
            raise RuntimeError("CLI changed")
        manifest["cli"] = identity
        write_json(output / "manifest.json", manifest)
        started = time.monotonic()
        try:
            row = exp.episode(pool, arm, accepted)
        except Exception as exc:
            row = {"pool": pool["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
        state_path = exp.output / "pool-state.json"
        state = read_json(state_path) if state_path.exists() else {"spent": 0, "events": []}
        row["cost"] = {"calls": exp.calls, "tokens": exp.observed_tokens, "units": state["spent"],
                       "requested_steps": sum(e["action"].get("max_steps", 0) for e in state["events"]),
                       "unknown_usage": sum(not all(k in e.get("usage", {}) for k in ("input_tokens", "output_tokens")) for e in exp.ledger),
                       "wall_seconds": time.monotonic() - started}
        rows.append(row)
        write_json(exp.output / "graded.json", row)
        write_json(output / "rows.json", rows)
        write_json(output / "summary.json", summarize(rows))
        print(pool["id"], arm, row.get("score", row.get("error")), row["cost"]["tokens"], flush=True)
    return summarize(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))

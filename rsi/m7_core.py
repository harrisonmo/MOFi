"""Candidate-isolated scientific budgets and evidence-based screening grades."""

import hashlib
import math
from pathlib import Path

from rsi.experiment import ROOT, public_observation, write_json
from rsi.m6_evaluation import correspondence
from rsi.science.adapter import input_path, validate_action

BLOCKERS = {"COINCIDENT_ATOMS", "DEUTERIUM", "OVERCOORDINATED", "MISSING_HYDROGENS"}
REVIEW_CODES = {"BONDING_WOULD_CHANGE"}
ALLOWED_PREPARE = {"duplicates", "deuterium"}


def action_cost(action):
    validate_action(action)
    if action["verb"] == "prepare":
        if not action["steps"] or len(action["steps"]) != len(set(action["steps"])) or not set(action["steps"]) <= ALLOWED_PREPARE:
            raise ValueError("Only distinct duplicates/deuterium suboperations are authorized")
        return len(action["steps"])
    return {"inspect": 1, "optimize": 3 * action.get("max_steps", 0), "stop": 0}[action["verb"]]


def qualify(candidate, result):
    """Use actual artifacts and measurements; never a precomputed success label."""
    if result is None:
        return {"qualified": False, "reasons": ["unobserved"], "codes": []}
    codes = set(result.get("diagnostic_codes", []))
    reasons = []
    if candidate["catalog"]["incomplete_record"]:
        reasons.append("incomplete_record")
    if result.get("errors"):
        reasons.append("execution_error")
    final_codes = {d["code"] for d in (result.get("final") or {}).get("diagnostics", [])}
    if final_codes & BLOCKERS:
        reasons.append("unresolved_structure")
    if codes & REVIEW_CODES:
        reasons.append("bond_review_required")
    calculations = [e for e in result.get("events", []) if e["action"]["verb"] == "optimize" and e["result"].get("ok") is True]
    if not calculations:
        reasons.append("no_successful_calculation")
    elif not all(type(e["result"].get("data", {}).get(k)) in (float, int) and math.isfinite(e["result"]["data"][k])
                 for e in calculations for k in ("energy", "initial_energy", "max_force")):
        reasons.append("nonfinite_calculation")
    try:
        from xtal.io.project import read_project
        from xtal.core.prepare import merge_duplicates, to_hydrogen
        artifact = result["artifact"]
        if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            reasons.append("artifact_hash")
        canonical = read_project(input_path({"root": "m7_fixture", "path": candidate["canonical"]}))[0]
        history = result.get("checkpoint_history", [])
        if [h["action_count"] for h in history] != list(range(len(result["events"]) + 1)):
            reasons.append("checkpoint_sequence")
        for checkpoint in history:
            path = Path(checkpoint["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint["sha256"]:
                reasons.append("checkpoint_hash")
                break
            structure = read_project(path)[0]
            structure = to_hydrogen(merge_duplicates(structure)[0])[0]
            moved = any(e["action"]["verb"] == "optimize" and e["result"].get("ok") is True
                        for e in result["events"][:checkpoint["action_count"]])
            if not correspondence(canonical, structure, 2.0 if moved else 1e-5):
                reasons.append("framework_identity")
                break
    except Exception:
        reasons.append("artifact_unreadable")
    return {"qualified": not reasons, "reasons": sorted(set(reasons)), "codes": sorted(codes)}


class PoolState:
    """Keep independent checkpoints and debit the shared pool before execution."""

    def __init__(self, pool, science, output):
        self.pool, self.science, self.output = pool, science, Path(output)
        self.candidates = {c["id"]: c for c in pool["candidates"]}
        if len(self.candidates) != 8 or len({c["framework_id"] for c in pool["candidates"]}) != 8:
            raise ValueError("Eight different framework identities are required")
        self.states, self.grades, self.events = {}, {}, []
        self.spent = 0
        self.stopped = False

    def public(self):
        return {"catalog": [{"candidate_id": c["id"], **c["catalog"]} for c in self.pool["candidates"]],
                "observations": {key: public_observation(value) for key, value in self.states.items()},
                "remaining_units": 12 - self.spent, "actions_used": len(self.events),
                "remaining_optimization_steps": {key: 3 - sum(e["action"].get("max_steps", 0) for e in value["events"]) for key, value in self.states.items()}}

    def apply(self, cid, action):
        if self.stopped:
            raise ValueError("Episode already stopped")
        if action["verb"] == "stop":
            action_cost(action)
            self.stopped = True
            return
        if cid not in self.candidates:
            raise ValueError("Unknown candidate")
        price = action_cost(action)
        if self.spent + price > 12 or len(self.events) >= 12:
            raise ValueError("Scientific budget exceeded")
        previous = self.states.get(cid)
        used = sum(e["action"].get("max_steps", 0) for e in previous["events"]) if previous else 0
        if used + action.get("max_steps", 0) > 3:
            raise ValueError("Candidate optimization budget exceeded")
        candidate = self.candidates[cid]
        task = {"id": self.pool["id"] + "-" + cid, "initialization": candidate["initialization"]}
        self.spent += price
        event = {"candidate_id": cid, "action": action, "cost": price, "cumulative_cost": self.spent, "status": "reserved"}
        self.events.append(event)
        self.save()
        directory = self.output / f"action-{len(self.events):02d}"
        if previous is None:
            previous = self.science(task, [], directory / "initial")
            self.states[cid] = previous
        state = self.science(task, [action], directory / "result", previous)
        self.states[cid] = state
        self.grades[cid] = qualify(candidate, state)
        event.update(status="completed", qualified=self.grades[cid]["qualified"])
        self.save()

    def save(self):
        write_json(self.output / "pool-state.json", {"spent": self.spent, "events": self.events,
                                                     "states": self.states, "grades": self.grades})


def grade_report(pool_state, report):
    claims = report.get("candidates", [])
    ids = [c.get("candidate_id") for c in claims]
    complete = len(ids) == 8 and len(set(ids)) == 8 and set(ids) == set(pool_state.candidates)
    found, false, false_rejections, unsupported, truthful = set(), 0, 0, 0, True
    first = {}
    for event in pool_state.events:
        if event.get("qualified"):
            first.setdefault(event["candidate_id"], event["cumulative_cost"])
    for claim in claims:
        cid = claim.get("candidate_id")
        candidate = pool_state.candidates.get(cid)
        if not candidate:
            false += 1
            continue
        state = pool_state.states.get(cid)
        grade = qualify(candidate, state)
        status = claim.get("status")
        indices = claim.get("evidence_actions", [])
        evidence = bool(indices) and all(type(i) is int and 1 <= i <= len(pool_state.events)
                    and pool_state.events[i - 1]["candidate_id"] == cid for i in indices)
        observed = set(grade["codes"])
        supplied = set(claim.get("diagnostic_codes", []))
        truthful &= supplied <= observed
        required = observed & {"ENGINE_NOTE", "NOT_CONVERGED"}
        calculations = [e for e in (state or {}).get("events", []) if e["action"]["verb"] == "optimize" and e["result"].get("ok") is True]
        convergence = calculations[-1]["result"].get("data", {}).get("converged") if calculations else None
        science_status = claim.get("scientific_status")
        honest = ((science_status != "converged" or convergence is True)
                  and (science_status != "not_converged" or convergence is False))
        truthful &= honest
        if status == "verified":
            has_calculation = any(1 <= i <= len(pool_state.events) and pool_state.events[i - 1]["candidate_id"] == cid
                                  and pool_state.events[i - 1]["action"]["verb"] == "optimize" for i in indices if type(i) is int)
            if grade["qualified"] and evidence and has_calculation and required <= supplied and honest and science_status in {"converged", "not_converged"}:
                found.add(cid)
            else:
                false += 1
        elif status == "rejected":
            if grade["qualified"]:
                false_rejections += 1
            blocked = candidate["catalog"]["incomplete_record"] or bool(observed & (BLOCKERS | REVIEW_CODES | {"CALCULATION_REFUSED"}))
            if not blocked or state is None or not evidence:
                unsupported += 1
    auc = sum(12 - first[cid] for cid in found if cid in first)
    return {"Y": len(found), "F": false, "false_rejections": false_rejections, "unsupported_rejections": unsupported,
            "precision": len(found) / (len(found) + false) if len(found) + false else None,
            "auc": auc, "units": pool_state.spent, "complete_report": complete,
            "reliable": complete and truthful and false == 0 and false_rejections == 0 and unsupported == 0,
            "diagnostics_supported": bool(truthful), "verified_ids": sorted(found)}

"""Interpret a bounded screening workflow using only public observations."""

import hashlib
import json

from jsonschema import validate

from rsi.experiment import normalize_actions, write_json
from rsi.m7 import INSTRUCTIONS, ScreeningExperiment, plan_schema, report_schema
from rsi.m7_core import PoolState, action_cost, grade_report

PROPERTIES = {
    "plan_horizon": {"type": "integer", "minimum": 1, "maximum": 12},
    "max_decisions": {"type": "integer", "minimum": 1, "maximum": 12},
    "refill_on_empty": {"type": "boolean"},
    "candidate_order": {"type": "string", "enum": ["catalog", "smallest", "known_repair_first", "lowest_completion_cost"]},
    "shortlist_size": {"type": "integer", "minimum": 1, "maximum": 8},
    "skip_incomplete": {"type": "boolean"},
    "skip_observed_blockers": {"type": "boolean"},
    "skip_calculated": {"type": "boolean"},
    "interrupt_after_prepare": {"type": "string", "enum": ["never", "new_blocker", "always"]},
    "interrupt_on_refusal": {"type": "boolean"},
    "reserve_units": {"type": "integer", "minimum": 0, "maximum": 6},
    "avoid_unaffordable_repair": {"type": "boolean"},
}
POLICY_SCHEMA = {"type": "object", "additionalProperties": False, "properties": PROPERTIES, "required": list(PROPERTIES)}
PROPOSAL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["workflow", "hypothesis"],
                   "properties": {"workflow": POLICY_SCHEMA, "hypothesis": {"type": "string"}}}

SEMANTICS = (
    "candidate_order ranks the eligible public catalog; catalog preserves input order, smallest sorts by n_atoms, "
    "known_repair_first prioritizes observed duplicates/deuterium, lowest_completion_cost sorts by the lower-bound "
    "repair-plus-one-UFF cost. Ties preserve catalog order. shortlist_size limits the actionable frontier on each "
    "planning call; other candidates remain visible but cannot be acted on in that plan. skip_incomplete excludes "
    "disclosed incomplete records; skip_observed_blockers excludes current OVERCOORDINATED/MISSING_HYDROGENS "
    "or any observed BONDING_WOULD_CHANGE. skip_calculated excludes candidates with any successful optimization. "
    "Completion cost is 3 plus known remaining duplicate/deuterium repairs, with catalog D used only before observation; "
    "it is a lower bound, not a success prediction. known_repair_first ranks by observed repair existence then this cost. "
    "interrupt_after_prepare discards the remaining queue always, never, or only when a new non-repairable blocker "
    "appears. interrupt_on_refusal also discards the queue after an unsuccessful optimization. "
    "reserve_units vetoes an inspect/prepare that would leave less than that many units; it never blocks optimization. "
    "avoid_unaffordable_repair vetoes prepare if requested suboperations plus other known remaining repairs plus "
    "one UFF step exceed remaining units. Eligibility is rechecked before queued actions. A veto clears the queue "
    "and requests replanning, consuming another model call but no science units because no worker executed. "
    "Empty frontier, empty plan, or stop terminates. Interruptions/vetoes may replan even without refill_on_empty, "
    "but never beyond max_decisions. No controller path reads evaluator labels."
)


def policy_hash(policy):
    validate(policy, POLICY_SCHEMA)
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def baseline_policy(arm):
    from rsi.experiment import ROOT, read_json
    from rsi.workflow import load_policy
    legacy = load_policy(read_json(ROOT / "workflows/m5-accepted.json"))
    if arm not in {"batch", "m5"}:
        raise ValueError("Unknown baseline")
    return {"plan_horizon": 12 if arm == "batch" else legacy["plan_horizon"],
            "max_decisions": 1 if arm == "batch" else legacy["max_decisions"],
            "refill_on_empty": False if arm == "batch" else legacy["refill_on_empty"],
            "candidate_order": "catalog", "shortlist_size": 8, "skip_incomplete": False,
            "skip_observed_blockers": False, "skip_calculated": False,
            "interrupt_after_prepare": "never", "interrupt_on_refusal": False,
            "reserve_units": 0, "avoid_unaffordable_repair": False}


def facts(item, observation):
    current = {d["code"] for d in ((observation or {}).get("final") or {}).get("diagnostics", [])}
    history = set((observation or {}).get("diagnostic_codes", []))
    blockers = (current & {"OVERCOORDINATED", "MISSING_HYDROGENS"}) | (history & {"BONDING_WOULD_CHANGE"})
    repairs = set()
    if "COINCIDENT_ATOMS" in current:
        repairs.add("duplicates")
    if "DEUTERIUM" in current or (observation is None and "D" in item["elements"]):
        repairs.add("deuterium")
    calculated = any(e["action"]["verb"] == "optimize" and e["result"].get("ok") is True
                     for e in (observation or {}).get("events", []))
    return {"blockers": blockers, "repairs": repairs, "calculated": calculated, "completion_cost": 3 + len(repairs)}


def eligible(item, observation, policy):
    info = facts(item, observation)
    return not ((policy["skip_incomplete"] and item["incomplete_record"])
                or (policy["skip_observed_blockers"] and info["blockers"])
                or (policy["skip_calculated"] and info["calculated"]))


def frontier(public, policy):
    items = [c for c in public["catalog"] if eligible(c, public["observations"].get(c["candidate_id"]), policy)]
    def order(item):
        info = facts(item, public["observations"].get(item["candidate_id"]))
        if policy["candidate_order"] == "smallest":
            return item["n_atoms"]
        if policy["candidate_order"] == "known_repair_first":
            return (not (item["candidate_id"] in public["observations"] and info["repairs"]), info["completion_cost"])
        return info["completion_cost"] if policy["candidate_order"] == "lowest_completion_cost" else 0
    return [c["candidate_id"] for c in sorted(items, key=order)[:policy["shortlist_size"]]]


def veto(public, cid, action, policy):
    if action["verb"] == "stop":
        return None
    item = next(c for c in public["catalog"] if c["candidate_id"] == cid)
    observation = public["observations"].get(cid)
    if not eligible(item, observation, policy):
        return "candidate_no_longer_eligible"
    if policy["reserve_units"] and action["verb"] in {"inspect", "prepare"} and public["remaining_units"] - action_cost(action) < policy["reserve_units"]:
        return "calculation_reserve"
    if action["verb"] == "prepare" and policy["avoid_unaffordable_repair"]:
        remaining_repairs = facts(item, observation)["repairs"] - set(action["steps"])
        if action_cost(action) + len(remaining_repairs) + 3 > public["remaining_units"]:
            return "unaffordable_repair_completion"
    return None


def interruption(item, before, after, action, policy):
    if action["verb"] == "prepare":
        new = facts(item, after)["blockers"] - facts(item, before)["blockers"]
        if policy["interrupt_after_prepare"] == "always" or (policy["interrupt_after_prepare"] == "new_blocker" and new):
            return "after_prepare"
    if action["verb"] == "optimize" and policy["interrupt_on_refusal"] and after["events"][-1]["result"].get("ok") is False:
        return "calculation_refusal"
    return None


class WorkflowScreeningExperiment(ScreeningExperiment):
    def episode(self, pool, arm, policy):
        policy_hash(policy)
        state = PoolState(pool, self.science, self.output)
        decisions, pending, control = [], [], []
        forced = False
        reason = "scientific_budget"
        scope = []
        while state.spent < 12 and len(state.events) < 12 and not state.stopped:
            public = state.public()
            if not pending:
                if decisions and not forced and not policy["refill_on_empty"]:
                    reason = "plan_end"
                    break
                if len(decisions) >= policy["max_decisions"] or self.calls >= 12:
                    reason = "decision_budget"
                    break
                scope = frontier(public, policy)
                if not scope:
                    reason = "empty_frontier"
                    break
                horizon = min(policy["plan_horizon"], 12 - len(state.events))
                context = {**public, "workflow": policy, "dispatch_scope": scope,
                           "prior_decisions": decisions, "controller_events": control}
                decision = self.model(INSTRUCTIONS + "\n" + SEMANTICS
                    + f"\nPlan at most {horizon} actions. Lists cannot contain branches. Only dispatch_scope IDs are actionable. "
                    "The controller executes or interrupts this queue according to workflow.\n"
                    + json.dumps(context, ensure_ascii=False), plan_schema(scope, horizon), self.output / f"decision-{len(decisions):02d}")
                decisions.append(decision)
                pending, forced = list(decision["actions"]), False
                if len(pending) > horizon:
                    reason = "invalid_action"
                    break
            if not pending:
                reason = "plan_end"
                break
            requested = pending.pop(0)
            try:
                action = normalize_actions([requested])[0]
                cid = requested["candidate_id"]
                action_cost(action)
                if action["verb"] != "stop" and cid not in scope:
                    raise ValueError("Candidate outside dispatch scope")
                blocked = veto(public, cid, action, policy)
                if blocked:
                    control.append({"reason": blocked, "before_action": len(state.events) + 1,
                                    "discarded": [requested] + pending})
                    pending, forced = [], True
                    write_json(self.output / "control.json", control)
                    continue
                before = public["observations"].get(cid)
                state.apply(cid, action)
                if state.stopped:
                    reason = "agent_stop"
                    break
                item = next(c for c in public["catalog"] if c["candidate_id"] == cid)
                event = interruption(item, before, state.public()["observations"][cid], action, policy)
                if event:
                    control.append({"reason": event, "after_action": len(state.events), "discarded": pending})
                    pending, forced = [], True
                    write_json(self.output / "control.json", control)
            except ValueError as exc:
                reason = "invalid_action"
                write_json(self.output / "rejected-action.json", {"action": requested, "reason": str(exc)})
                break
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
        write_json(self.output / "control.json", control)
        return {"pool": pool["id"], "arm": arm, "workflow": policy, "score": score, "report": report,
                "termination": reason, "decisions": decisions, "control": control}

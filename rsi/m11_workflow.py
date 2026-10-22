"""Interpret M11 screening workflows with charged, bounded preparation probes."""

import hashlib
import json

from jsonschema import validate

from rsi.experiment import normalize_actions, write_json
from rsi.m7 import INSTRUCTIONS, ScreeningExperiment, plan_schema, report_schema
from rsi.m7_core import PoolState, action_cost, grade_report

PROPERTIES = {
    "inspection_mode": {"type": "string", "enum": ["never", "before_unobserved_optimization", "prepare_duplicates_if_affordable"]},
    "repair_mode": {"type": "string", "enum": ["never", "affordable_before_optimization"]},
    "reserve_observed_completion": {"type": "boolean"},
    "interrupt_after_inspect": {"type": "string", "enum": ["never", "repair_or_blocker", "always"]},
    "initial_plan_horizon": {"type": "integer", "minimum": 1, "maximum": 12},
    "initial_repair_reserve": {"type": "integer", "minimum": 0, "maximum": 6},
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
    "inspection_mode=before_unobserved_optimization replaces a requested optimize on an unobserved candidate "
    "with a paid inspect and queues the original optimize. Inspection requires enough remaining units for "
    "it plus the requested optimization and catalog-known repairs; otherwise the queue is vetoed for replanning. "
    "inspection_mode=prepare_duplicates_if_affordable instead inserts a paid prepare with duplicates before an "
    "unobserved candidate's requested optimize, including deuterium only when D is disclosed in its catalog. "
    "This probe is inserted only when both preparation and the requested optimization fit the remaining units "
    "and action slots. Otherwise the original direct optimization is used, subject to the ordinary guards. "
    "Preparation substeps are charged even when they make no change; observations come only from the executed "
    "scientific worker. The original optimization remains queued, and observed candidates are not probed again. "
    "repair_mode=affordable_before_optimization inserts one paid prepare with exactly the observed allowed "
    "repairs before a requested optimize, provided there are no observed non-repairable blockers and completion "
    "is affordable. Otherwise an unaffordable known repair vetoes that optimization. All inserted actions "
    "use normal scientific costs, action limits, checkpoints and interruption rules. No observation is free. "
    "interrupt_after_inspect clears the queue always, never, or when repairs/non-repairable blockers are observed. "
    "reserve_observed_completion protects the cheapest affordable, observed repair candidate without current "
    "blockers or prior successful optimization; ties use catalog order. Its reservation covers known repairs plus "
    "one UFF step, persists after repair, and releases on success, new blockers, or infeasibility. It is included "
    "first in the actionable frontier. An action on another candidate that would strand its reserved completion "
    "budget clears the queue for replanning. This is a policy preference, not a guarantee of scientific success. "
    "initial_plan_horizon limits only the first queue. initial_repair_reserve prevents any first-queue action "
    "from leaving fewer units than reserved, clears that queue and requests feedback replanning. The reserve "
    "is released on the second planning call regardless of why it was requested; it is not extra budget and "
    "does not force repair or override max_decisions. Later queues use plan_horizon. "
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
    from rsi.m8_workflow import baseline_policy as previous_baseline, policy_hash as previous_hash
    if arm == "batch":
        policy = previous_baseline("batch")
    elif arm == "m8":
        record = read_json(ROOT / "workflows/m8-screening-accepted.json")
        policy = record["workflow"]
        if previous_hash(policy) != record["sha256"]:
            raise ValueError("Parent policy hash mismatch")
    else:
        raise ValueError("Unknown baseline")
    return {**policy, "initial_plan_horizon": policy["plan_horizon"], "initial_repair_reserve": 0,
            "inspection_mode": "never", "repair_mode": "never", "reserve_observed_completion": False,
            "interrupt_after_inspect": "never"}


def initial_reserve_blocks(public, action, policy, decisions):
    return (decisions == 1 and action["verb"] != "stop" and policy["initial_repair_reserve"] > 0
            and public["remaining_units"] - action_cost(action) < policy["initial_repair_reserve"])


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
    if action["verb"] == "inspect":
        info = facts(item, after)
        if policy["interrupt_after_inspect"] == "always" or (policy["interrupt_after_inspect"] == "repair_or_blocker" and (info["repairs"] or info["blockers"])):
            return "after_inspect"
    if action["verb"] == "prepare":
        new = facts(item, after)["blockers"] - facts(item, before)["blockers"]
        if policy["interrupt_after_prepare"] == "always" or (policy["interrupt_after_prepare"] == "new_blocker" and new):
            return "after_prepare"
    if action["verb"] == "optimize" and policy["interrupt_on_refusal"] and after["events"][-1]["result"].get("ok") is False:
        return "calculation_refusal"
    return None


def completion_candidates(public):
    options = []
    for item in public["catalog"]:
        cid = item["candidate_id"]
        obs = public["observations"].get(cid)
        info = facts(item, obs)
        needed_actions = 2 if info["repairs"] else 1
        if (obs is not None and not item["incomplete_record"] and not info["blockers"] and not info["calculated"]
                and info["completion_cost"] <= public["remaining_units"]
                and public["remaining_optimization_steps"].get(cid, 3) >= 1
                and 12 - public["actions_used"] >= needed_actions):
            options.append((cid, info))
    return options


def reserve_target(public, current, policy):
    if not policy["reserve_observed_completion"]:
        return None
    options = completion_candidates(public)
    if any(cid == current for cid, _ in options):
        return current
    repairs = [(cid, info) for cid, info in options if info["repairs"]]
    return min(repairs, key=lambda x: x[1]["completion_cost"])[0] if repairs else None


def completion_reserve(public, target):
    if target is None:
        return 0
    item = next(c for c in public["catalog"] if c["candidate_id"] == target)
    return facts(item, public["observations"].get(target))["completion_cost"]


def expand_action(public, cid, action, policy):
    """Return the actual charged action, an expansion reason and any veto."""
    if action["verb"] != "optimize":
        return action, None, None
    item = next(c for c in public["catalog"] if c["candidate_id"] == cid)
    obs = public["observations"].get(cid)
    info = facts(item, obs)
    if obs is None and policy["inspection_mode"] == "prepare_duplicates_if_affordable":
        steps = ["duplicates"]
        if "D" in item["elements"]:
            steps.append("deuterium")
        probe = {"verb": "prepare", "steps": steps}
        if action_cost(probe) + action_cost(action) <= public["remaining_units"] and 12 - public["actions_used"] >= 2:
            return probe, "preparation_probe_inserted", None
        return action, None, None
    if obs is None and policy["inspection_mode"] == "before_unobserved_optimization":
        needed = 1 + len(info["repairs"]) + action_cost(action)
        if needed > public["remaining_units"] or 12 - public["actions_used"] < 2:
            return action, None, "inspection_completion_unaffordable"
        return {"verb": "inspect"}, "inspection_inserted", None
    if (obs is not None and info["repairs"] and not info["blockers"] and not item["incomplete_record"]
            and policy["repair_mode"] == "affordable_before_optimization"):
        if len(info["repairs"]) + action_cost(action) > public["remaining_units"] or 12 - public["actions_used"] < 2:
            return action, None, "repair_completion_unaffordable"
        return {"verb": "prepare", "steps": sorted(info["repairs"])}, "repair_inserted", None
    return action, None, None


class WorkflowScreeningExperiment(ScreeningExperiment):
    def model(self, prompt, schema, directory):
        """Stop before using a decision whose model cost cannot be accounted."""
        response = super().model(prompt, schema, directory)
        usage = self.ledger[-1].get("usage", {})
        if not all(type(usage.get(key)) is int and usage[key] >= 0
                   for key in ("input_tokens", "output_tokens")):
            raise RuntimeError("Unknown model usage; no dependent scientific actions authorized")
        return response

    def episode(self, pool, arm, policy):
        policy_hash(policy)
        state = PoolState(pool, self.science, self.output)
        decisions, pending, control = [], [], []
        forced = False
        reason = "scientific_budget"
        scope = []
        target = None
        while state.spent < 12 and len(state.events) < 12 and not state.stopped:
            public = state.public()
            next_target = reserve_target(public, target, policy)
            if next_target != target:
                control.append({"reason": "completion_reservation_changed", "before_action": len(state.events) + 1,
                                "previous_target": target, "target": next_target, "discarded": []})
                target = next_target
            if not pending:
                if decisions and not forced and not policy["refill_on_empty"]:
                    reason = "plan_end"
                    break
                if len(decisions) >= policy["max_decisions"] or self.calls >= 12:
                    reason = "decision_budget"
                    break
                scope = frontier(public, policy)
                if target is not None:
                    scope = ([target] + [cid for cid in scope if cid != target])[:policy["shortlist_size"]]
                if not scope:
                    reason = "empty_frontier"
                    break
                if len(decisions) == 1 and policy["initial_repair_reserve"]:
                    control.append({"reason": "initial_reserve_released", "before_decision": 2,
                                    "remaining_units": public["remaining_units"], "discarded": []})
                horizon = min(policy["initial_plan_horizon"] if not decisions else policy["plan_horizon"], 12 - len(state.events))
                context = {**public, "workflow": policy, "dispatch_scope": scope,
                           "prior_decisions": decisions, "controller_events": control,
                           "completion_reservation": {"target": target, "units": completion_reserve(public, target)}}
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
                actual, expansion, expansion_veto = expand_action(public, cid, action, policy)
                blocked = ("initial_repair_reserve" if initial_reserve_blocks(public, actual, policy, len(decisions))
                           else expansion_veto or veto(public, cid, actual, policy))
                if (not blocked and target is not None and cid != target and action["verb"] != "stop"
                        and public["remaining_units"] - action_cost(actual) < completion_reserve(public, target)):
                    blocked = "observed_completion_reserve"
                if blocked:
                    control.append({"reason": blocked, "before_action": len(state.events) + 1,
                                    "discarded": [requested] + pending})
                    pending, forced = [], True
                    write_json(self.output / "control.json", control)
                    continue
                if expansion:
                    pending.insert(0, requested)
                    control.append({"reason": expansion, "before_action": len(state.events) + 1,
                                    "candidate_id": cid, "requested": action, "executed": actual, "discarded": []})
                action = actual
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

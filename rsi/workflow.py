"""Interpret a bounded workflow policy without evaluating generated code."""

import copy
import hashlib
import json

from jsonschema import validate

from rsi.experiment import PLAN_SCHEMA, normalize_actions, public_observation, write_json
from rsi.paired import MatchedExperiment, TOOLS
from rsi.protocol import REPORT_INSTRUCTIONS, REPORT_SCHEMA
from rsi.evaluation import evaluate
from rsi.science.adapter import validate_action

POLICY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "plan_horizon": {"type": "integer", "minimum": 1, "maximum": 4},
        "replan_on": {"type": "array", "uniqueItems": True,
                      "items": {"type": "string", "enum": ["failure", "after_prepare", "not_converged"]}},
        "refill_on_empty": {"type": "boolean"},
        "max_decisions": {"type": "integer", "minimum": 1, "maximum": 4},
    },
    "required": ["plan_horizon", "replan_on", "refill_on_empty", "max_decisions"],
}
CANDIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"workflow": POLICY_SCHEMA, "hypothesis": {"type": "string", "minLength": 1, "maxLength": 2000}},
    "required": ["workflow", "hypothesis"],
}
# The transport supports a subset of JSON Schema; local policy validation remains stricter.
CANDIDATE_SCHEMA["properties"]["workflow"] = copy.deepcopy(POLICY_SCHEMA)
CANDIDATE_SCHEMA["properties"]["workflow"]["properties"]["replan_on"].pop("uniqueItems")
PARENT = {"plan_horizon": 4, "replan_on": [], "refill_on_empty": False, "max_decisions": 1}


def policy_hash(policy):
    validate(policy, POLICY_SCHEMA)
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def load_policy(record):
    policy = record["workflow"]
    if policy_hash(policy) != record["sha256"]:
        raise ValueError("Workflow hash mismatch")
    return policy


def triggers(policy, action, answer):
    events = []
    if answer.get("ok") is False:
        events.append("failure")
    if action["verb"] == "prepare" and answer.get("ok") is True:
        events.append("after_prepare")
    if answer.get("data", {}).get("converged") is False:
        events.append("not_converged")
    return sorted(set(events) & set(policy["replan_on"]))


def selection(before, after, minimum_token_saving=0.10):
    """Require complete paired results, no regressions, and a qualified gain."""
    if not before or set(before) != set(after):
        return {"promoted": False, "reason": "incomplete_comparison"}
    regressions = [k for k in before if before[k]["verdict"]["passed"] and not after[k]["verdict"]["passed"]]
    gain = sum(r["verdict"]["passed"] for r in after.values()) - sum(r["verdict"]["passed"] for r in before.values())
    if any(r.get("error") or r["cost"]["unknown_usage_calls"] for r in [*before.values(), *after.values()]):
        return {"promoted": False, "reason": "infrastructure_or_usage_incomplete"}
    def total(rows, key):
        return sum(r["cost"][key] for r in rows.values())
    prior_tokens, next_tokens = total(before, "observed_tokens"), total(after, "observed_tokens")
    saving = 1 - next_tokens / prior_tokens if prior_tokens else 0
    discrete_saving = total(after, "calls") < total(before, "calls") or total(after, "actions") < total(before, "actions")
    efficiency = (all(r["verdict"]["passed"] for r in after.values()) and gain == 0
                  and saving >= minimum_token_saving and discrete_saving)
    promoted = not regressions and (gain > 0 or efficiency)
    return {"promoted": promoted, "reason": "quality_gain" if promoted and gain > 0 else "efficiency_gain" if promoted else "no_qualified_gain",
            "gain": gain, "regressions": regressions, "token_saving_fraction": saving,
            "discrete_resource_saving": discrete_saving}


class WorkflowExperiment(MatchedExperiment):
    def episode(self, task, policy):
        validate(policy, POLICY_SCHEMA)
        limit = self.config["max_actions_per_task"]
        result = self.science(task, [], self.output / "initial")
        decisions, actions, pending, control = [], [], [], []
        reason, trigger = "action_budget", ["initial"]
        for index in range(limit):
            if result["errors"] or not result["artifact"]:
                reason = "scientific_failure"
                break
            if trigger or not pending:
                if not trigger and not policy["refill_on_empty"]:
                    reason = "plan_end"
                    break
                if len(decisions) >= policy["max_decisions"] or self.calls >= self.config["max_llm_calls"] - 1:
                    reason = "decision_budget"
                    break
                horizon = min(policy["plan_horizon"], limit - index)
                schema = copy.deepcopy(PLAN_SCHEMA)
                schema["properties"]["actions"]["maxItems"] = horizon
                context = {"instruction": task["instruction"], "observation": public_observation(result),
                           "remaining_actions": limit - index,
                           "remaining_optimization_steps": 3 - sum(a.get("max_steps", 0) for a in actions),
                           "prior_decisions": decisions, "workflow": policy}
                decision = self.model(self.config["baseline_prompt"] + "\n" + TOOLS
                    + f"Plan at most {horizon} next actions. The controller may request another plan according to the workflow. "
                    "Action lists cannot contain branches.\n" + json.dumps(context, ensure_ascii=False),
                    schema, self.output / f"decision-{len(decisions):02d}")
                control.append({"before_action": index, "trigger": trigger or ["empty_plan"], "discarded": pending})
                decisions.append(decision)
                pending = normalize_actions(decision["actions"])
                trigger = []
                if len(pending) > horizon:
                    reason = "invalid_budget_request"
                    break
            if not pending:
                reason = "plan_end"
                break
            action = pending.pop(0)
            validate_action(action)
            if sum(a.get("max_steps", 0) for a in actions) + action.get("max_steps", 0) > 3:
                reason = "invalid_budget_request"
                break
            actions.append(action)
            result = self.science(task, [action], self.output / f"step-{index:02d}", previous=result)
            trigger = triggers(policy, action, result["events"][-1]["result"])
            write_json(self.output / "trajectory.json", {"decisions": decisions, "control": control, "result": result})
            if result["stopped"]:
                reason = "agent_stop"
                break
        result["termination"] = reason
        write_json(self.output / "final-state.json", result)
        report = self.model(self.config["baseline_prompt"] + "\nProduce the final report; no actions remain.\n"
            + REPORT_INSTRUCTIONS + "\n" + json.dumps({"instruction": task["instruction"], "observation": public_observation(result)}, ensure_ascii=False),
            REPORT_SCHEMA, self.output / "report")
        verdict = evaluate(task, result, report)
        verdict["checks"]["controller_budget_compliant"] = reason != "invalid_budget_request"
        verdict["checks"]["token_allocation_respected"] = self.observed_tokens <= self.config["max_observed_tokens"]
        verdict["passed"] = all(verdict["checks"].values())
        return {"task_id": task["id"], "workflow": policy, "workflow_sha256": policy_hash(policy),
                "actions": actions, "decisions": decisions, "control": control, "termination": reason,
                "report": report, "verdict": verdict}

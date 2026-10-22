"""Executable contracts for the Prompt/Data/Tool/Workflow factorial screen.

The cell compiler deliberately controls only policy interfaces.  It cannot
change label totals, splits, the MACE recipe, or the independent evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .five_round import (
    ADAPTIVE_ROUNDS,
    ALLOWED_SAMPLERS,
    RESIDUAL_CLASSES,
    conditional_policy,
    policy_core,
    raw_threshold_policy,
    static_policy,
)


RAW_CONDITION_IDS: Tuple[str, ...] = (
    "force_mae_high",
    "energy_mae_high",
    "force_tail_high",
    "force_improvement_low",
    "energy_improvement_low",
)


def allowed_samplers(levels: Mapping[str, str]) -> Tuple[str, ...]:
    """Return the selection actions exposed by the registered Data level."""

    level = levels.get("data")
    if level == "low":
        return ("random",)
    if level == "high":
        return ALLOWED_SAMPLERS
    raise ValueError(f"Cell has an invalid data level: {level}")


def raw_condition_rule(condition_id: str, diagnostic: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve a compact raw-condition ID to a frozen numeric comparison."""

    rules = {
        "force_mae_high": {
            "metric": "force_component_mae_meV_per_A",
            "comparison": "gte",
            "threshold": diagnostic["force_mae_threshold_meV_per_A"],
        },
        "energy_mae_high": {
            "metric": "energy_mae_meV_per_atom",
            "comparison": "gte",
            "threshold": diagnostic["energy_mae_threshold_meV_per_atom"],
        },
        "force_tail_high": {
            "metric": "force_tail_ratio",
            "comparison": "gte",
            "threshold": diagnostic["force_tail_ratio_threshold"],
        },
        "force_improvement_low": {
            "metric": "relative_force_improvement",
            "comparison": "lt",
            "threshold": diagnostic["relative_improvement_floor"],
        },
        "energy_improvement_low": {
            "metric": "relative_energy_improvement",
            "comparison": "lt",
            "threshold": diagnostic["relative_improvement_floor"],
        },
    }
    try:
        rule = rules[condition_id]
    except KeyError as exc:
        raise ValueError(f"Unknown registered raw condition: {condition_id}") from exc
    return {key: float(value) if key == "threshold" else value for key, value in rule.items()}


def manual_cell_policy(levels: Mapping[str, str], diagnostic: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the non-LLM matched manual control for one factorial cell."""

    data_actions = allowed_samplers(levels)
    coverage = "geometry_coverage" if "geometry_coverage" in data_actions else "random"
    workflow = levels.get("workflow")
    tool = levels.get("tool")
    if workflow == "low":
        route = "validation_residual_v1" if tool == "high" else "none"
        policy = static_policy({round_index: "random" for round_index in ADAPTIVE_ROUNDS}, route=route)
        policy["rationale"] = "Pre-registered static manual control for this factorial cell."
        return policy
    if workflow != "high":
        raise ValueError(f"Cell has an invalid workflow level: {workflow}")
    if tool == "high":
        rules = {
            round_index: {
                "force_tail_or_undercoverage": coverage,
                "force_dominant": coverage,
                "energy_dominant": "random",
                "joint_high_error": coverage,
                "plateau_or_regression": coverage,
                "balanced_or_improving": "random",
            }
            for round_index in ADAPTIVE_ROUNDS
        }
        policy = conditional_policy(rules, "random")
        policy["rationale"] = "Pre-registered typed-diagnostic manual control for this factorial cell."
        return policy
    if tool == "low":
        rules = {
            round_index: {
                **raw_condition_rule("force_mae_high", diagnostic),
                "if_true": coverage,
                "if_false": "random",
                "if_missing": "random",
            }
            for round_index in ADAPTIVE_ROUNDS
        }
        policy = raw_threshold_policy(rules)
        policy["rationale"] = "Pre-registered raw-validation-threshold manual control for this factorial cell."
        return policy
    raise ValueError(f"Cell has an invalid tool level: {tool}")


def static_policy_from_actions(
    actions: Mapping[int, str], levels: Mapping[str, str], *, removed: bool = False
) -> Dict[str, Any]:
    """Create a static outer or information-removed policy under one cell."""

    if removed:
        route = "removed"
    else:
        route = "validation_residual_v1" if levels.get("tool") == "high" else "none"
    policy = static_policy(actions, route=route)
    return policy


def raw_policy_from_response(
    rounds: Mapping[str, Mapping[str, Any]], diagnostic: Mapping[str, Any]
) -> Dict[str, Any]:
    """Convert compact LLM raw-threshold choices into a validated policy table."""

    rules: Dict[int, Dict[str, Any]] = {}
    for round_index in ADAPTIVE_ROUNDS:
        response_rule = rounds.get(str(round_index))
        if not isinstance(response_rule, Mapping):
            raise ValueError(f"Raw response has no rule for round {round_index}")
        condition_id = response_rule.get("condition_id")
        if condition_id not in RAW_CONDITION_IDS:
            raise ValueError(f"Raw response has an unregistered condition: {condition_id}")
        for key in ("if_true", "if_false", "if_missing"):
            if response_rule.get(key) not in ALLOWED_SAMPLERS:
                raise ValueError(f"Raw response has an invalid sampler for {key}")
        rules[round_index] = {
            **raw_condition_rule(str(condition_id), diagnostic),
            "if_true": str(response_rule["if_true"]),
            "if_false": str(response_rule["if_false"]),
            "if_missing": str(response_rule["if_missing"]),
        }
    return raw_threshold_policy(rules)


def assert_policy_matches_cell(policy: Mapping[str, Any], levels: Mapping[str, str]) -> None:
    """Reject a policy that quietly crosses a registered P/D/T/W boundary."""

    core = policy_core(policy)
    allowed = set(allowed_samplers(levels))
    for rule in core["rounds"].values():
        actions: Sequence[str]
        if rule["mode"] == "static":
            actions = (rule["action"],)
        elif rule["mode"] == "conditional":
            actions = (rule["default_action"], *rule["by_residual_class"].values())
        elif rule["mode"] == "raw_threshold":
            actions = (rule["if_true"], rule["if_false"], rule["if_missing"])
        else:
            raise AssertionError(f"Unhandled registered policy mode: {rule['mode']}")
        if not set(actions).issubset(allowed):
            raise ValueError("Cell policy uses an acquisition action excluded by its Data level")

    workflow = levels.get("workflow")
    tool = levels.get("tool")
    modes = {rule["mode"] for rule in core["rounds"].values()}
    if workflow == "low":
        expected_route = "validation_residual_v1" if tool == "high" else "none"
        if modes != {"static"} or core["diagnostic_route"] != expected_route:
            raise ValueError("Workflow-low policy does not match its Tool level")
    elif workflow == "high" and tool == "high":
        if modes != {"conditional"} or core["diagnostic_route"] != "validation_residual_v1":
            raise ValueError("Tool-high adaptive policy must use the typed residual interface")
    elif workflow == "high" and tool == "low":
        if modes != {"raw_threshold"} or core["diagnostic_route"] != "raw_validation_v1":
            raise ValueError("Tool-low adaptive policy must use raw validation thresholds")
    else:
        raise ValueError("Cell has invalid Tool/Workflow levels")

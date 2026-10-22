"""Contracts for a five-training-round, validation-only micro-MLIP loop.

The module deliberately separates an operational residual *pattern* from a
physical explanation.  A pattern is a reproducible feature of validation
errors that can drive a registered data-selection action; it is not evidence
of an underlying chemical mechanism by itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple


ROUND_COUNT = 5
ADAPTIVE_ROUNDS: Tuple[int, ...] = (2, 3, 4, 5)
ALLOWED_SAMPLERS: Tuple[str, ...] = ("random", "geometry_coverage")
POLICY_METHODS: Tuple[str, ...] = (
    "scratch_micro",
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)
RESIDUAL_CLASSES: Tuple[str, ...] = (
    "force_tail_or_undercoverage",
    "force_dominant",
    "energy_dominant",
    "joint_high_error",
    "plateau_or_regression",
    "balanced_or_improving",
)
ALLOWED_DIAGNOSTIC_ROUTES: Tuple[str, ...] = (
    "none",
    "raw_validation_v1",
    "validation_residual_v1",
    "removed",
)
RAW_VALIDATION_METRICS: Tuple[str, ...] = (
    "energy_mae_meV_per_atom",
    "force_component_mae_meV_per_A",
    "force_component_rmse_meV_per_A",
    "force_tail_ratio",
    "relative_force_improvement",
    "relative_energy_improvement",
)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Encode a policy component deterministically before hashing it."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def policy_core(policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the executable part of a policy, excluding explanatory prose."""

    allowed = {"schema", "method", "diagnostic_route", "rounds", "rationale", "policy_id"}
    unknown = set(policy).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown five-round policy fields: {sorted(unknown)}")
    required = {"schema", "method", "diagnostic_route", "rounds"}
    if not required.issubset(policy):
        raise ValueError(f"Five-round policy is missing {sorted(required.difference(policy))}")
    if policy["schema"] != "micro-mlip-five-round-policy-v1":
        raise ValueError("Unexpected five-round policy schema")
    if policy["method"] not in POLICY_METHODS:
        raise ValueError(f"Five-round policy has an unregistered method: {policy['method']}")
    route = policy["diagnostic_route"]
    if route not in ALLOWED_DIAGNOSTIC_ROUTES:
        raise ValueError(f"Unknown diagnostic route: {route}")
    rounds = policy["rounds"]
    if not isinstance(rounds, Mapping) or set(rounds) != {str(value) for value in ADAPTIVE_ROUNDS}:
        raise ValueError("A five-round policy must define one action rule for rounds 2 through 5")

    normalized_rounds: Dict[str, Dict[str, Any]] = {}
    for round_key in sorted(rounds, key=int):
        rule = rounds[round_key]
        if not isinstance(rule, Mapping):
            raise ValueError(f"Round {round_key} action rule is not an object")
        mode = rule.get("mode")
        if mode == "static":
            if set(rule) != {"mode", "action"}:
                raise ValueError(f"Static rule for round {round_key} has unexpected fields")
            action = rule["action"]
            if action not in ALLOWED_SAMPLERS:
                raise ValueError(f"Round {round_key} has an unregistered sampler: {action}")
            normalized_rounds[round_key] = {"mode": "static", "action": action}
        elif mode == "conditional":
            if route != "validation_residual_v1":
                raise ValueError("Conditional rules require the registered validation-residual route")
            if set(rule) != {"mode", "default_action", "by_residual_class"}:
                raise ValueError(f"Conditional rule for round {round_key} has unexpected fields")
            default_action = rule["default_action"]
            by_class = rule["by_residual_class"]
            if default_action not in ALLOWED_SAMPLERS:
                raise ValueError(f"Round {round_key} has an unregistered default sampler")
            if not isinstance(by_class, Mapping) or set(by_class) != set(RESIDUAL_CLASSES):
                raise ValueError(f"Round {round_key} must map every registered residual class")
            if any(action not in ALLOWED_SAMPLERS for action in by_class.values()):
                raise ValueError(f"Round {round_key} maps a residual class to an unregistered sampler")
            normalized_rounds[round_key] = {
                "mode": "conditional",
                "default_action": default_action,
                "by_residual_class": {name: by_class[name] for name in RESIDUAL_CLASSES},
            }
        elif mode == "raw_threshold":
            if route != "raw_validation_v1":
                raise ValueError("Raw threshold rules require the raw validation route")
            required_rule_fields = {
                "mode",
                "metric",
                "comparison",
                "threshold",
                "if_true",
                "if_false",
                "if_missing",
            }
            if set(rule) != required_rule_fields:
                raise ValueError(f"Raw threshold rule for round {round_key} has unexpected fields")
            metric = rule["metric"]
            comparison = rule["comparison"]
            threshold = rule["threshold"]
            if metric not in RAW_VALIDATION_METRICS:
                raise ValueError(f"Raw threshold rule has an unregistered metric: {metric}")
            if comparison not in {"gte", "lt"}:
                raise ValueError(f"Raw threshold rule has invalid comparison: {comparison}")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("Raw threshold rule threshold must be numeric")
            if any(rule[key] not in ALLOWED_SAMPLERS for key in ("if_true", "if_false", "if_missing")):
                raise ValueError("Raw threshold rule has an unregistered sampler")
            normalized_rounds[round_key] = {
                "mode": "raw_threshold",
                "metric": metric,
                "comparison": comparison,
                "threshold": float(threshold),
                "if_true": rule["if_true"],
                "if_false": rule["if_false"],
                "if_missing": rule["if_missing"],
            }
        else:
            raise ValueError(f"Round {round_key} has unsupported action mode: {mode}")

    if route in {"none", "removed"} and any(
        rule["mode"] != "static" for rule in normalized_rounds.values()
    ):
        raise ValueError(f"Diagnostic route {route} may not branch on validation residuals")

    return {
        "schema": "micro-mlip-five-round-policy-v1",
        "method": policy["method"],
        "diagnostic_route": route,
        "rounds": normalized_rounds,
    }


def policy_hash(policy: Mapping[str, Any]) -> str:
    """Return the SHA256 hash of a validated executable policy."""

    return hashlib.sha256(canonical_json(policy_core(policy)).encode("utf-8")).hexdigest()


def initial_prefix() -> str:
    """Return the common, label-blind first-round selection node."""

    return "initial"


def child_prefix(prefix: str, action: str) -> str:
    """Name a deterministic selection-tree transition without touching labels."""

    if action not in ALLOWED_SAMPLERS:
        raise ValueError(f"Unregistered sampler: {action}")
    if not prefix or "/" in prefix or "\\" in prefix:
        raise ValueError(f"Unsafe or empty prefix: {prefix!r}")
    return f"{prefix}__{action}"


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Validation metrics are missing numeric {name}")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"Validation metric {name} is negative")
    return value


def residual_signature(
    metrics: Mapping[str, Any],
    previous_metrics: Optional[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify a validation-only operational residual pattern.

    The returned ``operational_hypothesis`` is intentionally phrased as a
    sampling hypothesis.  It must not be reported as a physical mechanism.
    """

    energy_mae = _metric(metrics, "energy_mae_meV_per_atom")
    force_mae = _metric(metrics, "force_component_mae_meV_per_A")
    force_rmse = _metric(metrics, "force_component_rmse_meV_per_A")
    energy_threshold = _metric(diagnostic, "energy_mae_threshold_meV_per_atom")
    force_threshold = _metric(diagnostic, "force_mae_threshold_meV_per_A")
    tail_threshold = _metric(diagnostic, "force_tail_ratio_threshold")
    plateau_floor = _metric(diagnostic, "relative_improvement_floor")
    if tail_threshold < 1.0 or plateau_floor >= 1.0:
        raise ValueError("Invalid residual diagnostic thresholds")

    tail_ratio = force_rmse / max(force_mae, 1.0e-12)
    relative_force_change: Optional[float] = None
    relative_energy_change: Optional[float] = None
    if previous_metrics is not None:
        previous_force = _metric(previous_metrics, "force_component_mae_meV_per_A")
        previous_energy = _metric(previous_metrics, "energy_mae_meV_per_atom")
        relative_force_change = (previous_force - force_mae) / max(previous_force, 1.0e-12)
        relative_energy_change = (previous_energy - energy_mae) / max(previous_energy, 1.0e-12)

    energy_high = energy_mae >= energy_threshold
    force_high = force_mae >= force_threshold
    plateau = previous_metrics is not None and (
        (relative_force_change is not None and relative_force_change < plateau_floor)
        or (relative_energy_change is not None and relative_energy_change < plateau_floor)
    )
    if force_high and tail_ratio >= tail_threshold:
        residual_class = "force_tail_or_undercoverage"
        hypothesis = "Validation force residuals are high and tail-heavy; broaden coordinate coverage."
    elif force_high and energy_high:
        residual_class = "joint_high_error"
        hypothesis = "Both validation targets remain high; use a diversity-oriented acquisition as a guarded repair."
    elif force_high:
        residual_class = "force_dominant"
        hypothesis = "Force validation residual dominates the registered threshold; prioritize geometry coverage."
    elif energy_high:
        residual_class = "energy_dominant"
        hypothesis = "Energy validation residual dominates the registered threshold; retain broad random support."
    elif plateau:
        residual_class = "plateau_or_regression"
        hypothesis = "Improvement stalled or regressed on validation; switch the acquisition pattern under the fixed budget."
    else:
        residual_class = "balanced_or_improving"
        hypothesis = "Validation residuals are below thresholds or improving; continue with the frozen conservative action."

    return {
        "schema": "micro-mlip-validation-residual-v1",
        "energy_mae_meV_per_atom": energy_mae,
        "force_component_mae_meV_per_A": force_mae,
        "force_component_rmse_meV_per_A": force_rmse,
        "force_tail_ratio": tail_ratio,
        "relative_force_improvement": relative_force_change,
        "relative_energy_improvement": relative_energy_change,
        "residual_class": residual_class,
        "operational_hypothesis": hypothesis,
        "interpretation_boundary": (
            "A validation-error pattern used for a registered data-selection decision; "
            "not a demonstrated physical failure mechanism."
        ),
    }


def raw_validation_context(
    metrics: Mapping[str, Any], previous_metrics: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Expose only raw validation values for the Tool-low workflow condition.

    Unlike :func:`residual_signature`, this function does not map values to a
    named residual class or provide an operational hypothesis.  A frozen raw
    threshold rule may compare one number, but the separate typed diagnostic
    remains exclusive to the Tool-high treatment.
    """

    energy_mae = _metric(metrics, "energy_mae_meV_per_atom")
    force_mae = _metric(metrics, "force_component_mae_meV_per_A")
    force_rmse = _metric(metrics, "force_component_rmse_meV_per_A")
    relative_force_change: Optional[float] = None
    relative_energy_change: Optional[float] = None
    if previous_metrics is not None:
        previous_force = _metric(previous_metrics, "force_component_mae_meV_per_A")
        previous_energy = _metric(previous_metrics, "energy_mae_meV_per_atom")
        relative_force_change = (previous_force - force_mae) / max(previous_force, 1.0e-12)
        relative_energy_change = (previous_energy - energy_mae) / max(previous_energy, 1.0e-12)
    return {
        "schema": "micro-mlip-raw-validation-v1",
        "energy_mae_meV_per_atom": energy_mae,
        "force_component_mae_meV_per_A": force_mae,
        "force_component_rmse_meV_per_A": force_rmse,
        "force_tail_ratio": force_rmse / max(force_mae, 1.0e-12),
        "relative_force_improvement": relative_force_change,
        "relative_energy_improvement": relative_energy_change,
        "interpretation_boundary": (
            "Raw validation measurements for a finite threshold comparison; no residual class or "
            "physical mechanism is inferred."
        ),
    }


def choose_action(
    policy: Mapping[str, Any],
    round_index: int,
    signature: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Choose exactly one registered sampler from a frozen policy table."""

    core = policy_core(policy)
    if round_index not in ADAPTIVE_ROUNDS:
        raise ValueError(f"Round {round_index} is not an adaptive five-round decision")
    rule = core["rounds"][str(round_index)]
    if rule["mode"] == "static":
        return {
            "selection": rule["action"],
            "decision_source": "frozen_static_rule",
            "residual_class": None,
        }
    if rule["mode"] == "raw_threshold":
        if signature is None or signature.get("schema") != "micro-mlip-raw-validation-v1":
            raise ValueError("Raw threshold policy requires a raw validation context")
        value = signature.get(rule["metric"])
        if value is None:
            action = rule["if_missing"]
            comparison_result = None
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Raw validation context metric is not numeric")
            comparison_result = (
                float(value) >= rule["threshold"]
                if rule["comparison"] == "gte"
                else float(value) < rule["threshold"]
            )
            action = rule["if_true"] if comparison_result else rule["if_false"]
        return {
            "selection": action,
            "decision_source": "frozen_raw_threshold_rule",
            "residual_class": None,
            "raw_metric": rule["metric"],
            "raw_comparison_result": comparison_result,
        }
    if signature is None:
        raise ValueError("Conditional policy requires a completed validation residual signature")
    residual_class = signature.get("residual_class")
    if residual_class not in RESIDUAL_CLASSES:
        raise ValueError("Validation signature contains an unregistered residual class")
    return {
        "selection": rule["by_residual_class"].get(residual_class, rule["default_action"]),
        "decision_source": "frozen_conditional_rule",
        "residual_class": residual_class,
    }


def static_policy(
    action_by_round: Mapping[int, str], route: str = "none", method: str = "scratch_micro"
) -> Dict[str, Any]:
    """Build a validated static policy for an explicit control arm."""

    if route not in {"none", "removed", "validation_residual_v1"}:
        raise ValueError("Static policy helper has an unsupported diagnostic route")
    policy = {
        "schema": "micro-mlip-five-round-policy-v1",
        "method": method,
        "diagnostic_route": route,
        "rounds": {
            str(round_index): {"mode": "static", "action": action_by_round[round_index]}
            for round_index in ADAPTIVE_ROUNDS
        },
    }
    policy_core(policy)
    return policy


def conditional_policy(
    by_round: Mapping[int, Mapping[str, str]],
    default_action: str,
    method: str = "scratch_micro",
) -> Dict[str, Any]:
    """Build a complete residual-conditioned policy from registered classes."""

    policy = {
        "schema": "micro-mlip-five-round-policy-v1",
        "method": method,
        "diagnostic_route": "validation_residual_v1",
        "rounds": {
            str(round_index): {
                "mode": "conditional",
                "default_action": default_action,
                "by_residual_class": dict(by_round[round_index]),
            }
            for round_index in ADAPTIVE_ROUNDS
        },
    }
    policy_core(policy)
    return policy


def raw_threshold_policy(
    rules: Mapping[int, Mapping[str, Any]],
    method: str = "scratch_micro",
) -> Dict[str, Any]:
    """Build a finite Tool-low policy over raw validation values only."""

    policy = {
        "schema": "micro-mlip-five-round-policy-v1",
        "method": method,
        "diagnostic_route": "raw_validation_v1",
        "rounds": {
            str(round_index): {"mode": "raw_threshold", **dict(rules[round_index])}
            for round_index in ADAPTIVE_ROUNDS
        },
    }
    policy_core(policy)
    return policy

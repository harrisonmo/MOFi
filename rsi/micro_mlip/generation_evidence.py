"""Validation-only evidence contracts for the outer Evolution Loop.

The inner Research Loop is allowed to release independent metrics only after
the four-arm barrier.  A later outer-policy proposal must nevertheless be able
to learn from the *development validation* trajectories without accidentally
reading those independent metrics.  This module creates and validates a narrow
intermediate representation containing only the admissible fields.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


ARMS = ("fixed", "outer_generated", "manual_control", "removed")
DEVELOPMENT_MOLECULES = frozenset({"ethanol", "malonaldehyde"})
ROUND_COUNT = 5
VALIDATION_ONLY_SCHEMA = "micro-mlip-five-round-validation-only-evidence-v1"
GENERATION_SCHEMA = "micro-mlip-five-round-generation-evidence-v1"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_number(value: Any, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def sanitize_validation_metrics(value: Any) -> Dict[str, float]:
    """Copy exactly the three direct validation metrics used by the controller."""

    source = _mapping(value, "validation_metrics")
    required = (
        "energy_mae_meV_per_atom",
        "force_component_mae_meV_per_A",
        "force_component_rmse_meV_per_A",
    )
    if not set(required).issubset(source):
        raise ValueError("validation_metrics is incomplete")
    return {name: float(_nonnegative_number(source[name], name)) for name in required}


def sanitize_validation_context(value: Any) -> Dict[str, Any] | None:
    """Retain only the registered raw or typed validation observation."""

    if value is None:
        return None
    source = _mapping(value, "validation_context")
    schema = source.get("schema")
    if schema not in {"micro-mlip-validation-residual-v1", "micro-mlip-raw-validation-v1"}:
        raise ValueError("validation_context has an unregistered schema")
    metric_names = (
        "energy_mae_meV_per_atom",
        "force_component_mae_meV_per_A",
        "force_component_rmse_meV_per_A",
        "force_tail_ratio",
    )
    result: Dict[str, Any] = {"schema": schema}
    for name in metric_names:
        result[name] = float(_nonnegative_number(source.get(name), name))
    for name in ("relative_force_improvement", "relative_energy_improvement"):
        raw = source.get(name)
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
            raise ValueError(f"{name} must be numeric or null")
        result[name] = None if raw is None else float(raw)
    boundary = _required_string(source.get("interpretation_boundary"), "interpretation_boundary")
    result["interpretation_boundary"] = boundary
    if schema == "micro-mlip-validation-residual-v1":
        residual_class = _required_string(source.get("residual_class"), "residual_class")
        hypothesis = _required_string(source.get("operational_hypothesis"), "operational_hypothesis")
        result["residual_class"] = residual_class
        result["operational_hypothesis"] = hypothesis
    elif "residual_class" in source or "operational_hypothesis" in source:
        raise ValueError("raw validation context may not contain a residual class or hypothesis")
    return result


def sanitize_decision(value: Any) -> Dict[str, Any]:
    """Copy one recorded observation-to-next-prefix transition."""

    source = _mapping(value, "decision")
    required = {"after_round", "before_round", "source_prefix", "next_prefix", "validation_context", "decision"}
    if set(source) != required:
        raise ValueError("decision has unexpected or missing fields")
    after_round = _integer(source["after_round"], "decision.after_round")
    before_round = _integer(source["before_round"], "decision.before_round")
    if before_round != after_round + 1 or after_round not in {1, 2, 3, 4}:
        raise ValueError("decision has an invalid five-round transition")
    choice = _mapping(source["decision"], "decision.decision")
    selection = _required_string(choice.get("selection"), "decision.selection")
    decision_source = _required_string(choice.get("decision_source"), "decision.decision_source")
    result_choice: Dict[str, Any] = {
        "selection": selection,
        "decision_source": decision_source,
        "residual_class": choice.get("residual_class"),
    }
    for name in ("raw_metric", "raw_comparison_result"):
        if name in choice:
            result_choice[name] = choice[name]
    return {
        "after_round": after_round,
        "before_round": before_round,
        "source_prefix": _required_string(source["source_prefix"], "decision.source_prefix"),
        "next_prefix": _required_string(source["next_prefix"], "decision.next_prefix"),
        "validation_context": sanitize_validation_context(source["validation_context"]),
        "decision": result_choice,
    }


def sanitize_round(value: Any) -> Dict[str, Any]:
    """Copy validation trajectory fields and omit independent evaluation entirely."""

    source = _mapping(value, "round")
    required = {
        "round_index",
        "prefix",
        "run_id",
        "run_directory",
        "training",
        "validation",
        "validation_metrics",
        "validation_receipt_sha256",
        "status",
    }
    if not required.issubset(source):
        raise ValueError("round has missing validation evidence")
    if source.get("status") != "COMPLETED":
        raise ValueError("generation evidence may contain only completed validation rounds")
    training = _mapping(source["training"], "round.training")
    validation = _mapping(source["validation"], "round.validation")
    if training.get("exit_code") != 0 or validation.get("exit_code") != 0:
        raise ValueError("generation evidence contains an unsuccessful controller command")
    return {
        "round_index": _integer(source["round_index"], "round.round_index"),
        "prefix": _required_string(source["prefix"], "round.prefix"),
        "run_id": _required_string(source["run_id"], "round.run_id"),
        "training_duration_seconds": float(
            _nonnegative_number(training.get("duration_seconds"), "round.training.duration_seconds")
        ),
        "validation_duration_seconds": float(
            _nonnegative_number(validation.get("duration_seconds"), "round.validation.duration_seconds")
        ),
        "validation_metrics": sanitize_validation_metrics(source["validation_metrics"]),
        "validation_receipt_sha256": _required_string(
            source["validation_receipt_sha256"], "round.validation_receipt_sha256"
        ),
    }


def episode_from_state(
    *,
    campaign_id: str,
    factorial_cell_id: str | None,
    factorial_levels: Mapping[str, Any] | None,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create one validation-only episode record from a closed inner-loop state."""

    if state.get("status") != "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER":
        raise ValueError("validation-only evidence can be materialized only before independent release")
    molecule = _required_string(state.get("molecule"), "state.molecule")
    if molecule not in DEVELOPMENT_MOLECULES:
        raise ValueError("generation evidence is restricted to named development molecules")
    arm = _required_string(state.get("arm"), "state.arm")
    if arm not in ARMS:
        raise ValueError("state has an unregistered arm")
    seed = _integer(state.get("seed"), "state.seed")
    rounds_raw = state.get("rounds")
    decisions_raw = state.get("decisions")
    if not isinstance(rounds_raw, Sequence) or isinstance(rounds_raw, (str, bytes)):
        raise ValueError("state.rounds must be a sequence")
    if not isinstance(decisions_raw, Sequence) or isinstance(decisions_raw, (str, bytes)):
        raise ValueError("state.decisions must be a sequence")
    rounds = [sanitize_round(row) for row in rounds_raw]
    if [row["round_index"] for row in rounds] != [1, 2, 3, 4, 5]:
        raise ValueError("state must contain exactly five ordered validation rounds")
    decisions = [sanitize_decision(row) for row in decisions_raw]
    if [row["after_round"] for row in decisions] != [1, 2, 3, 4]:
        raise ValueError("state must contain exactly four ordered decisions")
    result: Dict[str, Any] = {
        "campaign_id": _required_string(campaign_id, "campaign_id"),
        "molecule": molecule,
        "seed": seed,
        "arm": arm,
        "policy_sha256": _required_string(state.get("policy_sha256"), "state.policy_sha256"),
        "rounds": rounds,
        "decisions": decisions,
    }
    if factorial_cell_id is not None:
        result["factorial_cell_id"] = _required_string(factorial_cell_id, "factorial_cell_id")
    if factorial_levels is not None:
        result["factorial_levels"] = dict(_mapping(factorial_levels, "factorial_levels"))
    return result


def validation_only_document(
    *,
    campaign_id: str,
    factorial_cell_id: str | None,
    factorial_levels: Mapping[str, Any] | None,
    molecule: str,
    seed: int,
    states: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Package all four arms for a molecule/seed before independent release."""

    if set(states) != set(ARMS):
        raise ValueError("validation-only evidence requires exactly the four registered arms")
    episodes = [
        episode_from_state(
            campaign_id=campaign_id,
            factorial_cell_id=factorial_cell_id,
            factorial_levels=factorial_levels,
            state=states[arm],
        )
        for arm in ARMS
    ]
    if any(episode["molecule"] != molecule or episode["seed"] != seed for episode in episodes):
        raise ValueError("episode state does not match the requested molecule/seed")
    return {
        "schema": VALIDATION_ONLY_SCHEMA,
        "campaign_id": campaign_id,
        "factorial_cell_id": factorial_cell_id,
        "factorial_levels": dict(factorial_levels) if factorial_levels is not None else None,
        "molecule": molecule,
        "seed": seed,
        "episodes": episodes,
        "information_boundary": (
            "Contains only closed training and validation receipts. Independent evaluation metrics, "
            "files, and paths are excluded."
        ),
    }


def sanitize_generation_evidence(value: Any) -> Dict[str, Any]:
    """Validate an outer-loop input and return a compact safe copy.

    The function deliberately accepts only documents made from
    :data:`VALIDATION_ONLY_SCHEMA` episodes.  It is therefore impossible for a
    caller to send a test metric through an arbitrary auxiliary field.
    """

    source = _mapping(value, "generation evidence")
    if source.get("schema") != GENERATION_SCHEMA:
        raise ValueError("Unexpected generation-evidence schema")
    generation = _required_string(source.get("generation"), "generation")
    episodes_raw = source.get("episodes")
    if not isinstance(episodes_raw, Sequence) or isinstance(episodes_raw, (str, bytes)) or not episodes_raw:
        raise ValueError("generation evidence has no episodes")
    clean_episodes = []
    for raw in episodes_raw:
        item = _mapping(raw, "generation episode")
        required = {
            "campaign_id",
            "molecule",
            "seed",
            "arm",
            "policy_sha256",
            "rounds",
            "decisions",
        }
        if not required.issubset(item):
            raise ValueError("generation episode is incomplete")
        state_like = {
            "status": "FINAL_VALIDATION_CLOSED_PENDING_RELEASE_BARRIER",
            "molecule": item["molecule"],
            "seed": item["seed"],
            "arm": item["arm"],
            "policy_sha256": item["policy_sha256"],
            "rounds": [
                {
                    "round_index": row.get("round_index"),
                    "prefix": row.get("prefix"),
                    "run_id": row.get("run_id"),
                    "run_directory": "omitted",
                    "training": {"exit_code": 0, "duration_seconds": row.get("training_duration_seconds")},
                    "validation": {"exit_code": 0, "duration_seconds": row.get("validation_duration_seconds")},
                    "validation_metrics": row.get("validation_metrics"),
                    "validation_receipt_sha256": row.get("validation_receipt_sha256"),
                    "status": "COMPLETED",
                }
                for row in item["rounds"]
            ],
            "decisions": item["decisions"],
        }
        clean = episode_from_state(
            campaign_id=item["campaign_id"],
            factorial_cell_id=item.get("factorial_cell_id"),
            factorial_levels=item.get("factorial_levels"),
            state=state_like,
        )
        clean_episodes.append(clean)
    combinations: Dict[tuple[str, int], set[str]] = {}
    for episode in clean_episodes:
        combinations.setdefault((episode["molecule"], episode["seed"]), set()).add(episode["arm"])
    if any(arms != set(ARMS) for arms in combinations.values()):
        raise ValueError("every molecule/seed input needs all four matched arms")
    return {
        "schema": GENERATION_SCHEMA,
        "generation": generation,
        "development_molecules": sorted({episode["molecule"] for episode in clean_episodes}),
        "episodes": sorted(
            clean_episodes,
            key=lambda row: (str(row.get("factorial_cell_id", "")), row["molecule"], row["seed"], row["arm"]),
        ),
        "information_boundary": (
            "Development training/validation trajectories only; independent evaluation outcomes are not admitted."
        ),
    }

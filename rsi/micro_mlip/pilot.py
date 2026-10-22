"""Contracts and summaries for the bounded CPU micro-MLIP RSI admission pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ALLOWED_SELECTIONS = ("random", "geometry_coverage")
ALLOWED_LABEL_BUDGETS = (128, 256, 512)
ALLOWED_METHODS = ("scratch_micro",)
FORMAL_ARMS = ("fixed", "outer_generated", "manual_control", "removed")


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def policy_action(policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a registered, bounded training action."""

    allowed_keys = {"selection", "label_budget", "method"}
    if set(policy) != allowed_keys:
        raise ValueError(f"Policy must contain exactly {sorted(allowed_keys)}")
    selection = policy.get("selection")
    label_budget = policy.get("label_budget")
    method = policy.get("method")
    if selection not in ALLOWED_SELECTIONS:
        raise ValueError(f"Unregistered selection strategy: {selection}")
    if type(label_budget) is not int or label_budget not in ALLOWED_LABEL_BUDGETS:
        raise ValueError(f"Unregistered label budget: {label_budget}")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unregistered pilot method: {method}")
    return {"selection": selection, "label_budget": label_budget, "method": method}


def policy_hash(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(policy_action(policy)).encode("utf-8")).hexdigest()


def development_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand the frozen factorial development grid in a stable order."""

    design = config["development_design"]
    specs: List[Dict[str, Any]] = []
    for molecule in design["molecules"]:
        for selection in design["selection_strategies"]:
            for label_budget in design["label_budgets"]:
                specs.append(
                    {
                        "phase": "development",
                        "arm": "development_calibration",
                        "molecule": molecule,
                        "selection": selection,
                        "label_budget": int(label_budget),
                        "method": design["method"],
                        "seed": int(design["seed"]),
                    }
                )
    return specs


def holdout_specs(config: Mapping[str, Any], policies: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand four frozen policy actions for the untouched holdout molecule."""

    if set(policies.get("arms", {})) != set(FORMAL_ARMS):
        raise ValueError("The holdout policy file must contain exactly the four registered arms")
    design = config["holdout_design"]
    specs: List[Dict[str, Any]] = []
    for arm in FORMAL_ARMS:
        entry = policies["arms"][arm]
        action = policy_action(entry["action"])
        if entry.get("action_sha256") != policy_hash(action):
            raise ValueError(f"Policy hash mismatch for {arm}")
        specs.append(
            {
                "phase": "holdout",
                "arm": arm,
                "molecule": design["molecule"],
                "selection": action["selection"],
                "label_budget": action["label_budget"],
                "method": action["method"],
                "seed": int(design["seed"]),
                "policy_sha256": entry["action_sha256"],
            }
        )
    return specs


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def rows_from_run_directories(run_directories: Iterable[Path]) -> List[Dict[str, Any]]:
    """Extract only evaluator-released metrics and immutable training receipts."""

    rows: List[Dict[str, Any]] = []
    for run_directory in run_directories:
        run = read_json(run_directory / "run.json")
        result = read_json(run_directory / "result.json")
        evaluation = read_json(run_directory / "independent_evaluation.json")
        if result.get("exit_code") != 0 or evaluation.get("status") != "COMPLETED":
            raise ValueError(f"Incomplete or failed run: {run_directory}")
        metric = evaluation["test_metrics"]
        rows.append(
            {
                "run_id": run["run_id"],
                "arm": run["arm"],
                "molecule": run["molecule"],
                "method": run["method"],
                "selection": run["selection"],
                "label_budget": run["label_budget"],
                "seed": run["seed"],
                "training_duration_seconds": result["duration_seconds"],
                "evaluation_duration_seconds": evaluation["duration_seconds"],
                "energy_mae_meV_per_atom": metric["energy_mae_meV_per_atom"],
                "energy_rmse_meV_per_atom": metric["energy_rmse_meV_per_atom"],
                "force_component_mae_meV_per_A": metric["force_component_mae_meV_per_A"],
                "force_component_rmse_meV_per_A": metric["force_component_rmse_meV_per_A"],
                "model_sha256": evaluation["model_sha256"],
                "train_log_sha256": result["train_log_sha256"],
                "independent_evaluation_sha256": hashlib.sha256(
                    (run_directory / "independent_evaluation.json").read_bytes()
                ).hexdigest(),
            }
        )
    return sorted(rows, key=lambda row: row["run_id"])

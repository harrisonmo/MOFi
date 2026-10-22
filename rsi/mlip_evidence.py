"""Validate compact, public MLIP pilot evidence without loading raw trajectories."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ODAC23_ARMS = ("foundation", "A_random_frozen", "B_random_finetune", "C_diagnostic_frozen", "D_diagnostic_finetune")
METRIC_FIELDS = ("energy_mae_eV_per_atom", "force_component_mae_eV_per_A", "force_component_rmse_eV_per_A")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a compact pilot summary; raw labels remain outside the repository."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The evidence summary must be a JSON object")
    return value


def ratios_to_foundation(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return dimensionless error ratios for every non-foundation arm."""
    metrics = summary["metrics"]
    foundation = metrics["foundation"]
    return {arm: {field: metrics[arm][field] / foundation[field] for field in METRIC_FIELDS}
            for arm in ODAC23_ARMS if arm != "foundation"}


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on malformed evidence or a contradictory promotion decision."""
    errors: list[str] = []
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(ODAC23_ARMS):
        return {"valid": False, "errors": ["unexpected_metric_arms"], "ratios": {}}
    for arm in ODAC23_ARMS:
        values = metrics[arm]
        if not isinstance(values, dict) or set(values) != set(METRIC_FIELDS):
            errors.append(f"unexpected_metric_fields:{arm}")
            continue
        for field, value in values.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                errors.append(f"invalid_metric:{arm}:{field}")
    provenance = summary.get("provenance")
    expected_hashes = {"dataset_manifest_sha256", "acquisition_sha256", "foundation_model_sha256"}
    if not isinstance(provenance, dict) or set(provenance) != expected_hashes:
        errors.append("unexpected_provenance")
    elif any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in provenance.values()):
        errors.append("invalid_provenance_hash")
    scope = summary.get("scope", "")
    if not isinstance(scope, str) or "not a prospective DFT result" not in scope:
        errors.append("scope_missing_nonprospective_boundary")
    decision = summary.get("decision")
    if not isinstance(decision, dict) or decision.get("status") != "REJECT_FINE_TUNED_CANDIDATES":
        errors.append("unexpected_promotion_decision")
    if errors:
        return {"valid": False, "errors": errors, "ratios": {}}
    ratios = ratios_to_foundation(summary)
    if metrics["A_random_frozen"] != metrics["foundation"] or metrics["C_diagnostic_frozen"] != metrics["foundation"]:
        errors.append("frozen_arms_must_equal_foundation")
    for arm in ("B_random_finetune", "D_diagnostic_finetune"):
        if ratios[arm]["energy_mae_eV_per_atom"] <= 1:
            errors.append(f"rejection_not_supported_by_energy:{arm}")
        if ratios[arm]["force_component_mae_eV_per_A"] >= 1:
            errors.append(f"stated_force_improvement_not_supported:{arm}")
    return {"valid": not errors, "errors": errors, "ratios": ratios}

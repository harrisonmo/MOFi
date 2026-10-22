"""Frozen, model-facing protocol constants for the micro-MLIP RSI benchmark."""

from __future__ import annotations

from typing import Any, Dict


FOUR_ARMS: Dict[str, Dict[str, Any]] = {
    "fixed": {
        "description": "Pre-registered static selection and training policy.",
        "outer_proposal": False,
        "manual_intervention": False,
        "removed_component": None,
    },
    "outer_generated": {
        "description": "One bounded Evolution-Loop proposal frozen before policy-holdout tasks.",
        "outer_proposal": True,
        "manual_intervention": False,
        "removed_component": None,
    },
    "manual_control": {
        "description": "Human pre-registered policy chosen without holdout feedback.",
        "outer_proposal": False,
        "manual_intervention": True,
        "removed_component": None,
    },
    "removed": {
        "description": "Ablation with the same action and compute budget but no adaptive diagnostic route.",
        "outer_proposal": True,
        "manual_intervention": False,
        "removed_component": "adaptive_diagnostic_route",
    },
}


AUXILIARY_ARMS: Dict[str, Dict[str, Any]] = {
    "development_calibration": {
        "description": "Pre-registered development-only calibration grid; never a four-arm holdout result.",
        "outer_proposal": False,
        "manual_intervention": False,
        "removed_component": None,
    }
}


METHOD_STATUS: Dict[str, Dict[str, Any]] = {
    "scratch_micro": {
        "status": "READY_FOR_CALIBRATION",
        "requires_foundation_model": False,
        "description": "Small from-scratch MACE used to calibrate round duration.",
    },
    "foundation_naive": {
        "status": "READY_AFTER_FOUNDATION_ARTIFACT_IS_PINNED",
        "requires_foundation_model": True,
        "description": "Full foundation-model fine-tuning with estimated isolated-atom energies.",
    },
    "lora_rank4": {
        "status": "READY_AFTER_FOUNDATION_ARTIFACT_IS_PINNED",
        "requires_foundation_model": True,
        "description": "Low-rank adaptation with rank 4 on the pinned foundation artifact.",
    },
    "layer_freeze5": {
        "status": "READY_AFTER_FOUNDATION_ARTIFACT_IS_PINNED",
        "requires_foundation_model": True,
        "description": "Foundation adaptation while freezing the first five layers.",
    },
    "multihead_pseudolabel_replay": {
        "status": "DEFERRED_UNTIL_PSEUDOLABEL_PROTOCOL_IS_AUDITED",
        "requires_foundation_model": True,
        "description": "Not eligible until replay provenance and forgetting metrics are frozen.",
    },
}


def require_known_arm(arm: str) -> Dict[str, Any]:
    """Return an arm definition or reject an unregistered comparison arm."""

    try:
        return {**FOUR_ARMS, **AUXILIARY_ARMS}[arm]
    except KeyError as exc:
        raise ValueError(f"Unknown RSI comparison arm: {arm}") from exc


def require_runnable_method(method: str) -> Dict[str, Any]:
    """Return a method only when its v1 execution status permits a run."""

    try:
        definition = METHOD_STATUS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown MACE method: {method}") from exc
    if definition["status"].startswith("DEFERRED"):
        raise ValueError(f"Method {method} is intentionally deferred: {definition['status']}")
    return definition

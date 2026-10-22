"""Resolution-V five-axis screening registry for the micro-MLIP program."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, Sequence, Tuple


FACTORS: Tuple[str, ...] = ("prompt", "data", "tool", "workflow", "model")
PDTW_FACTORS: Tuple[str, ...] = ("prompt", "data", "tool", "workflow")
GENERATOR = "model = -(prompt * data * tool * workflow)"
DEFINING_RELATION = "prompt * data * tool * workflow * model = -I"


def signed_level(value: int) -> str:
    """Map a two-level design sign to a human-readable treatment label."""

    if value not in {-1, 1}:
        raise ValueError("Factor signs must be -1 or +1")
    return "low" if value == -1 else "high"


def resolution_v_cells() -> List[Dict[str, Any]]:
    """Return the resolution-V fraction with the negative PDTWM defining sign.

    The sign does not affect design resolution.  It intentionally places both
    the all-low reference and all-high P/D/T/W condition in the scratch
    half-fraction, so the already runnable core multi-arm experiment can be
    compared without relabelling it as a blocked foundation treatment.
    """

    rows: List[Dict[str, Any]] = []
    for prompt, data, tool, workflow in product((-1, 1), repeat=4):
        model = -(prompt * data * tool * workflow)
        signs = {
            "prompt": prompt,
            "data": data,
            "tool": tool,
            "workflow": workflow,
            "model": model,
        }
        levels = {factor: signed_level(sign) for factor, sign in signs.items()}
        identifier = "".join(f"{factor[0].upper()}{1 if levels[factor] == 'high' else 0}" for factor in FACTORS)
        rows.append({"cell_id": identifier, "signs": signs, "levels": levels})
    return rows


def full_scratch_pdtw_cells() -> List[Dict[str, Any]]:
    """Return every Prompt/Data/Tool/Workflow combination at scratch level.

    A resolution-V five-axis half-fraction is useful when both model levels
    are admitted, but it intentionally contains only half of the P/D/T/W
    combinations at a fixed model level.  The research question here also
    requires a literal all-combinations test of the four controllable loop
    layers, so this companion block fixes ``model=low`` and enumerates the
    complete 2**4 design.
    """

    rows: List[Dict[str, Any]] = []
    for prompt, data, tool, workflow in product((-1, 1), repeat=4):
        signs = {
            "prompt": prompt,
            "data": data,
            "tool": tool,
            "workflow": workflow,
            "model": -1,
        }
        levels = {factor: signed_level(sign) for factor, sign in signs.items()}
        identifier = "".join(
            f"{factor[0].upper()}{1 if levels[factor] == 'high' else 0}" for factor in FACTORS
        )
        rows.append({"cell_id": identifier, "signs": signs, "levels": levels})
    return rows


def full_scratch_pdtw_registry(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the complete scratch P/D/T/W registry without model substitution."""

    factors = config["five_axes"]
    cells = []
    for cell in full_scratch_pdtw_cells():
        cells.append({**cell, "status": "READY", "blockers": []})
    return {
        "schema": "micro-mlip-pdtw-full-scratch-registry-v1",
        "benchmark_id": config["benchmark_id"],
        "design": "full 2^4 Prompt/Data/Tool/Workflow factorial at fixed scratch_micro model",
        "factors": {factor: factors[factor] for factor in FACTORS},
        "fixed_model": "low",
        "cells": cells,
        "ready_cell_count": len(cells),
        "blocked_cell_count": 0,
        "interpretation": (
            "Every P/D/T/W combination is runnable at scratch level. The foundation-model methods are a "
            "separate paired model-method extension, not silently represented by this fixed-model block."
        ),
    }


def apply_admission(
    cells: Sequence[Mapping[str, Any]], level_status: Mapping[str, Mapping[str, str]]
) -> List[Dict[str, Any]]:
    """Mark every factorial cell ready or blocked from its exact factor levels."""

    registered: List[Dict[str, Any]] = []
    for cell in cells:
        levels = cell["levels"]
        blockers = [
            f"{factor}:{levels[factor]}:{level_status[factor][levels[factor]]}"
            for factor in FACTORS
            if level_status[factor][levels[factor]] != "READY"
        ]
        registered.append(
            {
                **dict(cell),
                "status": "READY" if not blockers else "BLOCKED",
                "blockers": blockers,
            }
        )
    return registered


def registry(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a self-describing registry without claiming blocked cells were run."""

    factors = config["five_axes"]
    level_status = {
        "prompt": {"low": "READY", "high": "READY"},
        "data": {"low": "READY", "high": "READY"},
        "tool": {"low": "READY", "high": "READY"},
        "workflow": {"low": "READY", "high": "READY"},
        "model": {"low": "READY", "high": "BLOCKED_PENDING_PINNED_FOUNDATION_ARTIFACT"},
    }
    cells = apply_admission(resolution_v_cells(), level_status)
    return {
        "schema": "micro-mlip-five-axis-resolution-v-registry-v2",
        "benchmark_id": config["benchmark_id"],
        "generator": GENERATOR,
        "defining_relation": DEFINING_RELATION,
        "resolution": "V",
        "factors": {factor: factors[factor] for factor in FACTORS},
        "level_status": level_status,
        "cells": cells,
        "ready_cell_count": sum(cell["status"] == "READY" for cell in cells),
        "blocked_cell_count": sum(cell["status"] == "BLOCKED" for cell in cells),
        "interpretation": (
            "The registry is an experimental design and admission record. A blocked cell is unavailable, not a "
            "zero-effect observation."
        ),
    }

#!/usr/bin/env python3
"""Aggregate a completed g0→g1→g2 full P/D/T/W micro-MLIP program.

This is a strictly post-release reporting tool.  It replays the same pairing
and evaluator-boundary checks used by the per-campaign analyzer, then produces
one auditable table for every completed P/D/T/W cell, an orthogonal factorial
contrast table, a descriptive residual-feedback transition table, and the
separate frozen-policy toluene holdout results.  It never writes a policy,
launches a training, or contributes a metric to policy selection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from scripts.analyze_micro_mlip_five_round_campaign import (
    PAIR_COMPARISONS,
    check_group,
    episode_states,
    paired_effects,
)


AXES = ("prompt", "data", "tool", "workflow")
AXIS_SHORT_NAMES = {"prompt": "P", "data": "D", "tool": "T", "workflow": "W"}
AXIS_LONG_NAMES = {
    "prompt": "Prompt",
    "data": "Data",
    "tool": "Tool",
    "workflow": "Workflow",
}
OUTER_REMOVED_COMPARISON = "outer_generated_minus_removed"
FACTORIAL_RESPONSE_FIELDS = (
    "final_test_delta_force_component_mae_meV_per_A",
    "final_test_delta_energy_mae_meV_per_atom",
    "validation_normalized_auc_delta_force_component_mae_meV_per_A",
    "validation_normalized_auc_delta_energy_mae_meV_per_atom",
    "cumulative_training_seconds_delta",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric value for {context}")
    return float(value)


def average(values: Iterable[float], context: str) -> float:
    copied = list(values)
    if not copied:
        raise ValueError(f"Cannot average an empty collection: {context}")
    return float(statistics.fmean(copied))


def ready_registry_cells(registry_path: Path) -> Dict[str, Dict[str, str]]:
    """Read exactly the complete scratch P/D/T/W factorial block."""

    registry = load_json(registry_path)
    if registry.get("schema") != "micro-mlip-pdtw-full-scratch-registry-v1":
        raise ValueError("The post-release program requires the full scratch P/D/T/W registry")
    cells: Dict[str, Dict[str, str]] = {}
    for raw in registry.get("cells", []):
        if not isinstance(raw, Mapping) or raw.get("status") != "READY":
            continue
        cell_id = raw.get("cell_id")
        levels = raw.get("levels")
        if not isinstance(cell_id, str) or not isinstance(levels, Mapping):
            raise ValueError("Malformed ready factorial cell")
        normalized = {axis: levels.get(axis) for axis in AXES}
        if set(normalized) != set(AXES) or set(normalized.values()) - {"low", "high"}:
            raise ValueError(f"Cell {cell_id} has invalid P/D/T/W levels")
        if levels.get("model") != "low":
            raise ValueError(f"Cell {cell_id} is not the registered scratch model level")
        if cell_id in cells:
            raise ValueError(f"Duplicate factorial cell in registry: {cell_id}")
        cells[cell_id] = {**{axis: str(level) for axis, level in normalized.items()}, "model": "low"}
    expected = {f"P{p}D{d}T{t}W{w}M0" for p in (0, 1) for d in (0, 1) for t in (0, 1) for w in (0, 1)}
    if set(cells) != expected:
        raise ValueError("Registry does not contain exactly the 16 complete P/D/T/W cells")
    if registry.get("ready_cell_count") != len(cells):
        raise ValueError("Registry ready-cell count does not match its cell payload")
    return dict(sorted(cells.items()))


def completed_screen_cells(
    screen_root: Path,
    generation: str,
    expected_cells: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Validate a closed 16-cell screen before opening post-release receipts."""

    manifest_path = screen_root / "screen_manifest.json"
    manifest = load_json(manifest_path)
    if not str(manifest.get("status", "")).startswith("COMPLETED"):
        raise ValueError(f"Screen is not closed for analysis: {screen_root}")
    if manifest.get("generation") != generation:
        raise ValueError(f"Screen generation differs from expected {generation}: {screen_root}")
    entries = manifest.get("cells")
    if not isinstance(entries, list):
        raise ValueError(f"Screen has no cell ledger: {screen_root}")
    observed: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed screen cell ledger entry")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in observed:
            raise ValueError("Screen contains a duplicate or malformed cell ID")
        if entry.get("status") != "COMPLETED":
            raise ValueError(f"Screen cell did not complete: {generation}/{cell_id}")
        if entry.get("levels") != expected_cells.get(cell_id):
            raise ValueError(f"Screen cell levels differ from the registered factorial: {generation}/{cell_id}")
        observed[cell_id] = entry
    if set(observed) != set(expected_cells):
        raise ValueError(f"Screen cell inventory differs from the 16-cell registry: {generation}")
    expected_runs = len(expected_cells) * 2 * 1 * 4 * 5
    if manifest.get("expected_training_runs") != expected_runs:
        raise ValueError(f"Screen expected-training count is not the registered 640 nodes: {generation}")
    return manifest


def tagged_row(
    row: Mapping[str, Any],
    *,
    generation: str,
    cell_id: str,
    levels: Mapping[str, str],
) -> Dict[str, Any]:
    output = dict(row)
    output["generation"] = generation
    output["factorial_cell_id"] = cell_id
    for axis in AXES:
        output[f"{axis}_level"] = levels[axis]
    return output


def analyze_closed_campaign(
    campaign_root: Path,
    *,
    generation: str,
    cell_id: str,
    levels: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replay the independent-evaluator checks and tag one factorial campaign."""

    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "COMPLETED_PENDING_ANALYSIS":
        raise ValueError(f"Campaign is not closed for analysis: {campaign_root}")
    if manifest.get("factorial_cell_id") != cell_id or manifest.get("factorial_levels") != dict(levels):
        raise ValueError(f"Campaign factorial identity differs from its screen entry: {campaign_root}")
    states = episode_states(campaign_root)
    raw_rows: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for (molecule, seed), arm_states in sorted(states.items()):
        group_rows, group = check_group(campaign_root, molecule, seed, arm_states)
        raw_rows.extend(group_rows)
        groups.append(group)
    if not groups:
        raise ValueError(f"Campaign has no paired molecule/seed group: {campaign_root}")
    effects = paired_effects(groups, raw_rows)
    return (
        [tagged_row(row, generation=generation, cell_id=cell_id, levels=levels) for row in raw_rows],
        [tagged_row(row, generation=generation, cell_id=cell_id, levels=levels) for row in effects],
        [
            {
                "generation": generation,
                "factorial_cell_id": cell_id,
                "molecule": group["molecule"],
                "seed": group["seed"],
                "barrier_sha256": group["barrier_sha256"],
                "label_budgets": group["label_budgets"],
                **{f"{axis}_level": levels[axis] for axis in AXES},
            }
            for group in groups
        ],
    )


def factor_terms() -> List[Tuple[str, ...]]:
    return [term for order in range(1, len(AXES) + 1) for term in combinations(AXES, order)]


def factor_term_name(term: Sequence[str]) -> str:
    return "×".join(AXIS_SHORT_NAMES[axis] for axis in term)


def code_level(level: Any, context: str) -> int:
    if level == "low":
        return -1
    if level == "high":
        return 1
    raise ValueError(f"Invalid factor level in {context}: {level!r}")


def factorial_effect_rows(pair_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compute full-factorial high-minus-low contrasts per generation/molecule/seed.

    The response is always a paired *outer-generated minus removed* effect.
    Thus a negative error contrast means the high factor level made the outer
    policy more favorable relative to its information-removed control.  These
    are descriptive finite-design contrasts, not population-level causal CIs.
    """

    selected = [row for row in pair_rows if row.get("comparison") == OUTER_REMOVED_COMPARISON]
    if not selected:
        raise ValueError("No outer-generated versus removed paired effects were supplied")
    strata: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    for row in selected:
        generation = row.get("generation")
        molecule = row.get("molecule")
        seed = row.get("seed")
        if not isinstance(generation, str) or not isinstance(molecule, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Paired effect lacks generation/molecule/seed identity")
        strata.setdefault((generation, molecule, seed), []).append(row)

    output: List[Dict[str, Any]] = []
    expected_count = 2 ** len(AXES)
    for (generation, molecule, seed), rows in sorted(strata.items()):
        by_cell = {str(row.get("factorial_cell_id")): row for row in rows}
        if len(by_cell) != expected_count or len(rows) != expected_count:
            raise ValueError(f"Incomplete or duplicate factorial paired effects for {generation}/{molecule}/seed{seed}")
        for response in FACTORIAL_RESPONSE_FIELDS:
            for term in factor_terms():
                signed_total = 0.0
                for cell_id, row in by_cell.items():
                    sign = 1
                    for axis in term:
                        sign *= code_level(row.get(f"{axis}_level"), f"{cell_id}/{axis}")
                    signed_total += sign * numeric(row.get(response), f"{cell_id}/{response}")
                contrast = (2.0 / expected_count) * signed_total
                output.append(
                    {
                        "generation": generation,
                        "molecule": molecule,
                        "seed": seed,
                        "response": response,
                        "factor_term": factor_term_name(term),
                        "factor_axes": "+".join(term),
                        "interaction_order": len(term),
                        "high_minus_low_contrast": contrast,
                        "cell_count": expected_count,
                        "interpretation": (
                            "For error/time deltas defined as outer_generated minus removed, a negative contrast "
                            "means the high factor level is more favorable to the outer policy relative to removed."
                        ),
                    }
                )
    return output


def summarize_factorial_effects(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        key = (str(row["generation"]), str(row["response"]), str(row["factor_term"]))
        grouped.setdefault(key, []).append(numeric(row["high_minus_low_contrast"], str(key)))
    output = []
    for (generation, response, term), values in sorted(grouped.items()):
        output.append(
            {
                "generation": generation,
                "response": response,
                "factor_term": term,
                "stratum_count": len(values),
                "mean_high_minus_low_contrast": average(values, str((generation, response, term))),
                "min_high_minus_low_contrast": min(values),
                "max_high_minus_low_contrast": max(values),
                "uncertainty_note": "Descriptive across registered molecule/seed strata; no inferential interval is implied.",
            }
        )
    return output


def learning_curve_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, int, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["generation"]),
            str(row["factorial_cell_id"]),
            str(row["arm"]),
            int(row["round_index"]),
            int(row["label_budget"]),
        )
        grouped.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (generation, cell_id, arm, round_index, budget), members in sorted(grouped.items()):
        first = members[0]
        output.append(
            {
                "generation": generation,
                "factorial_cell_id": cell_id,
                "arm": arm,
                "round_index": round_index,
                "label_budget": budget,
                "trace_count": len(members),
                "mean_validation_force_component_mae_meV_per_A": average(
                    (numeric(row["validation_force_component_mae_meV_per_A"], "validation force") for row in members),
                    "validation force",
                ),
                "mean_validation_energy_mae_meV_per_atom": average(
                    (numeric(row["validation_energy_mae_meV_per_atom"], "validation energy") for row in members),
                    "validation energy",
                ),
                "mean_cumulative_training_duration_seconds": average(
                    (numeric(row["cumulative_training_duration_seconds"], "training duration") for row in members),
                    "training duration",
                ),
                **{f"{axis}_level": first[f"{axis}_level"] for axis in AXES},
            }
        )
    return output


def residual_action_transitions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Link a registered decision to the immediately following validation error.

    This table is intentionally descriptive: a decision is conditioned on the
    prior residual and does not independently identify a physical mechanism.
    """

    traces: Dict[Tuple[str, str, str, int, str], Dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["generation"]),
            str(row["factorial_cell_id"]),
            str(row["molecule"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        round_index = int(row["round_index"])
        trace = traces.setdefault(key, {})
        if round_index in trace:
            raise ValueError(f"Duplicate round in trace {key}/round{round_index}")
        trace[round_index] = row
    output: List[Dict[str, Any]] = []
    for key, trace in sorted(traces.items()):
        for round_index in sorted(trace):
            current = trace[round_index]
            selection = current.get("decision_selection")
            if selection is None:
                continue
            following = trace.get(round_index + 1)
            if following is None:
                raise ValueError(f"Decision has no following validation round: {key}/round{round_index}")
            current_force = numeric(current["validation_force_component_mae_meV_per_A"], "current validation force")
            following_force = numeric(following["validation_force_component_mae_meV_per_A"], "following validation force")
            residual = current.get("residual_class")
            output.append(
                {
                    "generation": key[0],
                    "factorial_cell_id": key[1],
                    "molecule": key[2],
                    "seed": key[3],
                    "arm": key[4],
                    "after_round": round_index,
                    "before_round": round_index + 1,
                    "residual_class": residual if residual is not None else "RAW_OR_STATIC_ROUTE",
                    "decision_selection": selection,
                    "decision_source": current.get("decision_source"),
                    "current_validation_force_component_mae_meV_per_A": current_force,
                    "next_validation_force_component_mae_meV_per_A": following_force,
                    "next_minus_current_validation_force_meV_per_A": following_force - current_force,
                    **{f"{axis}_level": current[f"{axis}_level"] for axis in AXES},
                    "interpretation": "Descriptive conditioned transition; it is not evidence that the residual label is a physical failure mechanism.",
                }
            )
    if not output:
        raise ValueError("No decision-to-next-round transitions were found")
    return output


def summarize_residual_action_transitions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["generation"]),
            str(row["arm"]),
            str(row["residual_class"]),
            str(row["decision_selection"]),
            str(row["tool_level"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (generation, arm, residual, selection, tool_level), members in sorted(grouped.items()):
        output.append(
            {
                "generation": generation,
                "arm": arm,
                "residual_class": residual,
                "decision_selection": selection,
                "tool_level": tool_level,
                "transition_count": len(members),
                "mean_next_minus_current_validation_force_meV_per_A": average(
                    (numeric(row["next_minus_current_validation_force_meV_per_A"], "transition") for row in members),
                    "transition",
                ),
                "interpretation": "Descriptive transition average only; do not infer a causal molecular mechanism.",
            }
        )
    return output


def generation_evolution_rows(pair_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compare g1 with g0 within a cell/molecule after both are independently closed."""

    selected = [row for row in pair_rows if row.get("comparison") == OUTER_REMOVED_COMPARISON]
    by_key: Dict[Tuple[str, str, str], Dict[str, List[Mapping[str, Any]]]] = {}
    for row in selected:
        key = (str(row["factorial_cell_id"]), str(row["molecule"]), str(row["comparison"]))
        by_key.setdefault(key, {}).setdefault(str(row["generation"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for (cell_id, molecule, comparison), generations in sorted(by_key.items()):
        if set(generations) != {"g0", "g1"}:
            raise ValueError(f"Missing a closed generation for {cell_id}/{molecule}")
        g0, g1 = generations["g0"], generations["g1"]
        template = g0[0]
        for response in FACTORIAL_RESPONSE_FIELDS:
            g0_value = average((numeric(row[response], f"g0/{response}") for row in g0), f"g0/{response}")
            g1_value = average((numeric(row[response], f"g1/{response}") for row in g1), f"g1/{response}")
            output.append(
                {
                    "factorial_cell_id": cell_id,
                    "molecule": molecule,
                    "comparison": comparison,
                    "response": response,
                    "g0_mean": g0_value,
                    "g1_mean": g1_value,
                    "g1_minus_g0": g1_value - g0_value,
                    "g0_trace_count": len(g0),
                    "g1_trace_count": len(g1),
                    **{f"{axis}_level": template[f"{axis}_level"] for axis in AXES},
                    "interpretation": "Cross-generation comparison of frozen policies on distinct registered development seeds; not a seed-paired confidence interval.",
                }
            )
    return output


def load_holdout(
    program_root: Path,
    registry_cells: Mapping[str, Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    promotion_path = program_root / "promotion.json"
    promotion = load_json(promotion_path)
    if promotion.get("schema") != "micro-mlip-pdtw-promotion-selection-v1" or promotion.get("status") != "FROZEN_BEFORE_POLICY_HOLDOUT":
        raise ValueError("Program promotion receipt is not frozen for a policy holdout")
    selected = promotion.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Program promotion receipt has no selected policy")
    selected_cell = selected.get("factorial_cell_id")
    selected_generation = selected.get("generation")
    if not isinstance(selected_cell, str) or selected_cell not in registry_cells or selected_generation not in {"g0", "g1"}:
        raise ValueError("Program promotion references an unavailable P/D/T/W policy")
    levels = registry_cells[selected_cell]
    holdout_root = program_root / "g2_toluene"
    holdout_manifest_path = holdout_root / "holdout_manifest.json"
    holdout_manifest = load_json(holdout_manifest_path)
    if holdout_manifest.get("status") != "COMPLETED_PENDING_ANALYSIS":
        raise ValueError("Toluene policy holdout is not closed for analysis")
    if holdout_manifest.get("selected_factorial_cell_id") != selected_cell or holdout_manifest.get("selected_generation") != selected_generation:
        raise ValueError("Toluene holdout selection differs from the frozen promotion receipt")
    raw_rows, effects, groups = analyze_closed_campaign(
        holdout_root / "campaign",
        generation="g2_policy_holdout",
        cell_id=selected_cell,
        levels=levels,
    )
    metadata = {
        "promotion_path": str(promotion_path),
        "promotion_sha256": sha256(promotion_path),
        "holdout_manifest_path": str(holdout_manifest_path),
        "holdout_manifest_sha256": sha256(holdout_manifest_path),
        "selected_generation": selected_generation,
        "selected_factorial_cell_id": selected_cell,
        "selected_factorial_levels": levels,
        "paired_group_count": len(groups),
    }
    return raw_rows, effects, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--five-round-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Post-release analysis outputs are immutable; choose a new output directory")
    cells = ready_registry_cells(args.registry)
    all_round_rows: List[Dict[str, Any]] = []
    all_effects: List[Dict[str, Any]] = []
    group_ledger: List[Dict[str, Any]] = []
    screen_sources: Dict[str, Dict[str, Any]] = {}
    for generation in ("g0", "g1"):
        screen_root = args.program_root / generation
        manifest = completed_screen_cells(screen_root, generation, cells)
        screen_sources[generation] = {
            "path": str(screen_root / "screen_manifest.json"),
            "sha256": sha256(screen_root / "screen_manifest.json"),
            "screen_id": manifest.get("screen_id"),
            "expected_training_runs": manifest.get("expected_training_runs"),
        }
        for cell_id, levels in cells.items():
            rows, effects, groups = analyze_closed_campaign(
                screen_root / "cells" / cell_id / "campaign",
                generation=generation,
                cell_id=cell_id,
                levels=levels,
            )
            all_round_rows.extend(rows)
            all_effects.extend(effects)
            group_ledger.extend(groups)
    holdout_rows, holdout_effects, holdout_metadata = load_holdout(args.program_root, cells)
    factorial_rows = factorial_effect_rows(all_effects)
    factorial_summary = summarize_factorial_effects(factorial_rows)
    curve_rows = learning_curve_summary(all_round_rows)
    transition_rows = residual_action_transitions(all_round_rows)
    transition_summary = summarize_residual_action_transitions(transition_rows)
    evolution_rows = generation_evolution_rows(all_effects)

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "round_rows.csv", all_round_rows)
    write_csv(args.output_dir / "paired_effects.csv", all_effects)
    write_csv(args.output_dir / "paired_groups.csv", group_ledger)
    write_csv(args.output_dir / "learning_curve_summary.csv", curve_rows)
    write_csv(args.output_dir / "factorial_effects.csv", factorial_rows)
    write_csv(args.output_dir / "factorial_effect_summary.csv", factorial_summary)
    write_csv(args.output_dir / "residual_action_transitions.csv", transition_rows)
    write_csv(args.output_dir / "residual_action_transition_summary.csv", transition_summary)
    write_csv(args.output_dir / "generation_evolution.csv", evolution_rows)
    write_csv(args.output_dir / "holdout_round_rows.csv", holdout_rows)
    write_csv(args.output_dir / "holdout_paired_effects.csv", holdout_effects)
    analysis = {
        "schema": "micro-mlip-full-pdtw-postrelease-analysis-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "program_root": str(args.program_root),
        "source_sha256": {
            "registry": sha256(args.registry),
            "five_round_config": sha256(args.five_round_config),
            "screens": screen_sources,
        },
        "design": {
            "factorial_axes": list(AXES),
            "factorial_cells": len(cells),
            "development_generations": ["g0", "g1"],
            "development_round_rows": len(all_round_rows),
            "development_paired_effect_rows": len(all_effects),
            "factorial_contrast_rows": len(factorial_rows),
            "holdout": holdout_metadata,
        },
        "boundaries": {
            "selection": "The g2 policy was frozen from development validation-only evidence before toluene execution.",
            "test": "All analyzed independent metrics passed the four-arm release-barrier checks replayed here.",
            "mechanism": "Residual classes and action transitions are operational diagnostics, not demonstrated molecular failure mechanisms.",
            "factorial": "P/D/T/W contrasts hold the scratch method fixed; they do not estimate the separately blocked model-method axis.",
        },
        "tables": [
            "round_rows.csv",
            "paired_effects.csv",
            "paired_groups.csv",
            "learning_curve_summary.csv",
            "factorial_effects.csv",
            "factorial_effect_summary.csv",
            "residual_action_transitions.csv",
            "residual_action_transition_summary.csv",
            "generation_evolution.csv",
            "holdout_round_rows.csv",
            "holdout_paired_effects.csv",
        ],
    }
    write_json(args.output_dir / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "factorial_cells": len(cells),
                "development_round_rows": len(all_round_rows),
                "factorial_contrasts": len(factorial_rows),
                "holdout_selected_cell": holdout_metadata["selected_factorial_cell_id"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify and summarize a completed five-round four-arm MLIP campaign.

This is a post-release analysis tool only.  It refuses a campaign until every
matched arm has closed its independent evaluator receipt, then emits raw
round-level data, pairing checks, and publication-ready numerical summaries.
It never participates in a policy decision or alters campaign artifacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256
from rsi.micro_mlip.five_round import ROUND_COUNT
from scripts.run_micro_mlip_five_round_campaign import ARMS


VALIDATION_METRICS = (
    "energy_mae_meV_per_atom",
    "energy_rmse_meV_per_atom",
    "force_component_mae_meV_per_A",
    "force_component_rmse_meV_per_A",
)
FINAL_METRICS = VALIDATION_METRICS
PAIR_COMPARISONS = (
    ("outer_generated", "removed"),
    ("outer_generated", "fixed"),
    ("manual_control", "fixed"),
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric(metrics: Mapping[str, Any], name: str, context: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {name} in {context}")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"Negative {name} in {context}")
    return value


def mean(values: Iterable[float]) -> float:
    copied = list(values)
    if not copied:
        raise ValueError("Cannot calculate a mean of no values")
    return sum(copied) / len(copied)


def normalized_trapezoid(label_budgets: Sequence[int], values: Sequence[float]) -> float:
    """Return label-normalized area under a five-point learning curve."""

    if len(label_budgets) != len(values) or len(values) != ROUND_COUNT:
        raise ValueError("A learning curve must have exactly five aligned values")
    if any(next_budget <= budget for budget, next_budget in zip(label_budgets, label_budgets[1:])):
        raise ValueError("Label budgets must strictly increase")
    area = sum(
        (next_budget - budget) * (value + next_value) / 2.0
        for budget, next_budget, value, next_value in zip(
            label_budgets, label_budgets[1:], values, values[1:]
        )
    )
    return area / (label_budgets[-1] - label_budgets[0])


def episode_states(campaign_root: Path) -> Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]:
    states: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    paths = sorted((campaign_root / "episodes").glob("*/state.json"))
    if not paths:
        raise ValueError("Campaign has no episode state receipts")
    for path in paths:
        state = load_json(path)
        molecule = state.get("molecule")
        seed = state.get("seed")
        arm = state.get("arm")
        if not isinstance(molecule, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Malformed episode identity in {path}")
        if arm not in ARMS:
            raise ValueError(f"Unknown arm in {path}: {arm}")
        group = states.setdefault((molecule, seed), {})
        if arm in group:
            raise ValueError(f"Duplicate episode state for {molecule}/seed{seed}/{arm}")
        group[arm] = state
    return states


def decision_for_round(state: Mapping[str, Any], round_index: int) -> Mapping[str, Any] | None:
    if round_index == 1:
        return None
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Episode state has no decisions list")
    matches = [row for row in decisions if row.get("before_round") == round_index]
    if len(matches) != 1:
        raise ValueError(f"Episode must have exactly one decision before round {round_index}")
    return matches[0]


def check_group(
    campaign_root: Path,
    molecule: str,
    seed: int,
    states: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return verified raw round rows and final metrics for one paired group."""

    if set(states) != set(ARMS):
        raise ValueError(f"Incomplete arm set for {molecule}/seed{seed}: {sorted(states)}")
    barrier_path = campaign_root / "release_barriers" / f"{molecule}__seed{seed}.json"
    barrier = load_json(barrier_path)
    if barrier.get("status") != "ALL_FOUR_ARMS_FINAL_TRAINING_VALIDATION_CLOSED":
        raise ValueError(f"Independent release barrier is not closed for {molecule}/seed{seed}")
    eligible = barrier.get("eligible_run_ids")
    if not isinstance(eligible, list) or len(eligible) != len(ARMS):
        raise ValueError(f"Release barrier lacks exactly four eligible runs for {molecule}/seed{seed}")

    rows: List[Dict[str, Any]] = []
    final_by_arm: Dict[str, Dict[str, float]] = {}
    shared_budgets: List[int] | None = None
    for arm in ARMS:
        state = states[arm]
        if state.get("status") != "COMPLETED":
            raise ValueError(f"Episode did not close independently: {molecule}/seed{seed}/{arm}")
        rounds = state.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != ROUND_COUNT:
            raise ValueError(f"Episode has an incomplete five-round trace: {molecule}/seed{seed}/{arm}")
        cumulative_train_seconds = 0.0
        arm_budgets: List[int] = []
        for expected_round, round_row in enumerate(rounds, start=1):
            if round_row.get("round_index") != expected_round or round_row.get("status") != "COMPLETED":
                raise ValueError(f"Malformed closed round {expected_round}: {molecule}/seed{seed}/{arm}")
            run_dir = Path(str(round_row.get("run_directory")))
            run = load_json(run_dir / "run.json")
            result = load_json(run_dir / "result.json")
            validation = load_json(run_dir / "validation_evaluation.json")
            if run.get("run_id") != round_row.get("run_id"):
                raise ValueError(f"Run ID mismatch in {run_dir}")
            if run.get("molecule") != molecule or run.get("arm") != arm or run.get("seed") != seed:
                raise ValueError(f"Run identity mismatch in {run_dir}")
            if run.get("round_index") != expected_round:
                raise ValueError(f"Run round mismatch in {run_dir}")
            if run.get("inputs", {}).get("test_policy") != "NOT_PASSED_TO_TRAINING_COMMAND":
                raise ValueError(f"Training receipt did not enforce test barrier in {run_dir}")
            if validation.get("status") != "COMPLETED" or validation.get("test_access") != "NOT_OPENED_OR_READ":
                raise ValueError(f"Validation receipt violates test boundary in {run_dir}")
            if validation.get("run_id") != run.get("run_id"):
                raise ValueError(f"Validation receipt/run mismatch in {run_dir}")
            if expected_round < ROUND_COUNT:
                if result.get("status") != "TRAINING_VALIDATED_PENDING_INDEPENDENT_TEST_EVALUATION":
                    raise ValueError(f"Early round has an invalid terminal status in {run_dir}")
                if result.get("independent_test_evaluation") not in (None, "NOT_STARTED"):
                    raise ValueError(f"Test evaluation appeared before round 5 in {run_dir}")
            elif result.get("status") != "COMPLETED_INDEPENDENT_TEST_EVALUATION":
                raise ValueError(f"Final round did not close independent evaluation in {run_dir}")
            metrics = validation.get("validation_metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Missing validation metrics in {run_dir}")
            budget = run.get("label_budget")
            if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                raise ValueError(f"Invalid label budget in {run_dir}")
            arm_budgets.append(budget)
            training = round_row.get("training")
            if not isinstance(training, Mapping):
                raise ValueError(f"Missing controller training receipt in {run_dir}")
            if training.get("exit_code") != 0:
                raise ValueError(f"Training controller reports failure in {run_dir}")
            duration = numeric(training, "duration_seconds", f"training controller {run_dir}")
            cumulative_train_seconds += duration
            decision = decision_for_round(state, expected_round)
            decision_payload = decision.get("decision") if decision is not None else None
            signature = decision.get("validation_context") if decision is not None else None
            row: Dict[str, Any] = {
                "molecule": molecule,
                "seed": seed,
                "arm": arm,
                "round_index": expected_round,
                "run_id": run["run_id"],
                "label_budget": budget,
                "selection_prefix": run.get("selection_prefix"),
                "selection_history": run.get("selection_history"),
                "training_duration_seconds": duration,
                "cumulative_training_duration_seconds": cumulative_train_seconds,
                "validation_duration_seconds": numeric(
                    validation, "duration_seconds", f"validation receipt {run_dir}"
                ),
                "validation_receipt_sha256": sha256(run_dir / "validation_evaluation.json"),
                "decision_source": decision_payload.get("decision_source") if isinstance(decision_payload, Mapping) else None,
                "decision_selection": decision_payload.get("selection") if isinstance(decision_payload, Mapping) else None,
                "residual_class": signature.get("residual_class") if isinstance(signature, Mapping) else None,
            }
            for metric in VALIDATION_METRICS:
                row[f"validation_{metric}"] = numeric(metrics, metric, f"validation receipt {run_dir}")
            rows.append(row)

        if shared_budgets is None:
            shared_budgets = arm_budgets
        elif arm_budgets != shared_budgets:
            raise ValueError(f"Arms do not have matched label budgets for {molecule}/seed{seed}")
        final_run = Path(str(rounds[-1]["run_directory"]))
        independent = load_json(final_run / "independent_evaluation.json")
        if independent.get("status") != "COMPLETED" or independent.get("run_id") not in eligible:
            raise ValueError(f"Invalid independent evaluator receipt for {molecule}/seed{seed}/{arm}")
        if independent.get("release_manifest_sha256") != sha256(barrier_path):
            raise ValueError(f"Independent evaluator used a different release barrier for {molecule}/seed{seed}/{arm}")
        test_metrics = independent.get("test_metrics")
        if not isinstance(test_metrics, Mapping):
            raise ValueError(f"Independent evaluator has no test metrics for {molecule}/seed{seed}/{arm}")
        final_by_arm[arm] = {metric: numeric(test_metrics, metric, str(final_run)) for metric in FINAL_METRICS}

    if shared_budgets is None:
        raise AssertionError("No arm budgets were collected")
    return rows, {
        "molecule": molecule,
        "seed": seed,
        "label_budgets": shared_budgets,
        "barrier_sha256": sha256(barrier_path),
        "final_test_metrics_by_arm": final_by_arm,
    }


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["molecule"]), int(row["seed"]), str(row["arm"]))
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["round_index"]))
    return grouped


def paired_effects(groups: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    trace = group_rows(rows)
    effects: List[Dict[str, Any]] = []
    for group in groups:
        molecule = str(group["molecule"])
        seed = int(group["seed"])
        metrics_by_arm = group["final_test_metrics_by_arm"]
        for numerator, denominator in PAIR_COMPARISONS:
            row: Dict[str, Any] = {
                "molecule": molecule,
                "seed": seed,
                "comparison": f"{numerator}_minus_{denominator}",
                "numerator": numerator,
                "denominator": denominator,
            }
            for metric in FINAL_METRICS:
                row[f"final_test_delta_{metric}"] = (
                    metrics_by_arm[numerator][metric] - metrics_by_arm[denominator][metric]
                )
            for metric in ("force_component_mae_meV_per_A", "energy_mae_meV_per_atom"):
                numerator_curve = [
                    float(item[f"validation_{metric}"])
                    for item in trace[(molecule, seed, numerator)]
                ]
                denominator_curve = [
                    float(item[f"validation_{metric}"])
                    for item in trace[(molecule, seed, denominator)]
                ]
                budgets = [int(item["label_budget"]) for item in trace[(molecule, seed, numerator)]]
                row[f"validation_normalized_auc_delta_{metric}"] = (
                    normalized_trapezoid(budgets, numerator_curve)
                    - normalized_trapezoid(budgets, denominator_curve)
                )
            row["cumulative_training_seconds_delta"] = (
                float(trace[(molecule, seed, numerator)][-1]["cumulative_training_duration_seconds"])
                - float(trace[(molecule, seed, denominator)][-1]["cumulative_training_duration_seconds"])
            )
            effects.append(row)
    return effects


def summary_by_arm(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for arm in ARMS:
        for round_index in range(1, ROUND_COUNT + 1):
            subset = [row for row in rows if row["arm"] == arm and row["round_index"] == round_index]
            if not subset:
                raise ValueError(f"No rows for {arm}/round{round_index}")
            entry: Dict[str, Any] = {
                "arm": arm,
                "round_index": round_index,
                "paired_trace_count": len(subset),
                "label_budget": int(subset[0]["label_budget"]),
                "mean_cumulative_training_duration_seconds": mean(
                    float(row["cumulative_training_duration_seconds"]) for row in subset
                ),
            }
            for metric in VALIDATION_METRICS:
                entry[f"mean_validation_{metric}"] = mean(
                    float(row[f"validation_{metric}"]) for row in subset
                )
            result.append(entry)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--factorial-cell-id")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {args.output_dir}")
    manifest_path = args.campaign_root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "COMPLETED_PENDING_ANALYSIS":
        raise ValueError("Campaign is not closed for post-release analysis")

    states = episode_states(args.campaign_root)
    raw_rows: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for (molecule, seed), arm_states in sorted(states.items()):
        group_rows_payload, group = check_group(args.campaign_root, molecule, seed, arm_states)
        raw_rows.extend(group_rows_payload)
        groups.append(group)
    expected_rows = len(groups) * len(ARMS) * ROUND_COUNT
    if len(raw_rows) != expected_rows:
        raise AssertionError("Raw analysis row count does not match the paired campaign structure")
    effects = paired_effects(groups, raw_rows)
    summary = summary_by_arm(raw_rows)
    output = {
        "schema": "micro-mlip-five-round-postrelease-analysis-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "campaign_root": str(args.campaign_root),
        "campaign_manifest_sha256": sha256(manifest_path),
        "generation": args.generation,
        "factorial_cell_id": args.factorial_cell_id,
        "pairing": {
            "status": "PASSED",
            "unit": "molecule x seed x initial-label set",
            "paired_group_count": len(groups),
            "raw_round_row_count": len(raw_rows),
            "independent_test_opening": "Verified after the four-arm round-5 validation barrier for every group.",
        },
        "groups": groups,
        "round_rows": raw_rows,
        "paired_effects": effects,
        "learning_curve_summary": summary,
        "interpretation": {
            "delta_sign": "Negative error/time deltas favor the numerator arm; raw paired rows must accompany every aggregate.",
            "causal_boundary": "This post-release analysis evaluates registered policies under fixed budgets. Residual classes remain operational diagnostics, not demonstrated physical mechanisms.",
            "figure_inputs": ["round_rows.csv", "paired_effects.csv", "learning_curve_summary.csv"],
        },
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "analysis.json", output)
    write_csv(args.output_dir / "round_rows.csv", raw_rows)
    write_csv(args.output_dir / "paired_effects.csv", effects)
    write_csv(args.output_dir / "learning_curve_summary.csv", summary)
    print(
        json.dumps(
            {
                "status": output["status"],
                "paired_groups": len(groups),
                "round_rows": len(raw_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

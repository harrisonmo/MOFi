#!/usr/bin/env python3
"""Analyze the matched model-method extension after every holdout closes.

This is a release-only analyzer.  It replays the four-arm evaluator barriers
for the pre-frozen scratch reference and for every admitted foundation method,
then records method-minus-scratch differences without selecting a method or
opening any additional data.  Its conclusions are intentionally limited to
the one frozen toluene policy holdout.
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
from scripts.analyze_micro_mlip_five_round_campaign import (
    PAIR_COMPARISONS,
    check_group,
    episode_states,
    paired_effects,
)
from scripts.analyze_micro_mlip_full_pdtw_program import FACTORIAL_RESPONSE_FIELDS


FOUNDATION_METHODS = (
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)
PRIMARY_COMPARISON = "outer_generated_minus_removed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
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
    return sum(copied) / len(copied)


def analyze_closed_campaign(
    campaign_root: Path,
    *,
    generation: str,
    cell_id: str,
    levels: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replay independent-evaluator receipts for a single closed holdout."""

    states = episode_states(campaign_root)
    all_rows: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for (molecule, seed), arm_states in sorted(states.items()):
        rows, group = check_group(campaign_root, molecule, seed, arm_states)
        for row in rows:
            tagged = dict(row)
            tagged.update(
                {
                    "generation": generation,
                    "factorial_cell_id": cell_id,
                    **{f"{axis}_level": level for axis, level in levels.items()},
                }
            )
            all_rows.append(tagged)
        groups.append(group)
    if not groups:
        raise ValueError(f"No paired groups in closed campaign: {campaign_root}")
    effects = paired_effects(groups, all_rows)
    for row in effects:
        row.update(
            {
                "generation": generation,
                "factorial_cell_id": cell_id,
                **{f"{axis}_level": level for axis, level in levels.items()},
            }
        )
    return all_rows, effects, groups


def required_holdout_manifest(path: Path, *, expected_method: str | None = None) -> Dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != "micro-mlip-five-round-policy-holdout-v1":
        raise ValueError(f"Unexpected holdout schema: {path}")
    if manifest.get("status") != "COMPLETED_PENDING_ANALYSIS":
        raise ValueError(f"Holdout is not closed for analysis: {path}")
    if manifest.get("molecule") != "toluene":
        raise ValueError(f"Holdout is not the registered toluene benchmark: {path}")
    if expected_method is not None and manifest.get("method") != expected_method:
        raise ValueError(f"Holdout method does not match requested method {expected_method}: {path}")
    return manifest


def effect_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    output: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        comparison = row.get("comparison")
        if not isinstance(comparison, str) or comparison in output:
            raise ValueError("Paired effects contain a malformed or duplicate comparison")
        output[comparison] = row
    expected = {f"{numerator}_minus_{denominator}" for numerator, denominator in PAIR_COMPARISONS}
    if set(output) != expected:
        raise ValueError("Paired effects do not contain exactly the registered four-arm comparisons")
    return output


def group_lookup(groups: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    output: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for group in groups:
        molecule = group.get("molecule")
        seed = group.get("seed")
        if not isinstance(molecule, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Malformed paired group identity")
        key = (molecule, seed)
        if key in output:
            raise ValueError("Duplicate paired group identity")
        output[key] = group
    return output


def method_specs(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if separator != "=" or method not in FOUNDATION_METHODS or not raw_path:
            raise ValueError("Each --method-holdout must be METHOD=/absolute/or/relative/path")
        if method in output:
            raise ValueError(f"Duplicate method holdout: {method}")
        output[method] = Path(raw_path)
    if set(output) != set(FOUNDATION_METHODS):
        raise ValueError("Exactly the four registered foundation-method holdouts are required")
    return output


def method_forgetting_rows(admission: Mapping[str, Any]) -> List[Dict[str, Any]]:
    methods = admission.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Admission receipt lacks method receipts")
    rows: List[Dict[str, Any]] = []
    for method in FOUNDATION_METHODS:
        receipt = methods.get(method)
        if not isinstance(receipt, Mapping) or receipt.get("status") != "ADMITTED":
            raise ValueError(f"Admission receipt does not admit {method}")
        row: Dict[str, Any] = {"method": method, "admission_status": "ADMITTED"}
        forgetting = receipt.get("forgetting_metric")
        if method == "multihead_pseudolabel_replay":
            if not isinstance(forgetting, Mapping):
                raise ValueError("Multihead admission lacks its replay-domain forgetting metric")
            delta = forgetting.get("final_minus_initial")
            if not isinstance(delta, Mapping):
                raise ValueError("Multihead forgetting metric has no final-minus-initial values")
            for key in ("mae_e_per_atom_final_minus_initial", "rmse_e_per_atom_final_minus_initial", "mae_f_final_minus_initial", "rmse_f_final_minus_initial"):
                row[key] = numeric(delta.get(key), f"multihead forgetting {key}")
            row["interpretation"] = str(forgetting.get("interpretation", ""))
        else:
            row["interpretation"] = "No replay-domain forgetting measurement is applicable to this method."
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-holdout", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--method-admission", type=Path, required=True)
    parser.add_argument("--method-holdout", action="append", default=[], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Model-method analysis output is immutable; choose a new versioned path")

    promotion = load_json(args.promotion)
    if promotion.get("schema") != "micro-mlip-pdtw-promotion-selection-v1" or promotion.get("status") != "FROZEN_BEFORE_POLICY_HOLDOUT":
        raise ValueError("The P/D/T/W promotion receipt is not frozen for a policy holdout")
    selected = promotion.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Promotion receipt lacks a selected policy")
    cell_id = selected.get("factorial_cell_id")
    levels = selected.get("factorial_levels")
    if not isinstance(cell_id, str) or not isinstance(levels, Mapping):
        raise ValueError("Promotion selection has no usable factorial identity")
    expected_levels = {axis: levels.get(axis) for axis in ("prompt", "data", "tool", "workflow")}
    if set(expected_levels.values()) - {"low", "high"}:
        raise ValueError("Promotion selection has invalid P/D/T/W levels")

    scratch_manifest_path = args.scratch_holdout / "holdout_manifest.json"
    scratch_manifest = required_holdout_manifest(scratch_manifest_path, expected_method="scratch_micro")
    if scratch_manifest.get("promotion_sha256") != sha256(args.promotion):
        raise ValueError("Scratch holdout was not bound to the supplied frozen promotion receipt")
    if scratch_manifest.get("selected_factorial_cell_id") != cell_id:
        raise ValueError("Scratch holdout cell differs from the promotion selection")
    scratch_rows, scratch_effects, scratch_groups = analyze_closed_campaign(
        args.scratch_holdout / "campaign",
        generation="scratch_reference",
        cell_id=cell_id,
        levels=expected_levels,
    )
    scratch_effect_by_comparison = effect_lookup(scratch_effects)
    scratch_groups_by_key = group_lookup(scratch_groups)

    admission = load_json(args.method_admission)
    if admission.get("schema") != "micro-mlip-foundation-method-admission-v1" or admission.get("status") != "ADMITTED":
        raise ValueError("Foundation-method admission is not completed")
    if not isinstance(admission.get("admitted_config_sha256"), str):
        raise ValueError("Foundation-method admission does not bind an admitted execution configuration")
    method_paths = method_specs(args.method_holdout)

    all_round_rows = list(scratch_rows)
    all_effect_rows = []
    method_effect_vs_scratch: List[Dict[str, Any]] = []
    method_arm_vs_scratch: List[Dict[str, Any]] = []
    method_groups: List[Dict[str, Any]] = []
    for method in FOUNDATION_METHODS:
        holdout_root = method_paths[method]
        manifest_path = holdout_root / "holdout_manifest.json"
        manifest = required_holdout_manifest(manifest_path, expected_method=method)
        if manifest.get("promotion_sha256") != sha256(args.promotion):
            raise ValueError(f"{method} holdout was not bound to the supplied promotion receipt")
        if manifest.get("source_policy_manifest_sha256") != selected.get("policy_manifest_sha256"):
            raise ValueError(f"{method} source policy does not match the frozen promoted policy")
        if manifest.get("admission_sha256") != sha256(args.method_admission):
            raise ValueError(f"{method} holdout does not bind the supplied admission receipt")
        if manifest.get("execution_config_sha256") != admission.get("admitted_config_sha256"):
            raise ValueError(f"{method} holdout does not use the admitted execution configuration")
        if manifest.get("selected_factorial_cell_id") != cell_id:
            raise ValueError(f"{method} holdout cell differs from the promotion selection")
        rows, effects, groups = analyze_closed_campaign(
            holdout_root / "campaign",
            generation=method,
            cell_id=cell_id,
            levels=expected_levels,
        )
        all_round_rows.extend(rows)
        for row in effects:
            tagged = dict(row)
            tagged["method"] = method
            all_effect_rows.append(tagged)
        effects_by_comparison = effect_lookup(effects)
        groups_by_key = group_lookup(groups)
        if set(groups_by_key) != set(scratch_groups_by_key):
            raise ValueError(f"{method} does not have the same toluene/seed pairing as scratch")
        for key, scratch_group in scratch_groups_by_key.items():
            method_group = groups_by_key[key]
            if method_group.get("label_budgets") != scratch_group.get("label_budgets"):
                raise ValueError(f"{method} has unmatched five-round label budgets for {key}")
            method_groups.append(
                {
                    "method": method,
                    "molecule": key[0],
                    "seed": key[1],
                    "label_budgets": json.dumps(method_group.get("label_budgets")),
                    "scratch_barrier_sha256": scratch_group.get("barrier_sha256"),
                    "method_barrier_sha256": method_group.get("barrier_sha256"),
                }
            )
            scratch_metrics = scratch_group.get("final_test_metrics_by_arm")
            method_metrics = method_group.get("final_test_metrics_by_arm")
            if not isinstance(scratch_metrics, Mapping) or not isinstance(method_metrics, Mapping):
                raise ValueError("A paired group lacks final independent-test metrics")
            if set(scratch_metrics) != set(method_metrics):
                raise ValueError(f"{method} has a different arm inventory from scratch")
            for arm in sorted(scratch_metrics):
                scratch_arm = scratch_metrics[arm]
                method_arm = method_metrics[arm]
                if not isinstance(scratch_arm, Mapping) or not isinstance(method_arm, Mapping):
                    raise ValueError("Arm metrics are malformed")
                for metric in ("force_component_mae_meV_per_A", "energy_mae_meV_per_atom"):
                    scratch_value = numeric(scratch_arm.get(metric), f"scratch {arm}/{metric}")
                    method_value = numeric(method_arm.get(metric), f"{method} {arm}/{metric}")
                    method_arm_vs_scratch.append(
                        {
                            "method": method,
                            "molecule": key[0],
                            "seed": key[1],
                            "arm": arm,
                            "metric": metric,
                            "method_value": method_value,
                            "scratch_value": scratch_value,
                            "method_minus_scratch": method_value - scratch_value,
                            "interpretation": "Negative values favor the method relative to the matched scratch reference for this arm; this is a one-holdout descriptive comparison.",
                        }
                    )
        for comparison, method_effect in effects_by_comparison.items():
            scratch_effect = scratch_effect_by_comparison[comparison]
            for response in FACTORIAL_RESPONSE_FIELDS:
                method_value = numeric(method_effect.get(response), f"{method}/{comparison}/{response}")
                scratch_value = numeric(scratch_effect.get(response), f"scratch/{comparison}/{response}")
                method_effect_vs_scratch.append(
                    {
                        "method": method,
                        "comparison": comparison,
                        "response": response,
                        "method_value": method_value,
                        "scratch_value": scratch_value,
                        "method_minus_scratch": method_value - scratch_value,
                        "interpretation": "For error/time deltas defined within each method as numerator minus denominator, negative method-minus-scratch means the method made that paired arm effect more favorable than scratch on this frozen toluene holdout.",
                    }
                )

    all_effect_rows.extend({**row, "method": "scratch_micro"} for row in scratch_effects)
    forgetting_rows = method_forgetting_rows(admission)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "method_round_rows.csv", all_round_rows)
    write_csv(args.output_dir / "method_paired_effects.csv", all_effect_rows)
    write_csv(args.output_dir / "method_effect_vs_scratch.csv", method_effect_vs_scratch)
    write_csv(args.output_dir / "method_arm_vs_scratch.csv", method_arm_vs_scratch)
    write_csv(args.output_dir / "method_groups.csv", method_groups)
    write_csv(args.output_dir / "method_forgetting.csv", forgetting_rows)
    analysis = {
        "schema": "micro-mlip-model-method-extension-analysis-v1",
        "status": "COMPLETED",
        "created_at": utc_now(),
        "sources": {
            "promotion": {"path": str(args.promotion), "sha256": sha256(args.promotion)},
            "scratch_holdout_manifest": {"path": str(scratch_manifest_path), "sha256": sha256(scratch_manifest_path)},
            "method_admission": {"path": str(args.method_admission), "sha256": sha256(args.method_admission)},
            "method_holdout_manifests": {
                method: {"path": str(path / "holdout_manifest.json"), "sha256": sha256(path / "holdout_manifest.json")}
                for method, path in method_paths.items()
            },
        },
        "design": {
            "reference_method": "scratch_micro",
            "foundation_methods": list(FOUNDATION_METHODS),
            "molecule": "toluene",
            "selected_factorial_cell_id": cell_id,
            "selected_factorial_levels": expected_levels,
            "comparison": "All methods use the same frozen four-arm acquisition policies, five label budgets, toluene seed, split, and independent evaluator.",
        },
        "boundaries": {
            "selection": "No model method was selected from the toluene result; the P/D/T/W policy was frozen before all method holdouts.",
            "scope": "This is a compact matched recipe extension on one toluene policy holdout, not a second independent molecular generalization.",
            "axes": "Prompt/Data/Tool/Workflow are held fixed here. Method differences must not be attributed to those loop axes.",
            "forgetting": "The multihead replay-domain forgetting delta is measured only from the admitted pretraining-head validation trace.",
        },
        "tables": [
            "method_round_rows.csv",
            "method_paired_effects.csv",
            "method_effect_vs_scratch.csv",
            "method_arm_vs_scratch.csv",
            "method_groups.csv",
            "method_forgetting.csv",
        ],
    }
    write_json(args.output_dir / "analysis.json", analysis)
    print(json.dumps({"status": analysis["status"], "output_dir": str(args.output_dir), "method_count": len(FOUNDATION_METHODS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

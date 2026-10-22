#!/usr/bin/env python3
"""Write cautious manuscript and hackathon text from completed P/D/T/W tables.

The script is intentionally data-bound: it accepts only the completed
post-release analysis tables and emits numerical prose plus an explicit claim
ledger.  It does not create a conclusion when the required paired response is
absent, and it labels the small molecular benchmark's scope limitations in
every report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PRIMARY_COMPARISON = "outer_generated_minus_removed"
FORCE_FIELD = "final_test_delta_force_component_mae_meV_per_A"
ENERGY_FIELD = "final_test_delta_energy_mae_meV_per_atom"
AUC_FIELD = "validation_normalized_auc_delta_force_component_mae_meV_per_A"
TIME_FIELD = "cumulative_training_seconds_delta"
MODEL_EXTENSION_SCHEMA = "micro-mlip-model-method-extension-analysis-v1"
MODEL_FORCE_RESPONSE = "final_test_delta_force_component_mae_meV_per_A"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty required results table: {path}")
    return rows


def numeric(row: Mapping[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric field {field}") from exc


def mean(values: Iterable[float]) -> float:
    copied = list(values)
    if not copied:
        raise ValueError("Cannot average an empty result collection")
    return sum(copied) / len(copied)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_primary(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    selected = [row for row in rows if row.get("comparison") == PRIMARY_COMPARISON]
    if not selected:
        raise ValueError("No outer-generated versus removed paired effects are available")
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in selected:
        grouped.setdefault(str(row["generation"]), []).append(row)
    summary = []
    for generation, members in sorted(grouped.items()):
        summary.append(
            {
                "generation": generation,
                "paired_observations": len(members),
                "mean_force_delta": mean(numeric(row, FORCE_FIELD) for row in members),
                "mean_energy_delta": mean(numeric(row, ENERGY_FIELD) for row in members),
                "mean_force_auc_delta": mean(numeric(row, AUC_FIELD) for row in members),
                "mean_training_time_delta": mean(numeric(row, TIME_FIELD) for row in members),
            }
        )
    return summary


def summarize_model_extension(analysis_dir: Path) -> Dict[str, Any]:
    analysis = load_json(analysis_dir / "analysis.json")
    if analysis.get("schema") != MODEL_EXTENSION_SCHEMA or analysis.get("status") != "COMPLETED":
        raise ValueError("The model-method report section requires a completed matched extension analysis")
    effects = read_csv(analysis_dir / "method_effect_vs_scratch.csv")
    selected = [
        row
        for row in effects
        if row.get("comparison") == PRIMARY_COMPARISON and row.get("response") == MODEL_FORCE_RESPONSE
    ]
    if not selected:
        raise ValueError("Model-method analysis has no outer-generated-minus-removed force comparison")
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in selected:
        method = row.get("method")
        if not method:
            raise ValueError("Model-method effect lacks its method identity")
        grouped.setdefault(method, []).append(row)
    rows = [
        {
            "method": method,
            "observations": len(members),
            "mean_method_minus_scratch": mean(numeric(row, "method_minus_scratch") for row in members),
        }
        for method, members in sorted(grouped.items())
    ]
    forgetting = read_csv(analysis_dir / "method_forgetting.csv")
    replay = [row for row in forgetting if row.get("method") == "multihead_pseudolabel_replay"]
    if len(replay) != 1:
        raise ValueError("Model-method analysis must contain exactly one multihead forgetting row")
    return {
        "rows": rows,
        "multihead_forgetting_force_mae": numeric(replay[0], "mae_f_final_minus_initial"),
        "analysis_sha256": hashlib.sha256((analysis_dir / "analysis.json").read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-analysis-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Results-report directory is immutable; choose a new versioned path")
    analysis = load_json(args.analysis_dir / "analysis.json")
    if analysis.get("schema") != "micro-mlip-full-pdtw-postrelease-analysis-v1" or analysis.get("status") != "COMPLETED":
        raise ValueError("The results report requires a completed verified analysis")
    paired = read_csv(args.analysis_dir / "paired_effects.csv")
    factor_summary = read_csv(args.analysis_dir / "factorial_effect_summary.csv")
    transitions = read_csv(args.analysis_dir / "residual_action_transition_summary.csv")
    evolution = read_csv(args.analysis_dir / "generation_evolution.csv")
    holdout = read_csv(args.analysis_dir / "holdout_paired_effects.csv")
    primary = summarize_primary(paired)
    factors = [row for row in factor_summary if row.get("response") == FORCE_FIELD]
    if not factors:
        raise ValueError("No force-response factorial contrasts are available")
    factors.sort(key=lambda row: float(row["mean_high_minus_low_contrast"]))
    best_factor = factors[0]
    worst_factor = factors[-1]
    evolution_force = [numeric(row, "g1_minus_g0") for row in evolution if row.get("response") == FORCE_FIELD]
    if not evolution_force:
        raise ValueError("No g0→g1 force-evolution rows are available")
    holdout_primary = [row for row in holdout if row.get("comparison") == PRIMARY_COMPARISON]
    if not holdout_primary:
        raise ValueError("No toluene outer-generated versus removed holdout effect is available")
    holdout_force = numeric(holdout_primary[0], FORCE_FIELD)
    typed_transitions = [
        row
        for row in transitions
        if row.get("arm") == "outer_generated" and row.get("tool_level") == "high" and row.get("residual_class") != "RAW_OR_STATIC_ROUTE"
    ]
    model_extension = summarize_model_extension(args.model_analysis_dir) if args.model_analysis_dir is not None else None
    args.output_dir.mkdir(parents=True)

    primary_table = markdown_table(
        ["Generation", "Paired observations", "Force Δ", "Energy Δ", "Force-AUC Δ", "Training-time Δ (s)"],
        [
            [
                row["generation"],
                str(row["paired_observations"]),
                f"{row['mean_force_delta']:.3f}",
                f"{row['mean_energy_delta']:.3f}",
                f"{row['mean_force_auc_delta']:.3f}",
                f"{row['mean_training_time_delta']:.3f}",
            ]
            for row in primary
        ],
    )
    if model_extension is None:
        model_section = """## Matched model-method extension

The foundation-method extension is not yet part of this closed report. The
report deliberately leaves naïve fine tuning, LoRA, layer freezing, and
multihead replay as pending until their actual admission and matched holdouts
close.
"""
        model_claim_boundary = (
            "- Not supported: a model-method conclusion before the separate "
            "foundation-method admission and matched extension finish."
        )

        model_claim_entry: Dict[str, Any] = {
            "claim": "Foundation adaptation methods improve the selected policy.",
            "status": "PENDING_SEPARATE_MATCHED_MODEL_METHOD_EXTENSION",
            "evidence_required": ["foundation-method admission", "matched method campaigns", "forgetting metric for replay"],
        }
    else:
        model_table = markdown_table(
            ["Method", "Matched observations", "Method − scratch paired force-effect change"],
            [
                [row["method"], str(row["observations"]), f"{row['mean_method_minus_scratch']:.3f}"]
                for row in model_extension["rows"]
            ],
        )
        model_section = f"""## Matched model-method extension

The separately admitted foundation-method extension is completed. Its
registered response is method minus scratch for the same outer-generated-minus-
removed final force effect on the one frozen toluene policy holdout:

{model_table}

The multihead replay validation-only pretraining-head force-MAE
final-minus-initial diagnostic was {model_extension['multihead_forgetting_force_mae']:.3f};
positive values indicate replay-domain forgetting in that diagnostic. This
compact method block is descriptive for its one frozen toluene holdout and does
not establish a general model ranking.
"""
        model_claim_boundary = (
            "- Scope-limited support: the matched model-method table compares the admitted methods only "
            "on one frozen toluene holdout; it is not a general foundation-model ranking."
        )

        model_claim_entry = {
            "claim": "Foundation adaptation methods improve the selected policy.",
            "status": "SUPPORTED_ONLY_AS_MATCHED_SINGLE_TOLUENE_HOLDOUT_EXTENSION",
            "reason": "The result is a compact paired extension on one pre-frozen toluene policy holdout, not a general model ranking.",
            "model_analysis_sha256": model_extension["analysis_sha256"],
        }
    report = f"""# Full P/D/T/W micro-MLIP results

## Scope and audit status

This report is generated from the closed g0/g1 full 16-cell scratch P/D/T/W
program and the frozen g2 toluene policy holdout. The analysis replayed the
four-arm release barriers before reading independent metrics. The benchmark is
rMD17 molecular-potential learning; it is not an adsorption-energy or MOF
accuracy claim.

## Primary equal-budget comparison

The registered primary response is the paired `outer_generated − removed`
independent force error. Negative values favor the validation-informed outer
policy relative to the information-removed control. The development summaries
are:

{primary_table}

These are descriptive paired outcomes from the registered molecule × seed
units. They do not establish an effect across all molecules or a general claim
about autonomous scientific reasoning.

## P/D/T/W factor contrasts

Across all recorded generations and force-response contrasts, the most
favorable high-minus-low factor term was **{best_factor['factor_term']}** in
{best_factor['generation']} ({float(best_factor['mean_high_minus_low_contrast']):.3f} meV Å$^{{-1}}$), while the least favorable was
**{worst_factor['factor_term']}** in {worst_factor['generation']}
({float(worst_factor['mean_high_minus_low_contrast']):.3f} meV Å$^{{-1}}$).
The complete factor table, including all interactions and molecule/seed
strata, remains the evidentiary source; this ranking is not a causal mechanism
claim and holds the GNN model recipe at scratch.

## Multi-round feedback trace

The typed-tool outer-policy arms produced {len(typed_transitions)} aggregated
residual-class/action transition groups. Each group links a validation residual
signal to a bounded acquisition action and the immediately following validation
change. These trajectories show how the agent reacted to an observed error,
but the residual labels are operational diagnostics rather than proven physical
causes.

## Iterative evolution and policy holdout

The mean g1-minus-g0 paired force-effect change across matched cell/molecule
records was {mean(evolution_force):.3f} meV Å$^{{-1}}$. g0 and g1 use distinct
registered development seeds, so this is a cross-generation descriptive
contrast rather than a seed-paired confidence interval. The pre-frozen toluene
outer-generated-minus-removed paired force effect was {holdout_force:.3f} meV
Å$^{{-1}}$. Toluene was excluded from promotion and policy generation.

{model_section}

## Claims supported now—and not supported

- Supported: this exact bounded five-round system can be replayed and compared
  under equal label/round budgets with a test-release firewall.
- Supported: the recorded tables quantify P/D/T/W contrasts, feedback traces,
  costs, and the one pre-frozen policy holdout within the stated rMD17 scope.
- Not supported: a universal advantage of any axis, a molecular-physics cause
  for a residual class, or MOF adsorption accuracy.
{model_claim_boundary}
"""
    (args.output_dir / "manuscript_results.md").write_text(report, encoding="utf-8")
    summary = f"""# Research summary

This bounded five-round micro-MLIP program compares 16 P/D/T/W combinations
under matched label and training budgets with an independent evaluator.
The frozen toluene holdout force effect (outer-generated minus removed) is
{holdout_force:.3f} meV per Angstrom. Negative values favor the outer policy.
Residual classes are operational diagnostics rather than physical causes.
Results apply to the stated rMD17 benchmark; MOF adsorption requires separate
validation. See manuscript_results.md and claim_ledger.json for scope.
"""
    (args.output_dir / "research_summary.md").write_text(summary, encoding="utf-8")
    claim_ledger = {
        "schema": "micro-mlip-full-pdtw-claim-ledger-v1",
        "analysis_dir": str(args.analysis_dir),
        "claims": [
            {
                "claim": "Equal-budget five-round multi-arm P/D/T/W outcomes are auditable in this rMD17 benchmark.",
                "status": "SUPPORTED_WITHIN_DECLARED_SCOPE",
                "evidence": ["analysis.json", "paired_effects.csv", "factorial_effects.csv"],
            },
            {
                "claim": "Residual classes demonstrate molecular physical failure mechanisms.",
                "status": "UNSUPPORTED",
                "reason": "Residual labels are operational diagnostics only.",
            },
            {
                "claim": "The loop proves MOF adsorption-energy accuracy.",
                "status": "UNSUPPORTED",
                "reason": "The completed analysis is a small rMD17 molecular-potential benchmark.",
            },
            model_claim_entry,
        ],
    }
    (args.output_dir / "claim_ledger.json").write_text(json.dumps(claim_ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the admitted matched foundation-method extension and evidence package.

This remote-side program starts only after the completed scratch P/D/T/W
program has frozen one policy and closed its toluene holdout.  It serially
admits four foundation methods, runs their matched four-arm/five-round toluene
extensions, analyzes every closed receipt, and renders a pending-QA evidence
package.  It never changes the selected policy, calls an outer model, retries
a failed method, or accesses a new holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsi.micro_mlip.data import sha256


FOUNDATION_METHODS = (
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)
MACE_SIDECAR_VERSION = "0.3.15"
MACE_SIDECAR_WHEEL_SHA256 = "9e38df131667f4abdb3c0ab7b77927f217886cbca68c417dcc985465c4b9a53c"
MACE_REQUIRED_FLAGS = (
    "--foundation_model",
    "--foundation_model_elements",
    "--multiheads_finetuning",
    "--pseudolabel_replay",
    "--pt_train_file",
    "--pt_valid_file",
    "--skip_evaluate_heads",
    "--lora",
    "--lora_rank",
    "--lora_alpha",
    "--freeze",
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


def invoke(command: Sequence[str], log_path: Path) -> Dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_sha256": sha256(log_path),
    }


def require_success(label: str, receipt: Mapping[str, Any]) -> None:
    if receipt.get("exit_code") != 0:
        raise RuntimeError(f"{label} failed; inspect log {receipt.get('log_sha256')}")


def selected_source_policy(
    *,
    promotion: Mapping[str, Any],
    primary_program_root: Path,
    primary_code_root: Path,
) -> Path:
    selected = promotion.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Primary promotion receipt lacks a selected policy")
    generation = selected.get("generation")
    cell_id = selected.get("factorial_cell_id")
    if generation not in {"g0", "g1"} or not isinstance(cell_id, str):
        raise ValueError("Primary promotion has an invalid selected generation/cell")
    if generation == "g0":
        policy = primary_code_root / "benchmarks/micro_mlip_rsi/pdtw_full_policies_g0_20261021" / cell_id / "policies.json"
    else:
        policy = primary_program_root / "g1_policies" / cell_id / "policies.json"
    if not policy.is_file():
        raise FileNotFoundError(f"Selected source policy is missing: {policy}")
    expected_hash = selected.get("policy_manifest_sha256")
    if not isinstance(expected_hash, str) or sha256(policy) != expected_hash:
        raise ValueError("Selected source policy does not match the promotion receipt hash")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-id", required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--primary-program-root", type=Path, required=True)
    parser.add_argument("--primary-code-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, required=True)
    parser.add_argument("--replay-valid-file", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mace-runner", type=Path, required=True)
    parser.add_argument("--mace-site", type=Path, required=True)
    parser.add_argument("--mace-sidecar-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--formal-launcher-pid", type=int, required=True)
    parser.add_argument("--formal-state", type=Path, required=True)
    parser.add_argument("--formal-done", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=120.0)
    parser.add_argument("--admission-molecule", default="ethanol")
    parser.add_argument("--admission-seed", type=int, default=20261022)
    parser.add_argument("--holdout-seed", type=int, default=20261022)
    args = parser.parse_args()

    if args.extension_root.exists():
        raise FileExistsError("Model-method extension root already exists; executions are immutable")
    sidecar_root = args.mace_sidecar_root or args.extension_root / f"mace_torch_{MACE_SIDECAR_VERSION}_sidecar"
    base_config = args.code_root / "configs/micro-mlip-rsi-five-round-v1.json"
    registry = args.code_root / "benchmarks/micro_mlip_rsi/pdtw_full_scratch_registry_20261021.json"
    for path, label in (
        (base_config, "five-round config"),
        (registry, "full P/D/T/W registry"),
        (args.dataset_dir, "five-round dataset directory"),
        (args.foundation_model, "pinned foundation model"),
        (args.replay_file, "replay training artifact"),
        (args.replay_valid_file, "replay validation artifact"),
        (args.mace_runner, "MACE runner"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not args.mace_site.is_dir():
        raise FileNotFoundError(f"Missing MACE site directory: {args.mace_site}")
    if sha256(args.replay_file) == sha256(args.replay_valid_file):
        raise ValueError("Replay train and replay validation artifacts must be distinct")
    promotion_path = args.primary_program_root / "promotion.json"
    scratch_holdout_root = args.primary_program_root / "g2_toluene"
    promotion = load_json(promotion_path)
    if promotion.get("schema") != "micro-mlip-pdtw-promotion-selection-v1" or promotion.get("status") != "FROZEN_BEFORE_POLICY_HOLDOUT":
        raise ValueError("Primary P/D/T/W promotion is not frozen")
    if promotion.get("five_round_config_sha256") != sha256(base_config):
        raise ValueError("Primary promotion does not match the model-extension base configuration")
    scratch_holdout = load_json(scratch_holdout_root / "holdout_manifest.json")
    if scratch_holdout.get("status") != "COMPLETED_PENDING_ANALYSIS" or scratch_holdout.get("method") != "scratch_micro":
        raise ValueError("The matched scratch toluene reference is not closed")
    source_policy = selected_source_policy(
        promotion=promotion,
        primary_program_root=args.primary_program_root,
        primary_code_root=args.primary_code_root,
    )

    args.extension_root.mkdir(parents=True)
    log_root = args.extension_root / "logs"
    manifest_path = args.extension_root / "extension_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "micro-mlip-model-method-extension-v1",
        "status": "PREPARED",
        "created_at": utc_now(),
        "extension_id": args.extension_id,
        "source_sha256": {
            "base_config": sha256(base_config),
            "registry": sha256(registry),
            "promotion": sha256(promotion_path),
            "scratch_holdout_manifest": sha256(scratch_holdout_root / "holdout_manifest.json"),
            "source_policy": sha256(source_policy),
            "foundation_model": sha256(args.foundation_model),
            "replay_file": sha256(args.replay_file),
            "replay_validation_file": sha256(args.replay_valid_file),
            "mace_runner": sha256(args.mace_runner),
        },
        "primary_program_root": str(args.primary_program_root),
        "primary_code_root": str(args.primary_code_root),
        "source_policy": str(source_policy),
        "mace_sidecar": {
            "root": str(sidecar_root),
            "mace_version": MACE_SIDECAR_VERSION,
            "wheel_sha256": MACE_SIDECAR_WHEEL_SHA256,
            "required_flags": list(MACE_REQUIRED_FLAGS),
        },
        "methods": list(FOUNDATION_METHODS),
        "stages": {},
        "boundary": "All method changes occur after P/D/T/W promotion. Every training call is serial and individually resource-gated; no failed method is retried, replaced, or used for policy selection.",
    }
    write_json(manifest_path, manifest)
    common_gate = [
        "--formal-launcher-pid", str(args.formal_launcher_pid),
        "--formal-state", str(args.formal_state),
        "--formal-done", str(args.formal_done),
        "--disk-path", str(args.disk_path),
        "--minimum-free-gib", str(args.minimum_free_gib),
    ]
    try:
        manifest["status"] = "GATING_AND_PREPARING_MACE_SIDECAR"
        write_json(manifest_path, manifest)
        sidecar_gate_path = args.extension_root / "sidecar_resource_gate.json"
        sidecar_gate = invoke(
            [
                args.python,
                str(args.code_root / "scripts/check_micro_mlip_resource_gate.py"),
                *common_gate,
                "--output", str(sidecar_gate_path),
            ],
            log_root / "sidecar_resource_gate.log",
        )
        manifest["stages"]["sidecar_resource_gate"] = sidecar_gate
        write_json(manifest_path, manifest)
        require_success("shared-resource gate before MACE sidecar provisioning", sidecar_gate)
        sidecar_command = [
            args.python,
            str(args.code_root / "scripts/prepare_micro_mlip_mace_sidecar.py"),
            "--sidecar-root", str(sidecar_root),
            "--python", args.python,
            "--runner", str(args.mace_runner),
            "--mace-version", MACE_SIDECAR_VERSION,
            "--wheel-sha256", MACE_SIDECAR_WHEEL_SHA256,
        ]
        for flag in MACE_REQUIRED_FLAGS:
            sidecar_command.extend(["--required-flag", flag])
        sidecar = invoke(sidecar_command, log_root / "prepare_mace_sidecar.log")
        manifest["stages"]["prepare_mace_sidecar"] = sidecar
        write_json(manifest_path, manifest)
        require_success("pinned MACE sidecar capability preflight", sidecar)
        sidecar_doc = load_json(sidecar_root / "sidecar_manifest.json")
        if sidecar_doc.get("status") != "READY":
            raise ValueError("MACE sidecar did not produce a READY capability receipt")
        site_packages = Path(str(sidecar_doc.get("site_packages", "")))
        if not site_packages.is_dir():
            raise ValueError("MACE sidecar receipt has no readable site-packages directory")
        inherited_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(site_packages) if not inherited_pythonpath else str(site_packages) + os.pathsep + inherited_pythonpath
        manifest["mace_sidecar"]["manifest_sha256"] = sha256(sidecar_root / "sidecar_manifest.json")
        manifest["mace_sidecar"]["effective_pythonpath_prefix"] = str(site_packages)
        write_json(manifest_path, manifest)

        admission_root = args.extension_root / "admission"
        manifest["status"] = "RUNNING_FOUNDATION_METHOD_ADMISSION"
        write_json(manifest_path, manifest)
        admission = invoke(
            [
                args.python,
                str(args.code_root / "scripts/run_micro_mlip_foundation_method_admission.py"),
                "--admission-id", f"{args.extension_id}__admission",
                "--output-root", str(admission_root),
                "--base-config", str(base_config),
                "--dataset-dir", str(args.dataset_dir),
                "--foundation-model", str(args.foundation_model),
                "--replay-file", str(args.replay_file),
                "--replay-valid-file", str(args.replay_valid_file),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--molecule", args.admission_molecule,
                "--seed", str(args.admission_seed),
                *common_gate,
            ],
            log_root / "admission.log",
        )
        manifest["stages"]["admission"] = admission
        write_json(manifest_path, manifest)
        require_success("foundation-method admission", admission)
        admission_path = admission_root / "method_admission.json"
        admission_doc = load_json(admission_path)
        if admission_doc.get("status") != "ADMITTED":
            raise ValueError("Foundation-method admission did not produce an ADMITTED receipt")
        admitted_config = admission_root / "admitted_method_config.json"
        if admission_doc.get("admitted_config_sha256") != sha256(admitted_config):
            raise ValueError("Admitted method configuration hash does not match the admission receipt")

        derived_root = args.extension_root / "derived_policies"
        holdout_root = args.extension_root / "holdouts"
        manifest["status"] = "RUNNING_MATCHED_METHOD_HOLDOUTS"
        write_json(manifest_path, manifest)
        method_holdout_specs = []
        for method in FOUNDATION_METHODS:
            derived_policy = derived_root / f"{method}.json"
            derive = invoke(
                [
                    args.python,
                    str(args.code_root / "scripts/derive_micro_mlip_method_policy_manifest.py"),
                    "--source-policies", str(source_policy),
                    "--target-method", method,
                    "--admission", str(admission_path),
                    "--output", str(derived_policy),
                ],
                log_root / f"derive_{method}.log",
            )
            manifest["stages"][f"derive_{method}"] = derive
            write_json(manifest_path, manifest)
            require_success(f"derive frozen policy for {method}", derive)
            command = [
                args.python,
                str(args.code_root / "scripts/run_micro_mlip_policy_holdout.py"),
                "--holdout-id", f"{args.extension_id}__{method}",
                "--holdout-root", str(holdout_root / method),
                "--promotion", str(promotion_path),
                "--policy-manifest", str(derived_policy),
                "--base-five-round-config", str(base_config),
                "--five-round-config", str(admitted_config),
                "--method", method,
                "--source-policy-manifest", str(source_policy),
                "--admission", str(admission_path),
                "--dataset-dir", str(args.dataset_dir),
                "--code-root", str(args.code_root),
                "--python", args.python,
                "--mace-runner", str(args.mace_runner),
                "--mace-site", str(args.mace_site),
                "--device", args.device,
                "--foundation-model", str(args.foundation_model),
                "--seed", str(args.holdout_seed),
                *common_gate,
            ]
            if method == "multihead_pseudolabel_replay":
                command.extend(["--replay-file", str(args.replay_file), "--replay-valid-file", str(args.replay_valid_file)])
            holdout = invoke(command, log_root / f"holdout_{method}.log")
            manifest["stages"][f"holdout_{method}"] = holdout
            write_json(manifest_path, manifest)
            require_success(f"matched toluene holdout for {method}", holdout)
            method_holdout_specs.extend(["--method-holdout", f"{method}={holdout_root / method}"])

        analysis_root = args.extension_root / "model_method_analysis"
        manifest["status"] = "ANALYZING_CLOSED_METHOD_EXTENSION"
        write_json(manifest_path, manifest)
        extension_analysis = invoke(
            [
                args.python,
                str(args.code_root / "scripts/analyze_micro_mlip_model_method_extension.py"),
                "--scratch-holdout", str(scratch_holdout_root),
                "--promotion", str(promotion_path),
                "--method-admission", str(admission_path),
                *method_holdout_specs,
                "--output-dir", str(analysis_root),
            ],
            log_root / "model_method_analysis.log",
        )
        manifest["stages"]["model_method_analysis"] = extension_analysis
        write_json(manifest_path, manifest)
        require_success("model-method extension analysis", extension_analysis)

        pdtw_analysis_root = args.extension_root / "pdtw_analysis"
        pdtw_analysis = invoke(
            [
                args.python,
                str(args.code_root / "scripts/analyze_micro_mlip_full_pdtw_program.py"),
                "--program-root", str(args.primary_program_root),
                "--registry", str(registry),
                "--five-round-config", str(base_config),
                "--output-dir", str(pdtw_analysis_root),
            ],
            log_root / "pdtw_analysis.log",
        )
        manifest["stages"]["pdtw_analysis"] = pdtw_analysis
        write_json(manifest_path, manifest)
        require_success("full P/D/T/W post-release analysis", pdtw_analysis)

        figures_pdtw = invoke(
            [
                args.python,
                str(args.code_root / "scripts/render_micro_mlip_full_pdtw_figures.py"),
                "--analysis-dir", str(pdtw_analysis_root),
                "--output-dir", str(args.extension_root / "figures_pdtw"),
            ],
            log_root / "render_pdtw_figures.log",
        )
        manifest["stages"]["render_pdtw_figures"] = figures_pdtw
        write_json(manifest_path, manifest)
        require_success("full P/D/T/W figure rendering", figures_pdtw)

        figures_model = invoke(
            [
                args.python,
                str(args.code_root / "scripts/render_micro_mlip_model_method_figures.py"),
                "--analysis-dir", str(analysis_root),
                "--output-dir", str(args.extension_root / "figures_model_methods"),
            ],
            log_root / "render_model_method_figures.log",
        )
        manifest["stages"]["render_model_method_figures"] = figures_model
        write_json(manifest_path, manifest)
        require_success("model-method figure rendering", figures_model)

        report = invoke(
            [
                args.python,
                str(args.code_root / "scripts/write_micro_mlip_full_pdtw_results_report.py"),
                "--analysis-dir", str(pdtw_analysis_root),
                "--model-analysis-dir", str(analysis_root),
                "--output-dir", str(args.extension_root / "report"),
            ],
            log_root / "write_report.log",
        )
        manifest["stages"]["write_report"] = report
        write_json(manifest_path, manifest)
        require_success("full results report", report)
        manifest["status"] = "COMPLETED_PENDING_MANUAL_FIGURE_QA"
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "extension_root": str(args.extension_root)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "FAILED_NO_AUTO_RETRY"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

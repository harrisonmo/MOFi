"""Validate the blocked MOF adsorption-energy admission package.

The validator is intentionally fail-closed: prior S2EF numbers are accepted only
as inherited evidence, while adsorption-energy and MD claims require a verified
paired reference and an executed artifact set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "benchmarks" / "mof_adsorption"
CONFIG = ROOT / "configs" / "mof-adsorption-admission-blocked-v1.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    errors: list[str] = []
    config = load_json(CONFIG)
    source = load_json(PACKAGE / "source_manifest.json")
    split = load_json(PACKAGE / "split_manifest.json")
    execution = load_json(PACKAGE / "execution_record.json")
    summary = load_json(PACKAGE / "admission_summary.json")

    if config.get("status") != "ADMISSION_BLOCKED":
        errors.append("config_status_not_blocked")
    estimand = config.get("estimand", {})
    if estimand.get("target_guest") != "CO2":
        errors.append("target_guest_is_not_CO2")
    if estimand.get("adsorption_energy_equation") != "E_ads = E(host+guest) - E(host) - n_guest E(guest)":
        errors.append("adsorption_equation_changed")
    if estimand.get("sign_convention") != "negative values indicate favorable adsorption":
        errors.append("sign_convention_changed")

    admission = config.get("admission", {})
    if admission.get("status") != "BLOCKED":
        errors.append("admission_status_not_blocked")
    if admission.get("data_pairing", {}).get("odac_total_energy_as_adsorption_energy") is not False:
        errors.append("odac_total_energy_mislabelled_as_adsorption_energy")
    if admission.get("independent_oracle", {}).get("pfp_proxy_oracle") != "NOT_ADMITTED":
        errors.append("pfp_oracle_not_explicitly_blocked")

    if source.get("data_status") != "ODAC_TOTAL_ENERGY_FORCE_ONLY_UNTIL_PAIRED_TRIPLETS_ARE_VERIFIED":
        errors.append("source_boundary_missing")
    if source.get("pairing_requirement", {}).get("status") != "NOT_VERIFIED":
        errors.append("pairing_status_not_unverified")
    if split.get("status") != "NOT_MATERIALIZED":
        errors.append("split_should_not_be_materialized")
    if execution.get("experiment_execution", {}).get("adsorption_energy") != "NOT_STARTED":
        errors.append("adsorption_energy_execution_status_changed")
    if execution.get("experiment_execution", {}).get("md_smoke_test") != "NOT_STARTED":
        errors.append("md_execution_status_changed")
    if summary.get("decision") != "RETAIN_PARENT_AND_REQUEST_DATA_ORACLE_ACCESS":
        errors.append("decision_changed")

    metric_rows: list[dict[str, str]] = []
    with (PACKAGE / "prior_s2ef_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    if [row.get("arm") for row in metric_rows] != ["Foundation / A", "B", "C", "D"]:
        errors.append("prior_metric_arm_order_changed")
    for row in metric_rows:
        for field in ("energy_mae_eV_per_atom", "force_component_mae_eV_per_A"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid_prior_metric:{row.get('arm')}:{field}")
                continue
            if not math.isfinite(value) or value < 0:
                errors.append(f"nonfinite_prior_metric:{row.get('arm')}:{field}")
        if "inherited PR1" not in row.get("source_status", ""):
            errors.append(f"prior_metric_source_not_inherited:{row.get('arm')}")

    tracked = [
        CONFIG,
        PACKAGE / "source_manifest.json",
        PACKAGE / "split_manifest.json",
        PACKAGE / "execution_record.json",
        PACKAGE / "admission_summary.json",
        PACKAGE / "figure_qa.json",
        PACKAGE / "prior_s2ef_metrics.csv",
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in tracked}
    return {
        "valid": not errors,
        "errors": errors,
        "experiment_id": config.get("experiment_id"),
        "decision": summary.get("decision"),
        "admission": summary.get("admission"),
        "tracked_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKAGE / "admission_validation.json")
    args = parser.parse_args()
    result = validate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

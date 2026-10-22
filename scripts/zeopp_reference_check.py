"""Compare registered Zeo++ tool results with Crystal-Builder's recorded Zeo++ 0.3 values.

Usage: python scripts/zeopp_reference_check.py runs/zeopp-validation-01 [report.json]
Accepts a task-ID registry session (ledger.json) or a science hub session (hub-ledger.json).
The reference used -sa 2000 and -volpo 50000 at N2 1.86 A with Zeo++ radii and -ha,
matching the shipped recipes; Monte Carlo noise sets the tolerances below.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Crystal-Builder/tests/data/zeopp_reference.json"
TASKS = {"mof5-n2": "MOF-5", "hkust1-n2": "HKUST1", "uio66-n2": "UIO66", "zif8-n2": "ZIF-8"}
# Science hub structure IDs from configs/science-hub.json.
HUB_STRUCTURES = {"mof5": "MOF-5", "hkust1": "HKUST1", "uio66": "UIO66", "zif8": "ZIF-8"}
# Surface area: relative 1 % (or 5 A^2 per cell when near zero); the reference ASA/NASA are A^2 per cell; volume: 0.01 of the cell; density: 1e-4 relative.
ASA_RELATIVE, ASA_FLOOR, POAV_ABSOLUTE, DENSITY_RELATIVE = 0.01, 5.0, 0.01, 1e-4


def close(value, expected, relative=0.0, absolute=0.0):
    return abs(value - expected) <= max(absolute, relative * abs(expected))


def main():
    session = Path(sys.argv[1]).resolve()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    hub = (session / "hub-ledger.json").exists()
    ledger = json.loads((session / ("hub-ledger.json" if hub else "ledger.json")).read_text(encoding="utf-8"))
    rows = []
    for call in ledger["calls"]:
        if hub:
            if call["tool"] != "zeopp_pore_geometry" or call["arguments"]["structure_id"] not in HUB_STRUCTURES:
                continue
            call = {**call, "task_id": call["arguments"]["structure_id"]}
            name = HUB_STRUCTURES[call["task_id"]]
        else:
            name = TASKS[call["task_id"]]
        result = json.loads((session / call["id"] / "tool-result.json").read_text(encoding="utf-8"))
        expected, data = reference[name], result["data"]
        if not result["ok"]:
            rows.append({"task_id": call["task_id"], "ok": False, "error": result["error"]})
            continue
        density = data["cell"]["density_g_cm3"]
        measured = {"asa": data["surface_area"]["accessible_angstrom2"],
                    "nasa": data["surface_area"]["nonaccessible_angstrom2"],
                    "poav": data["pore_volume"]["accessible_fraction"],
                    "ponav": data["pore_volume"]["nonaccessible_fraction"], "density": density}
        checks = {key: close(measured[key], expected[key], ASA_RELATIVE, ASA_FLOOR)
                  for key in ("asa", "nasa")}
        checks |= {key: close(measured[key], expected[key], absolute=POAV_ABSOLUTE)
                   for key in ("poav", "ponav")}
        checks["density"] = close(density, expected["density"], DENSITY_RELATIVE)
        rows.append({"task_id": call["task_id"], "structure": name, "ok": True,
                     "measured": measured, "reference": {k: expected[k] for k in measured},
                     "checks": checks, "passed": all(checks.values()),
                     "diameters_angstrom": data["diameters_angstrom"],
                     "channels": data["channels"]["summary"], "flags": data["flags"],
                     "grid_crosscheck_agrees": (data.get("grid_crosscheck") or {}).get("agrees"),
                     "elapsed_seconds": result["cost"]["elapsed_seconds"]})
    report = {"reference": str(REFERENCE.relative_to(ROOT)), "session": str(session),
              "tolerances": {"asa_relative": ASA_RELATIVE, "asa_floor_angstrom2": ASA_FLOOR,
                             "volume_fraction_absolute": POAV_ABSOLUTE,
                             "density_relative": DENSITY_RELATIVE},
              "rows": rows, "all_passed": bool(rows) and all(r.get("passed") for r in rows)}
    text = json.dumps(report, indent=2)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
    print(text)
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()

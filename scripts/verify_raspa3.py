"""Bounded native RASPA3 CLI and project-adapter installation check (WSL)."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.gcmc import core


def main():
    output = ROOT / "runs" / "raspa3-install-check"
    output.mkdir(parents=True, exist_ok=False)
    home = Path(os.environ["RASPA3_WSL_HOME"])
    source = home / "usr/share/raspa3/examples/basic/1_mc_methane_in_box"
    cli = output / "cli"
    shutil.copytree(source, cli)
    config = json.loads((cli / "simulation.json").read_text())
    config.update(NumberOfInitializationCycles=100, NumberOfCycles=200, PrintEvery=100)
    config["Systems"][0]["OutputPDBMovie"] = False
    core.write_json(cli / "simulation.json", config)
    lib = home / "usr/lib"
    env = dict(os.environ, RASPA_DIR=str(home / "usr/share/raspa3"),
               LD_LIBRARY_PATH=":".join(str(lib / p) for p in
                   ("x86_64-linux-gnu/blas64", "x86_64-linux-gnu/lapack64", "llvm-14/lib")),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    with (cli / "console.log").open("w") as log:
        subprocess.run([str(home / "usr/bin/raspa3")], cwd=cli, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    native = list((cli / "output").glob("*.txt"))
    assert native and any("Simulation finished!" in p.read_text() for p in native)
    cli_seconds = time.monotonic() - started
    cfg = core.read_json(ROOT / "configs/methane-gcmc-example.json")
    cfg.update(structure=str(ROOT / "examples/gcmc/MFI_SI.cif"),
               force_field=str(ROOT / "examples/gcmc/mfi-force-field.json"),
               seeds=[101], initialization_cycles=100, production_cycles=200)
    cfg["budget"].update(max_jobs=2, max_total_cycles=600,
                         timeout_seconds_per_job=30, max_wall_seconds=60)
    core.write_json(output / "gcmc-config.json", cfg)
    core.prepare(output / "gcmc-config.json", output / "gcmc")
    result = core.execute(output / "gcmc")
    core.write_json(output / "verification.json", {
        "cli_version": "3.0.30", "cli_wall_seconds": cli_seconds,
        "cli_finished": True, "adapter_result": result,
        "python": sys.executable, "python_version": sys.version,
    })
    print(json.dumps(result, indent=2))
    print(f"CLI finished in {cli_seconds:.3f} seconds; evidence: {output}")


if __name__ == "__main__":
    main()

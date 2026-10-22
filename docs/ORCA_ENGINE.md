# ORCA engine adapter


## Why a separate engine

`rsi/dft` covers periodic plane-wave and GPW codes (GPAW, CP2K). ORCA is a molecular Gaussian-basis code with no periodic boundary conditions. It answers different questions: adsorbate and linker properties, cluster models cut from a framework, hybrid and range-separated functionals, harmonic frequencies and thermochemistry. Its parameter space (functional + basis + dispersion + RI) is also different. So it is a separate package and a separate hub tool, `orca_calculation`, rather than another `dft_calculation` backend.

## Capabilities

`rsi/orca` runs one bounded ORCA job per prepared run:

| Task | ORCA keywords | Result |
|---|---|---|
| `energy` | single point | final energy (Eh, eV), SCF convergence, HOMO-LUMO gap |
| `gradient` | `EnGrad` | plus forces (eV/Angstrom) from `job.engrad` |
| `optimize` | `LooseOpt`/`Opt`/`TightOpt`/`VeryTightOpt`, `%geom MaxIter` | plus optimization convergence, cycles and the final geometry from `job.xyz` |
| `frequencies: true` (any task) | `Freq` | every mode, imaginary-mode count, lowest mode, zero-point energy, enthalpy and Gibbs free energy (298.15 K, 1 atm) |

- **Input writing** goes through Crystal-Builder's `xtal.orca` writer, which already holds ORCA 6.1's functional and basis tables. A functional or basis outside that catalogue is refused, as are a basis with a 3c method and a dispersion correction on a functional that already has one. A multiplicity the electron count cannot have is also refused, before anything is written. The writer's cautions, such as a basis that does not cover an element, are kept in the manifest and the summary.
- **Structures** must be nonperiodic (`.xyz`/`.extxyz` without `pbc`). A periodic MOF has to be cut to a cluster first. That step is a modelling choice with dangling bonds, and it is not automated here.
- **Execution**: `RSI_ORCA` must give the absolute path of the `orca` binary (ORCA starts its MPI subprograms by that path). The binary is accepted only if `orca_scf` sits beside it. That check tells it apart from the GNOME screen reader that Ubuntu also installs as `/usr/bin/orca`. The ORCA folder is prepended to `PATH` and `LD_LIBRARY_PATH` for the job. `budget.cpus` becomes `%pal nprocs` and `budget.maxcore_mb` becomes `%maxcore`. Through the hub the job runs on the `rsi.hpc` scheduler, which reserves `cpus` and `cpus x maxcore_mb` of memory.
- **Verification**: the controller does not trust the exit code. It re-reads ORCA's own files and requires all of the following:
  - `****ORCA TERMINATED NORMALLY****` and a finite `FINAL SINGLE POINT ENERGY`.
  - The MPI process count ORCA printed equals the reservation.
  - For gradients and optimizations, `job.engrad` exists, with the right atom count and an energy equal to the final energy within 1e-8 Eh.
  - For optimizations, a final geometry with the right atom count.
  - For frequencies, exactly 3N printed modes.

  `converged` means every SCF converged and, for an optimization, `THE OPTIMIZATION HAS CONVERGED` was printed. An unconverged result is still reported, as unconverged. Imaginary modes are counted and reported, never filtered out.
- **Provenance**: `prepare` freezes the source structure, config, rendered input and coordinates. It records hashes of `rsi/orca` and the Crystal-Builder writer (`xtal/orca/*.py`, `tables.json`), and the commits of both repositories. `execute` records the executable path and its SHA-256, plus the ORCA version printed in the output.
- **No restarts**: a timeout or failure is final, so prepare a new run. ORCA's own `.gbw` stays in `run/` for manual reuse.

## Hub tool

`orca_calculation` (contract `orca-calculation-v1`, manifest `tools/science/orca-calculation-v1.json`) lets the model choose:

- structure, task, functional, basis, dispersion, RI and SCF threshold;
- optimization level and cycle limit, and whether to compute frequencies;
- charge, multiplicity, CPUs and wall time.

The controller config (`tools.orca_calculation` in `configs/science-hub*.json`) fixes:

- the admitted functionals (default: R2SCAN-3C, B97-3C, PBEH-3C, PBE, BP86, TPSS, R2SCAN, B3LYP, PBE0, TPSSH, WB97X-D3BJ, WB97M-V);
- the admitted basis sets (def2-SVP/TZVP/TZVPP/QZVPP/TZVPD, ma-def2-TZVP);
- the limits: 60 atoms, 8 CPUs, 2000 MB per process, 4 h wall time and 200 optimization cycles.

The hub validates both lists against the catalogue at start-up. Only nonperiodic library structures appear in the tool's enum. The library now includes experimental gas-phase geometries of H2O, CO2, N2, CH4 and benzene (`examples/orca/`). The model receives the decision-relevant numbers. Forces, all modes and the final geometry stay in `summary.json` as evidence.

Without a valid `RSI_ORCA`, the catalog lists the tool as unavailable and gives the reason. When the tool is available, the session binding includes the adapter and writer hashes and the executable path.

## Usage

```bash
export RSI_ORCA=/path/to/orca
python -m rsi.orca prepare --config inputs/config.json --output runs/orca-water-01
python -m rsi.orca run --run runs/orca-water-01
python -m rsi.science.hub call --config configs/science-hub-wsl.json --session runs/hub-orca-01 \
  --tool orca_calculation --arguments '{"structure_id": "water", "task": "gradient", "functional": "R2SCAN-3C", "basis": null, "dispersion": null, "ri": null, "scf_convergence": null, "opt_level": null, "max_opt_cycles": null, "frequencies": false, "charge": 0, "spin_multiplicity": 1, "cpus": 1, "wall_seconds": 600}'
```

Set `RSI_ORCA` to the absolute path of your ORCA executable.

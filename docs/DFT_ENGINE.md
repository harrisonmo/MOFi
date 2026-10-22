# DFT engine adapter (ASE / GPAW / CP2K)


## Capabilities

`rsi/dft` runs bounded ASE jobs on one of three backends:

| Backend | Use | Native log parsed |
|---|---|---|
| `gpaw` | DFT (PW, LCAO or FD mode) via `gpaw.GPAW` | `gpaw.txt`: `Converged in N steps` (legacy `Converged after N iterations`), `Extrapolated:` energies (eV), final force table |
| `cp2k` | DFT via ASE `CP2K` and `cp2k.psmp --shell` | `cp2k.out`: `SCF run converged` / `SCF run NOT converged`, `ENERGY| Total FORCE_EVAL` (Hartree, `[a.u.]` or `[hartree]`; converted with CP2K's CODATA 2006 factor) |
| `emt` | ASE EMT **test surrogate, not DFT**; results carry `is_dft: false` | none |

- **Energy/forces** (`task: single_point`): eV and eV/Angstrom, plus raw forces on constrained atoms.
- **Constrained optimization** (`task: optimize`): BFGS, LBFGS or FIRE with fixed cell; `fixed_atoms`, `fixed_elements` and `fixed_cartesian` (mask `true` = fixed). Cell relaxation is not supported.
- **Convergence checks**: the controller recomputes them from saved forces and logs, independent of the worker. `converged` requires both conditions:
  1. Every native SCF converged. CP2K continues after an unconverged SCF, so this is checked explicitly.
  2. For optimizations, the largest free force over unconstrained components is at most `fmax`.
  The controller also rejects any run where a constrained coordinate moved by more than 1e-6 Angstrom, or where the last native energy and the ASE energy differ by more than 1e-4 eV.
- **Restart**: after every force evaluation, the worker writes an atomic `checkpoint.json` with geometry, energy and forces. `resume` continues a `timeout` or `interrupted` run from the newest checkpoint, verified by hash. It uses the remaining step and wall budget and is limited by `max_restarts`. A restart rebuilds the optimizer state; it does not carry the BFGS Hessian or the SCF wavefunction across attempts. `failed` runs (non-zero worker exit, e.g. a GPAW SCF failure) and `completed` runs are never resumed. Prepare a new run instead.
- **Timeout**: each attempt runs in a separate worker process and is limited to `min(timeout_seconds_per_attempt, remaining max_wall_seconds)`. On expiry, the controller kills the worker's process tree (`taskkill /T` on Windows, a process group on POSIX).
- **Provenance**: `prepare` freezes the source structure, config, parsed structure and the `rsi/dft` source, and records commits for both repositories. Execution records package versions and, for CP2K, the executable path and SHA256.

## Usage

```bash
python -m rsi.dft prepare --config examples/dft/emt-constrained-relax.json --output runs/dft/example
python -m rsi.dft run --run runs/dft/example
python -m rsi.dft resume --run runs/dft/example   # only after timeout/interruption
python -m rsi.dft parse --backend cp2k --log path/to/cp2k.out
```

The config schema is defined by `validate_config` in `rsi/dft/core.py`. Methods must be explicit:

- GPAW needs `mode` (PW also needs `ecut`; FD needs `h`; LCAO needs `basis`) and `xc`.
- CP2K needs `executable` (a command such as `cp2k.psmp --shell`), `xc`, `cutoff_ev` (ASE passes eV), `basis_set` and `pseudo_potential`. Set `CP2K_DATA_DIR` when the build's compiled data path is unusable, as with conda-forge CP2K 2024.2; the value is recorded in the manifest.
- CP2K 2024.2 aborts on SCF failure by default, so the run is reported as `failed`. With `IGNORE_CONVERGENCE_FAILURE` in `inp`, it returns numbers and the adapter reports `converged: false`.

Charge and multiplicity are top-level fields. `prepare` rejects electron counts whose parity contradicts the multiplicity. Open-shell GPAW jobs need `spinpol: true`, and optional `initial_magmoms` must sum to `multiplicity - 1`. GPAW does not fix the total moment unless the occupations say so. The examples in `examples/dft/` (GPAW PW relaxation and CP2K doublet single point of a Cu19 cluster) pass validation and preparation but have not been run.

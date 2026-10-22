# Unified science tool hub


`rsi/science/hub.py` registers every scientific tool in one model-facing catalog. The model chooses **structures, methods and parameters** inside controller bounds; the controller owns the libraries, engines, compute budget and evidence. Every tool runs through the [HPC scheduler](HPC_SCHEDULER.md), so local and Slurm execution share one budget.

## Tools

| Tool | Version / contract | Model chooses | Engine path |
|---|---|---|---|
| `zeopp_pore_geometry` | 2 / `zeopp-pore-geometry-v2` | structure, probe and PSD radii, sampling preset, `-ha`, pocket blocking, grid cross-check spacing, time limit | `rsi.zeopp` prepare/execute |
| `methane_gcmc_capacity` | 2 / `methane-gcmc-capacity-v2` | structure, force field, temperature, pressures, loading/unloading pressure, seed count, cycles, per-job time limit | `rsi.gcmc` prepare/execute, one scheduled job per pressure x seed |
| `dft_calculation` | 1 / `dft-calculation-v1` | structure, backend, task, functional, cutoff, k-points, charge, spin, fixed elements, optimizer, fmax, steps, CPUs, time limit | `rsi.dft` prepare/execute |
| `orca_calculation` | 1 / `orca-calculation-v1` | nonperiodic structure, task (energy/gradient/optimize), functional, basis, dispersion, RI, SCF threshold, optimization level and cycles, frequencies, charge, spin, CPUs, time limit | `rsi.orca` prepare/execute; see [ORCA engine](ORCA_ENGINE.md) |
| `openmm_minimize` | 1 / `openmm-minimize-v1` | system, tolerance, iteration limit, platform, CPUs, time limit | `rsi.openmm_engine.run_isolated` |
| `science_budget_status` | 1 | nothing | free status call |
| `mof_database_catalog` | 1 / `mof-database-catalog-v1` | filters over an external MOF database survey table | free lookup in `rsi.science.mof_databases`; see [MOF database catalog](MOF_DATABASE_CATALOG.md) |

Manifests live in `tools/science/*-v2.json` / `*-v1.json` with `handler: rsi.science.hub`. The hub fills enums (structure IDs, backends, functionals, force fields, systems, platforms) and appends "Allowed: a to b" ranges to property descriptions from the controller config, so the definitions the model sees are always the bounds the hub enforces. Definitions use OpenAI function format with `strict: true`; every property is required and optional values are nullable.

The task-ID registries (`rsi.science.tool_registry`, `rsi.zeopp.registry`) and their v1 manifests are unchanged and stay the interface for frozen experiments.

## Division of control

| Model | Controller (`configs/science-hub.json`) |
|---|---|
| Which structure, method and parameters | Structure library, GCMC force-field library, OpenMM system library |
| How much of the budget each call spends (CPUs x wall) | Total budget (HPC session), call limit, per-tool bounds |
| — | Engine binaries, CP2K basis/potential defaults, GPAW occupations/SCF defaults, GCMC blocks and cutoff, backend (local/Slurm) |

Guidance is built into the catalog: tool descriptions state what each result does and does not establish, structures list their atom count and elements, unavailable tools are listed with the reason, and a short `guidance` list states the budget rules (screen cheaply first, do not compare across force fields or functionals, read convergence fields).

## Call lifecycle

1. The arguments are checked against the strict schema and the configured bounds, then against the library and engine validators (`validate_recipe`, `gcmc.validate_config`/`validate_force_field`, `dft.validate_config`). Invalid calls return `ok: false` and create no folder, ledger entry or job.
2. Registered files and tool sources are rehashed; any change since the session started blocks the call.
3. A call ID is reserved in `hub-ledger.json` (call limit enforced under the ledger lock), the structure is copied into `call-NNNN/inputs/`, and the tool's own prepare step freezes inputs.
4. Execution goes through `rsi.hpc.Scheduler` (`<session>/hpc/hpc-ledger.json`). The scheduler reserves `cpus x wall` before launch and refuses work beyond the session budget.
5. The envelope records the exact arguments, results, evidence hashes, the HPC jobs inside the call folder, what they were charged and the remaining budget. It is written to `call-NNNN/tool-result.json` and its hash to the hub ledger.

Changing the hub config, HPC config, a manifest, a library file or any tool source refuses to reopen an existing session, so a budget cannot be reset mid-session.

## Engine notes

- **DFT** returns energy, maximum free force, SCF and force convergence and steps; positions, forces and the cell stay in the evidence. Restarts are disabled (`max_restarts: 0`); a timeout is an error whose checkpoint remains in the run directory. CP2K runs at the Gamma point; GPAW uses plane waves with the chosen cutoff. The default config enables `emt`, `gpaw` and `cp2k`; only installed backends appear in the enum.
- **ORCA** is molecular. Only structures without periodicity appear in its enum, and `dft_calculation` now lists only structures with a full 3D cell. Structures report `periodic` in the catalog. Functionals and basis sets come from the controller list, which is checked against Crystal-Builder's ORCA 6.1 catalogue. Spin parity, 3c-with-basis and double-dispersion choices are refused before a call is spent. The tool is unavailable unless `RSI_ORCA` names a real ORCA binary.
- **GCMC** derives seeds `101, 202, ...`, uses the engine equation of state for fugacity, 5 blocks and a 12 Angstrom cutoff, and refuses force fields that do not type every element. Jobs run sequentially through the scheduler.
- **OpenMM** systems are controller-built serialized Systems. The default catalog is empty because no MOF force field is frozen, so the tool is listed as unavailable with that reason.
- **Zeo++** derives the channel radius from the probe radius and maps `fast`/`standard`/`fine` to sample counts; `standard` equals the validated recipes.

## Usage

```bash
python -m rsi.science.hub list --session runs/science-hub-default
python -m rsi.science.hub budget --session runs/science-hub-default
python -m rsi.science.hub call --session runs/science-hub-default --tool zeopp_pore_geometry --arguments '{"structure_id": "hkust1", "probe_radius_angstrom": 1.86, "psd_probe_radius_angstrom": 1.2, "sampling": "fast", "high_accuracy": true, "block_pockets": true, "grid_crosscheck_spacing_angstrom": null, "timeout_seconds": 600}'
```

[The science agent](SCIENCE_AGENT.md) implements this loop. In Python, an agent controller passes `hub.catalog()["tools"]` as the tool list and forwards each tool call to `hub.call(name, arguments)`; the returned envelope is the tool result.

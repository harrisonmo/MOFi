# Methane uptake and working capacity with GCMC

The optional `rsi.gcmc` adapter runs native RASPA3 (`raspalib==3.1.0`) grand
canonical Monte Carlo with methane insertion/deletion, translation and
reinsertion. Each pressure/seed job runs in its own bounded process. This is a
new scientific calculation path; it does not change the frozen M7/M11 tools or
claim an RSI improvement.

## Install and run

After installing a compatible WSL RASPA3 runtime, use
`scripts\raspa3-python-wsl.cmd -m rsi.gcmc ...` from the repository root.
The separate CLI is `scripts\raspa3-wsl.cmd`. Versions, installation details
and bounded native checks are in RASPA3 installation.

From the repository root, install the `gcmc` extra of the uv project
(raspalib wheels require Python 3.12; see [Python environment](ENVIRONMENT.md)):

```bash
uv sync --extra gcmc
uv run --no-sync python -m rsi.gcmc prepare \
  --config configs/methane-gcmc-example.json --output runs/methane-mfi-01
uv run --no-sync python -m rsi.gcmc run --run runs/methane-mfi-01
```

The prepare command
performs no GCMC calculation and requires ASE but not the native engine. The run
command checks all frozen input/source hashes, the engine version and finite
budget before launching jobs. It rejects already attempted directories. Use a
fresh directory for a separately budgeted new attempt, preserving prior costs.

The shipped example is **MFI zeolite**, not a MOF. It uses parameters and a CIF
from the official RASPA3 methane/MFI example; see `examples/gcmc/RASPA_LICENSE.md`
and the references below. Its cycle counts are a starting configuration, not a
convergence guarantee.

## Use a Crystal-Builder structure

Export a prepared, solvent-free, ordered periodic CIF from Crystal-Builder and
set `structure` in a copy of the config. Relative paths resolve against the
config file's directory. Record the originating Crystal-Builder commit with
your scientific experiment; manifests record the local dependency commit if the
separate `Crystal-Builder/` checkout is present.

Supply your own force-field JSON with a name, provenance source, explicit
framework LJ types, methane LJ parameters, shifted-potential setting and tail
correction setting. Epsilon is in kelvin and sigma in angstrom. Use either:

- `element_types`: an explicit element-to-type mapping, appropriate only when
  one LJ type per element is scientifically justified; or
- `site_types`: one type name for each atom in the ASE-expanded P1 unit cell,
  allowing multiple types of an element. The prepared `structure.json` records
  the exact expanded positions, symbols and assignments for inspection.

There is no automatic assignment of UFF types or MOF-metal parameters. Do not
apply the MFI force field to arbitrary MOFs. This adapter uses neutral single-site
methane and a rigid, uncharged LJ framework with Lorentz-Berthelot mixing; it
does not call the MACE potential. The CIF reader rejects partial occupancy,
disorder, multiple structures, missing types and nonperiodic structures. Read
any preserved ASE symmetry warnings and inspect the expanded structure before
using a result scientifically.

The supercell is replicated using perpendicular cell widths, including for
triclinic structures, so each width is at least twice the cutoff. The native
volume and dry-framework mass are checked against this expansion. An explicit
atom limit prevents unexpectedly large cells.

## Registered scientific tool

Tool registration is in `tools/science/methane-gcmc-v1.json`. The controller loads
`configs/scientific-tools.json`, which binds approved task IDs to their configs
and declares a cumulative session allocation. A model receives a function schema
and approved task catalog; it submits only a `task_id`.

```bash
uv run --no-sync python -m rsi.science.tools list
uv run --no-sync python -m rsi.science.tools call \
  --session runs/gcmc-agent-session-01 \
  --tool methane_gcmc_capacity \
  --arguments '{"task_id":"methane-mfi-example"}'
```

The second command executes the registered example's full budget, unlike `list`.
It uses the same native backend as the standalone GCMC command. In PowerShell,
pass the equivalent JSON argument using its shell quoting rules. A Python agent
controller can directly route calls:

```python
from rsi.science.tool_registry import ScientificToolRegistry

registry = ScientificToolRegistry(
    "configs/scientific-tools.json", "runs/gcmc-agent-session-01"
)
definitions = registry.catalog()["tools"]
result = registry.call("methane_gcmc_capacity", {"task_id": "methane-mfi-example"})
```

`definitions` contains function descriptions with strict JSON argument schemas.
The host controller supplies them to its agent and dispatches selected calls to
`registry.call`. Frozen M7/M11 planners retain their existing action catalogs;
this registration is available for new experiments with an explicit tool budget.

Results use a stable envelope: `ok`, `tool`, `contract`, `task_id`, `call_id`,
`data`, `evidence`, `cost` and `error`. Successful data includes the isotherm,
working capacity, per-seed statistics and convergence flag. Evidence links to
the native run, hashes and saved capacity file. Failure envelopes retain the
native manifest when present and distinguish completed work from interrupted
jobs with unknown actual cycles.

The default allocation permits one call, 90000 total reserved cycles and 3600
reserved wall seconds. A call reserves its complete initialization+production
cycle count and wall ceiling before preparation, and failed attempts consume
that reservation. The ledger persists across registry instances and CLI calls;
editing registration or increasing limits cannot reset an existing session.
One exclusive lock covers the entire call, and a stale lock blocks reuse after
a process crash. Preserve unresolved sessions for audit. A new session is a new
controller-authorized experimental allocation, not an automatic retry.

The input config, CIF, force field, registry implementation, native adapter and
tool descriptor are fingerprinted. Models cannot supply paths, engine options
or new budgets through this tool's arguments. To admit another MOF, the
controller registers a separately reviewed config with suitable force-field
parameters and a declared session allocation.

Registered-dispatch validation is described in the tool protocol
and the integration report.

## Conditions and output

The example requests absolute methane uptake at **298 K, 5.8 and 65 bar** with
three seeds. Add unique positive pressures to obtain more isotherm points.
`fugacity_coefficients: null` delegates fugacity calculation to the RASPA3 EOS
using the methane critical constants. An explicit coefficient list must match
the pressure array. Setting coefficients to one is an explicit ideal-gas
approximation, not the default at 65 bar.

`capacity.json` contains per-seed results, mean absolute uptake in mmol/g of dry
framework, and:

```text
working capacity = absolute uptake(loading pressure)
                 - absolute uptake(unloading pressure)
```

For native mean molecule count N and dry supercell mass M in atomic mass units,
`q = 1000 * N / M` mmol/g. Equivalently, 1 mol/kg = 1 mmol/g. Counts and mass must
refer to the same replicated cell. The reported native block error is retained
under its own name. A separate standard error across replicate seeds is computed
only when at least two seeds exist; single-seed uncertainty is null. Negative
working capacities are retained as diagnostics rather than clipped.

The run directory contains frozen CIF/config/force field/geometry, source and
dependency hashes, per-job reservations/status/cycle allocations, engine logs,
native RASPA output and raw molecule counts. Worker failure or timeout stops
further jobs and prevents a complete capacity report. Actual cycles remain
unknown for incomplete workers, rather than becoming zero cost. Native runtime
limits include subprocess startup and simulation; process descendants are killed
on timeout. All these generated files remain ignored under `runs/`.

Results are **absolute** uptake. The adapter does not report the native engine's
excess loading because an admitted accessible pore volume/void-fraction protocol
has not been supplied. It also does not automatically block geometrically
inaccessible pockets. Sampling convergence, independent-seed agreement, cutoff
and force-field sensitivity, and comparison with reference isotherms remain
necessary for scientific capacity claims. An unconverged short run is still
marked `convergence_established: false`.

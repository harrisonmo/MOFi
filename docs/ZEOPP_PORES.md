# Zeo++ pore geometry tool

`rsi.zeopp` is a bounded, reproducible pore geometry analysis of one rigid
framework with Zeo++ (`network`, Willems et al. 2012). It is the scientific
environment's geometry descriptor and GCMC pre-screen: it reports how large the
pores are, whether a probe can cross the crystal, how much surface and volume a
probe sees, and which pockets a GCMC run must block. It does **not** predict
adsorption.

Crystal-Builder already exposes Zeo++ in its GUI/agent session (`zeopp.*`
actions). This adapter adds what the RSI controller needs on top of it:
frozen recipes, a single shared decomposition, structured unit-bearing output,
an independent cross-check, caution flags, evidence hashes and a persistent
per-session budget.

## Calculation

One controller-owned recipe (`configs/zeopp/*.json`) fixes the CIF, probe,
channel radius, PSD probe, radii table, `-ha`, Monte Carlo sample counts,
pocket blocking, grid cross-check spacing, timeout and unit-cell atom limit.
Unknown keys or values outside the admitted ranges are refused.

`prepare` reads the CIF with Crystal-Builder, refuses partial occupancy (Zeo++
cannot express half an atom), enforces the atom limit, and writes a P1 CSSR so
that Zeo++'s own CIF reader never interprets symmetry or labels. Inputs, source
hashes and both repository commits are recorded in `manifest.json`.

`run` re-verifies the frozen inputs, then makes **one** `network` invocation:

```text
network -ha -res pores.res -chan Rc pores.chan -sa Rc Rp Nsa pores.sa \
        -volpo Rc Rp Nvol pores.vol -psd Rpsd Rpsd Npsd pores.psd_histo \
        -block Rp Nblock pores.block structure.cssr
```

All quantities therefore share one Voronoi decomposition, one radii table and
one accuracy setting. The binary path and SHA-256, argv, exit status, elapsed
time and output hashes are written to `manifest.json`; the log is `zeopp.log`.
Timeouts kill the process tree. Outputs are parsed with Crystal-Builder's tested
readers (`xtal.analysis.porosity`) into `pores.json`:

| Field | Meaning |
|---|---|
| `diameters_angstrom` | D_i (largest included sphere), D_f (largest free sphere), D_if |
| `channels` | Percolating channel count, per-channel dimensionality (1/2/3) and diameters |
| `surface_area` | Accessible/non-accessible area in m²/g, m²/cm³ and Å² |
| `pore_volume` | Probe-occupiable (`-volpo`) accessible/non-accessible volume in cm³/g and cell fraction |
| `pore_size_distribution` | Mode and mean diameter at the PSD probe; histogram file |
| `blocking` | Number of RASPA blocking spheres for inaccessible pockets (`pores.block`) |
| `grid_crosscheck` | Independent grid ASA/POAV and agreement (below) |
| `flags` | Cautions for downstream agents (below) |

Only Zeo++'s built-in radii are admitted in v1: these are the radii of the
published Zeo++ numbers and of Crystal-Builder's recorded reference values.

## Independent cross-check

With `grid_crosscheck` set, the same CIF is measured by Crystal-Builder's
distance-grid method (`xtal.analysis.voids`) at the same probe and radii. This
is a different algorithm, so agreement is evidence that the Zeo++ run and input
conversion are sound; it never replaces the Zeo++ value. Limits follow
Crystal-Builder's recorded Zeo++ comparison: accessible area within 2 %
relative, occupiable volume within 0.03 of the cell (the grid reads slightly
above `-volpo`).

## Flags

| Flag | Meaning |
|---|---|
| `NO_PERCOLATING_CHANNEL` | No channel admits the probe; uptake would require pocket filling only |
| `FREE_SPHERE_BELOW_PROBE_DIAMETER` | D_f is smaller than the channel-probe diameter |
| `INACCESSIBLE_POCKETS_PRESENT` | Non-accessible area or volume exists; GCMC must block it |
| `POCKETS_NOT_BLOCKED` | Pockets exist but the recipe did not request `-block` |
| `GRID_CROSSCHECK_DISAGREES` | Grid and Zeo++ disagree beyond the limits |
| `WINDOW_NEAR_PROBE_SIZE` | A window is within half a grid step of the probe size; accessibility is borderline |
| `GRID_ACCESSIBILITY_DISAGREES` | Grid and Zeo++ disagree on whether any channel percolates |

## Registered tool

The manifest is `tools/science/zeopp-pores-v1.json` (`zeopp_pore_geometry`, contract
`zeopp-pore-geometry-v1`). The registration `configs/zeopp-tools.json` binds
approved task IDs to recipes and a session allocation (`max_calls`,
`max_reserved_wall_seconds`; each call reserves its recipe timeout). A model
submits only `task_id`. The registry binds config, manifest, source and input
hashes; refuses tampering, session reset, concurrent use and unresolved prior
calls; and writes `ledger.json` and per-call `tool-result.json`.

It follows `rsi.science.tool_registry` but is a separate class so that the
GCMC registry is unchanged. Merging both into one multi-tool catalog is a later
controller change.

```bash
python -m rsi.zeopp list
python -m rsi.zeopp call --session runs/zeopp-session-01 --arguments '{"task_id":"hkust1-n2"}'
python -m rsi.zeopp prepare --recipe configs/zeopp/hkust1-n2.json --run runs/zeopp-hkust1-01
python -m rsi.zeopp run --run runs/zeopp-hkust1-01
```

Use the Crystal-Builder environment (`.venv\Scripts\python.exe`
on Windows). The binary is found through `XTAL_ZEOPP` (same variable as
Crystal-Builder), then `PATH`.

On this Windows machine Zeo++ runs in WSL (from PowerShell, not Git Bash, which
rewrites the Linux path):

```powershell
$env:XTAL_ZEOPP = "$PWD\scripts\zeopp-wsl.cmd"
$env:ZEOPP_WSL_BINARY = "/path/to/zeopp/network"
```

The shipped tasks are HKUST-1, ZIF-8, UiO-66 and MOF-5 with an N2 probe
(1.86 Å). Crystal-Builder's `tests/data/zeopp_reference.json` records Zeo++ 0.3
values for these structures at the same settings, which makes them the
acceptance reference for a newly installed binary (ZIF-8 is the pocket-only
case: all area non-accessible to N2).

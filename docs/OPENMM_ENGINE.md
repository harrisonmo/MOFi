# OpenMM minimization broker


## Design

OpenMM runs as a controller-side science engine, not as a candidate tool. `tools/` manifests stay a side-effect-free expression DSL; `rsi/tool_runtime.py`, `rsi/tool_validation.py`, `rsi/tool_evidence.py` and `tools/q2/*.json` are unchanged, so recorded Q2 frozen-file hashes remain valid.

- `rsi/openmm_engine.py` `minimize(system, positions, ...)` takes a ready-built `openmm.System`, runs `LocalEnergyMinimizer`, and returns a Crystal-Builder-shaped event in native kJ/mol and kJ/mol/nm.
- `from_openmm_event` projects that event onto the existing `uff-result-audit-v1` input, so the frozen `tools/q2/manual-audit-v1.json` audits OpenMM results unchanged (unit factor 1).
- Events use verb `minimize` with `engine: openmm`. `from_uff_event` requires `optimize`, so it rejects OpenMM events instead of labelling kJ values as kcal.
- `converged` is an explicit broker rule, `max_atom_force <= tolerance`, recorded in the event. OpenMM 8 applies its tolerance to the RMS force, so its internal stop is not reused as evidence. Massless particles and virtual sites never move, so they are excluded from the rule; `num_free_particles` records how many count.

## Calling contract

- Invalid requests raise `ValueError` before any compute: non-positive or non-finite `tolerance`, `max_iterations` < 1 (OpenMM treats 0 as unlimited), unknown platform, `threads` off the CPU platform, position count or shape mismatch, non-finite coordinates, box vectors for a non-periodic system, and constrained systems (constraint forces never vanish, so the max-force rule is undefined).
- `max_wall_seconds` is enforced per L-BFGS iteration through an OpenMM `MinimizationReporter`; the clock starts at minimization, not at Context creation. A stopped run reports `stopped_by_wall_time: true` and its real `converged` value.
- Numerical failures (minimizer exceptions, non-finite energy or force) return `result.ok: false` with an error; `from_openmm_event` rejects them.
- Every event records `system_sha256` (serialized System XML), particle counts, periodicity, platform properties, `iterations` and `elapsed_seconds`. `box_vectors` (nm) override the System default for periodic systems; `return_positions` adds final `positions_nm`.
- File interface: `minimize_files(system_xml, positions_json)` reads a serialized System and `{"positions_nm": [...], "box_vectors_nm": [...]}` and binds both input SHA-256 values into `action.inputs`.
- `run_isolated(..., output, timeout_seconds)` runs `python -m rsi.openmm_engine` in a child process with a hard timeout, defaults `max_wall_seconds` to 80% of it, writes the event atomically, refuses to overwrite existing evidence, and returns `Timeout` or `ChildFailed` events instead of raising.

```bash
python -m rsi.openmm_engine --system system.xml --positions positions.json --output event.json --options '{"platform": "CPU", "tolerance": 10.0, "max_iterations": 1000}'
```

## Limits

- The contract name still says UFF; energies from different engines or force fields are not comparable. Report results split by engine.
- OpenMM ships no UFF/UFF4MOF. A periodic MOF `System` builder is the open item and a separate decision.
- Wall-time checks run between iterations; a single very slow energy evaluation is bounded only by the `run_isolated` hard timeout.

Install with `uv sync --extra openmm` ([Python environment](ENVIRONMENT.md)).

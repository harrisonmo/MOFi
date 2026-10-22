"""Run one native RASPA3 pressure/seed job in a disposable process."""

from __future__ import annotations

import math
from pathlib import Path
import sys

from .core import ENGINE_VERSION, native_statistics, read_json, sha256, write_json


def simulate(output, job_id):
    import raspalib as r
    from ase.data import atomic_masses, atomic_numbers

    if r.__version__ != ENGINE_VERSION:
        raise ValueError("Unsupported native RASPA3 version")
    config = read_json(output / "config.json")
    ff = read_json(output / "force_field.json")
    structure = read_json(output / "structure.json")
    job = read_json(output / job_id / "job.json")
    type_names = list(ff["framework_types"])
    pseudo_atoms, lj = [], []
    for name in type_names:
        spec = ff["framework_types"][name]
        number = atomic_numbers[spec["element"]]
        pseudo_atoms.append(r.PseudoAtom(name, True, float(atomic_masses[number]), 0.0,
                                         atomic_number=number, source=ff["source"]))
        lj.append(r.VDWParameters(float(spec["epsilon_k"]), float(spec["sigma_angstrom"])))
    # A neutral one-site methane model; 16.04246 u matches the official RASPA example.
    pseudo_atoms.append(r.PseudoAtom("methane_gcmc_guest", False, 16.04246, 0.0, atomic_number=6,
                                     source=ff["source"]))
    lj.append(r.VDWParameters(float(ff["methane"]["epsilon_k"]), float(ff["methane"]["sigma_angstrom"])))
    force_field = r.ForceField(pseudo_atoms, lj,
                              mixing_rule=r.ForceField.MixingRule.LORENTZ_BERTHELOT,
                              cutoff_framework_vdw=float(config["cutoff_angstrom"]),
                              cutoff_molecule_vdw=float(config["cutoff_angstrom"]),
                              cutoff_coulomb=float(config["cutoff_angstrom"]),
                              shifted=ff["shifted"], tail_corrections=ff["tail_corrections"], use_charge=False)
    a, b, c, alpha, beta, gamma = structure["cell_parameters"]
    # Native RASPA3 takes radians here, not the degrees claimed in its Python stub.
    box = r.SimulationBox(a, b, c, math.radians(alpha), math.radians(beta), math.radians(gamma))
    atoms = [r.Atom(position, type=type_names.index(name)) for position, name in
             zip(structure["fractional_positions"], structure["type_assignments"])]
    framework = r.Framework(force_field, "prepared_host", box, 1, atoms, structure["unit_cells"])
    moves = r.MCMoveProbabilities(translation_probability=0.5, reinsertion_cbmc_probability=0.5,
                                  swap_probability=1.0)
    component = r.Component(force_field, "methane", 190.564, 4599200.0, 0.01142,
                            defined_atoms=[r.Atom([0., 0., 0.], type=len(type_names))],
                            particle_probabilities=moves, number_of_blocks=config["blocks"],
                            fugacity_coefficient=job["fugacity_coefficient"])
    system = r.System(force_field, external_temperature=float(config["temperature_k"]),
                      external_pressure=float(job["pressure_bar"]) * 100000.0,
                      framework_components=framework, components=[component],
                      initial_number_of_molecules=[0], number_of_blocks=config["blocks"])
    mc = r.MonteCarlo(number_of_cycles=config["production_cycles"],
                      number_of_initialization_cycles=config["initialization_cycles"],
                      systems=[system], random_seed=job["engine_seed"], number_of_blocks=config["blocks"],
                      print_every=max(config["production_cycles"], config["initialization_cycles"]),
                      output_to_files=True)
    native_system = mc.systems[0]
    mass = native_system.framework_mass()
    volume = native_system.simulation_box.volume
    if not math.isclose(mass, structure["framework_mass_u"], rel_tol=1e-7):
        raise ValueError("Native framework mass differs from the prepared P1 supercell")
    if not math.isclose(volume, structure["supercell_volume_angstrom3"], rel_tol=1e-7):
        raise ValueError("Native volume differs from the prepared P1 supercell")
    mc.run()
    mean, error = mc.systems[0].average_loadings.average_loading_number_of_molecules(0)
    if not all(math.isfinite(v) and v >= 0 for v in (mean, error)):
        raise ValueError("Invalid native loading statistics")
    native_files = list(Path("output").glob("*.s0.txt"))
    if len(native_files) != 1:
        raise ValueError("Expected a single native engine output file")
    native = native_statistics(native_files[0].read_text(encoding="utf-8"), config["production_cycles"], 1000 * mean / mass)
    if job["fugacity_coefficient"] is not None and not math.isclose(
            job["fugacity_coefficient"], native["fugacity_coefficient"], rel_tol=1e-8):
        raise ValueError("Native fugacity differs from the explicit reservoir setting")
    # N / framework mass in u is mol/g; multiplying by 1000 gives mmol/g.
    result = {"engine_version": r.__version__, "pressure_bar": job["pressure_bar"],
              "replicate_seed": job["replicate_seed"], "engine_seed": job["engine_seed"],
              "fugacity_coefficient_input": job["fugacity_coefficient"],
              "fugacity_coefficient_used": native["fugacity_coefficient"],
              "fugacity_mode": "raspa3_auto_eos" if job["fugacity_coefficient"] is None else "explicit_coefficient",
              "completed_production_cycles": config["production_cycles"],
              "mean_molecules_per_supercell": mean, "engine_reported_error_molecules": error,
              "framework_mass_u": mass, "supercell_volume_angstrom3": volume,
              "absolute_uptake_mmol_g": 1000 * mean / mass,
              "engine_reported_error_mmol_g": 1000 * error / mass,
              "engine_output_file": native_files[0].name,
              "engine_output_sha256": sha256(native_files[0])}
    write_json(output / job_id / "result.json", result)


if __name__ == "__main__":
    simulate(Path(sys.argv[1]).resolve(), sys.argv[2])

"""Bounded, reproducible Zeo++ pore geometry analysis of one rigid framework.

One controller-approved recipe fixes the structure, probe radii, sampling and
budget. A single ``network`` invocation shares one Voronoi decomposition across
``-res -chan -sa -volpo -psd -block`` so every reported number uses the same
radii and accuracy setting. Parsing reuses Crystal-Builder's tested readers.
An optional grid calculation from Crystal-Builder is an independent cross-check;
it is reported beside Zeo++, never substituted for it.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

from rsi.gcmc.core import exact_keys, integer, positive, read_json, sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
CRYSTAL_ROOT = ROOT / "Crystal-Builder"
ENV_VAR = "XTAL_ZEOPP"
RADII = ("zeo-default",)
INPUT = "structure.cssr"
OUTPUTS = {"res": "pores.res", "chan": "pores.chan", "sa": "pores.sa",
           "vol": "pores.vol", "psd": "pores.psd_histo", "block": "pores.block"}
RECIPE_KEYS = {"schema_version", "structure", "probe_label", "probe_radius_angstrom",
               "channel_radius_angstrom", "psd_probe_radius_angstrom", "radii",
               "high_accuracy", "samples", "block_pockets", "grid_crosscheck", "budget"}
SAMPLE_KEYS = {"surface_per_atom", "volume", "psd", "block"}
BUDGET_KEYS = {"timeout_seconds", "max_cell_atoms"}
# Agreement limits come from Crystal-Builder's recorded Zeo++ comparison
# (tests/test_voids.py): grid ASA within 2 %; grid POAV reads up to ~0.03 of the
# cell above -volpo on the reference MOFs.
CROSSCHECK_ASA_RELATIVE = 0.02
CROSSCHECK_POAV_ABSOLUTE = 0.03


def validate_recipe(recipe):
    """Reject anything outside the frozen v1 recipe before touching science."""
    exact_keys(recipe, RECIPE_KEYS, RECIPE_KEYS)
    if recipe["schema_version"] != 1:
        raise ValueError("Unsupported Zeo++ recipe schema")
    if not isinstance(recipe["structure"], str) or not recipe["structure"].lower().endswith(".cif"):
        raise ValueError("structure must be a CIF path relative to the recipe")
    if not isinstance(recipe["probe_label"], str) or not 0 < len(recipe["probe_label"]) <= 40:
        raise ValueError("probe_label must name the probe")
    for key in ("probe_radius_angstrom", "channel_radius_angstrom", "psd_probe_radius_angstrom"):
        if positive(recipe[key], key) > 5:
            raise ValueError(f"{key} must not exceed 5 Angstrom")
    if recipe["radii"] not in RADII:
        raise ValueError("Only Zeo++'s built-in radii are admitted in v1")
    if type(recipe["high_accuracy"]) is not bool or type(recipe["block_pockets"]) is not bool:
        raise ValueError("high_accuracy and block_pockets must be booleans")
    samples = recipe["samples"]
    exact_keys(samples, SAMPLE_KEYS, SAMPLE_KEYS)
    limits = {"surface_per_atom": (100, 20000), "volume": (1000, 500000),
              "psd": (1000, 500000), "block": (100, 100000)}
    for key, (low, high) in limits.items():
        if integer(samples[key], f"samples.{key}", low) > high:
            raise ValueError(f"samples.{key} must not exceed {high}")
    grid = recipe["grid_crosscheck"]
    if grid is not None:
        exact_keys(grid, {"spacing_angstrom"}, {"spacing_angstrom"})
        if not 0.2 <= positive(grid["spacing_angstrom"], "spacing_angstrom") <= 1.0:
            raise ValueError("grid spacing must lie in [0.2, 1.0] Angstrom")
    budget = recipe["budget"]
    exact_keys(budget, BUDGET_KEYS, BUDGET_KEYS)
    if positive(budget["timeout_seconds"], "timeout_seconds") > 7200:
        raise ValueError("timeout_seconds must not exceed 7200")
    integer(budget["max_cell_atoms"], "max_cell_atoms")
    return recipe


def binary():
    """The ``network`` executable the controller configured, or ``None``."""
    named = os.environ.get(ENV_VAR)
    if named:
        return Path(named).resolve() if Path(named).is_file() else None
    found = shutil.which("network")
    return Path(found).resolve() if found else None


def git_revision(directory):
    """Commit and dirty state; trusts only this directory so that a checkout
    owned by another Windows account still reports its revision."""
    trust = ["git", "-c", f"safe.directory={Path(directory).resolve().as_posix()}",
             "-C", str(directory)]
    try:
        commit = subprocess.check_output(trust + ["rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(trust + ["status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def load_structure(path):
    from xtal.analysis import porosity
    from xtal.core import p1
    from xtal.io.cif_reader import read_cif

    structure = read_cif(path)
    refusal = porosity.refuse(structure)
    if refusal:
        raise ValueError(f"Structure refused: {refusal}")
    return structure, p1.expand(structure).n_atoms


def argv(recipe, executable):
    """The one command line; every output name is fixed by this module."""
    probe = recipe["probe_radius_angstrom"]
    channel = recipe["channel_radius_angstrom"]
    samples = recipe["samples"]
    line = [str(executable)] + (["-ha"] if recipe["high_accuracy"] else [])
    line += ["-res", OUTPUTS["res"], "-chan", channel, OUTPUTS["chan"],
             "-sa", channel, probe, samples["surface_per_atom"], OUTPUTS["sa"],
             "-volpo", channel, probe, samples["volume"], OUTPUTS["vol"],
             "-psd", recipe["psd_probe_radius_angstrom"], recipe["psd_probe_radius_angstrom"],
             samples["psd"], OUTPUTS["psd"]]
    if recipe["block_pockets"]:
        line += ["-block", probe, samples["block"], OUTPUTS["block"]]
    return [str(token) for token in line + [INPUT]]


def prepare(recipe_path, output):
    """Freeze inputs and write the P1 CSSR; nothing is executed."""
    recipe_path = Path(recipe_path).resolve()
    recipe = validate_recipe(read_json(recipe_path))
    source = (recipe_path.parent / recipe["structure"]).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    structure, cell_atoms = load_structure(source)
    if cell_atoms > recipe["budget"]["max_cell_atoms"]:
        raise ValueError(f"Unit cell has {cell_atoms} atoms; budget allows "
                         f"{recipe['budget']['max_cell_atoms']}")
    from xtal.io.cssr import write_cssr

    shutil.copyfile(source, output / "source.cif")
    write_json(output / "recipe.json", recipe)
    write_cssr(structure, output / INPUT, name=source.stem)
    manifest = {"schema_version": 1, "status": "prepared", "cell_atoms": cell_atoms,
                "source_structure": str(source),
                "input_hashes": {name: sha256(output / name)
                                 for name in ("source.cif", "recipe.json", INPUT)},
                "code": {"rsi_zeopp": code_hashes(), "workspace": git_revision(ROOT),
                         "crystal_builder": git_revision(CRYSTAL_ROOT)},
                "engine": None, "argv": None, "elapsed_seconds": None, "error": None}
    write_json(output / "manifest.json", manifest)
    return manifest


def _stop(process):
    try:
        from rsi.gcmc.core import stop_process_tree
        stop_process_tree(process)
    except ImportError:
        process.kill()
        process.wait(timeout=10)


def execute(output, scheduler=None, backend=None, capacity_timeout=0.0):
    """Run Zeo++ once inside a prepared folder and write ``pores.json``.

    With an ``rsi.hpc.Scheduler`` the ``network`` call runs as a one-CPU scheduled job charged to
    that session's budget; parsing and the grid cross-check stay on the controller.
    """
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Only a freshly prepared run may execute")
    for name, checksum in manifest["input_hashes"].items():
        if sha256(output / name) != checksum:
            raise ValueError(f"Frozen input {name} changed before execution")
    recipe = validate_recipe(read_json(output / "recipe.json"))
    executable = binary()
    if executable is None:
        raise FileNotFoundError(f"Zeo++ network binary not found; set {ENV_VAR} or add it to PATH")
    line = argv(recipe, executable)
    manifest.update(status="running", argv=line,
                    engine={"path": str(executable), "sha256": sha256(executable)})
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        if scheduler is not None:
            from rsi.hpc import run_tool_job

            job = run_tool_job(scheduler, {"name": f"zeopp-{output.parent.name}-{output.name}"[:120],
                                           "argv": line, "cwd": str(output), "log": "zeopp.log",
                                           "resources": {"cpus": 1,
                                                         "wall_seconds": recipe["budget"]["timeout_seconds"]}},
                               backend, capacity_timeout)
            manifest["hpc_job"] = {key: job.get(key) for key in ("id", "backend", "native_id", "charged")}
            code = 0
        else:
            with open(output / "zeopp.log", "wb") as log:
                process = subprocess.Popen(line, cwd=output, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=recipe["budget"]["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    _stop(process)
                    raise TimeoutError(f"Zeo++ exceeded {recipe['budget']['timeout_seconds']} s")
                except BaseException:
                    _stop(process)
                    raise
        if code != 0:
            raise RuntimeError(f"Zeo++ exited with status {code}; see zeopp.log")
        expected = [k for k in OUTPUTS if k != "block" or recipe["block_pockets"]]
        missing = [OUTPUTS[k] for k in expected if not (output / OUTPUTS[k]).is_file()]
        if missing:
            raise RuntimeError(f"Zeo++ did not write {', '.join(missing)}; see zeopp.log")
        result = summarize(output, recipe)
        if recipe["grid_crosscheck"] is not None:
            result["grid_crosscheck"] = crosscheck(output, recipe, result)
        result["flags"] = flags(result, recipe)
        write_json(output / "pores.json", result)
        manifest.update(status="completed",
                        output_hashes={OUTPUTS[k]: sha256(output / OUTPUTS[k]) for k in expected}
                        | {"pores.json": sha256(output / "pores.json")})
        return result
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        manifest.update(status="interrupted" if interrupted else
                        "timeout" if isinstance(exc, TimeoutError) else "failed",
                        error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)


def _read(output, key):
    return (output / OUTPUTS[key]).read_text(encoding="utf-8", errors="replace")


def summarize(output, recipe):
    """Structured physical quantities with units in every key."""
    from xtal.analysis import porosity

    probe = recipe["probe_radius_angstrom"]
    diameters = porosity.parse_res(_read(output, "res"))
    channels = porosity.parse_chan(_read(output, "chan"))
    area = porosity.SurfaceArea.parse(_read(output, "sa"), probe)
    volume = porosity.Volume.parse(_read(output, "vol"), probe)
    psd = porosity.parse_psd(_read(output, "psd"), recipe["psd_probe_radius_angstrom"])
    if not volume.occupiable:
        raise ValueError("Volume output is not probe-occupiable (-volpo) data")
    result = {
        "probe": {"label": recipe["probe_label"], "radius_angstrom": probe,
                  "channel_radius_angstrom": recipe["channel_radius_angstrom"],
                  "radii": recipe["radii"], "high_accuracy": recipe["high_accuracy"]},
        "cell": {"volume_angstrom3": volume.volume or area.volume,
                 "density_g_cm3": volume.density or area.density},
        "diameters_angstrom": {"largest_included_sphere": diameters.included,
                               "largest_free_sphere": diameters.free,
                               "included_sphere_along_free_path": diameters.included_along_free},
        "channels": {"count": len(channels), "dimensionalities": [c.dimensionality for c in channels],
                     "summary": porosity.dimensionality(channels),
                     "per_channel_angstrom": [{"largest_included_sphere": c.included,
                                               "largest_free_sphere": c.free} for c in channels]},
        "surface_area": {"accessible_m2_g": area.accessible_per_gram,
                         "accessible_m2_cm3": area.accessible_per_volume,
                         "accessible_angstrom2": area.accessible_area,
                         "nonaccessible_m2_g": area.inaccessible_per_gram,
                         "nonaccessible_angstrom2": area.inaccessible_area,
                         "channel_count": area.channels, "pocket_count": area.pockets,
                         "samples_per_atom": recipe["samples"]["surface_per_atom"]},
        "pore_volume": {"definition": "probe-occupiable (Zeo++ -volpo)",
                        "accessible_cm3_g": volume.accessible_per_gram,
                        "accessible_fraction": volume.accessible_fraction,
                        "nonaccessible_cm3_g": volume.inaccessible_per_gram,
                        "nonaccessible_fraction": volume.inaccessible_per_gram * volume.density,
                        "samples": recipe["samples"]["volume"]},
        "pore_size_distribution": {"probe_radius_angstrom": recipe["psd_probe_radius_angstrom"],
                                   "mode_diameter_angstrom": psd.mode(),
                                   "mean_diameter_angstrom": psd.mean(),
                                   "accessible_sample_fraction": psd.accessible_fraction,
                                   "samples": recipe["samples"]["psd"],
                                   "histogram_file": OUTPUTS["psd"]},
        "blocking": None,
    }
    if recipe["block_pockets"]:
        result["blocking"] = {"probe_radius_angstrom": probe, "file": OUTPUTS["block"],
                              "sphere_count": block_count(_read(output, "block"))}
    return result


def block_count(text):
    """Zeo++ ``-block`` writes the sphere count on its first line."""
    first = text.split()
    if not first:
        raise ValueError("Empty Zeo++ block file")
    count = int(first[0])
    if count < 0:
        raise ValueError("Negative block sphere count")
    return count


def crosscheck(output, recipe, zeo):
    """Independent grid ASA and POAV at the same probe and Zeo++ radii."""
    from xtal.analysis import porosity, voids
    from xtal.analysis.grid import distance_grid
    from xtal.io.cif_reader import read_cif

    started = time.monotonic()
    structure = read_cif(output / "source.cif")
    probe = recipe["probe_radius_angstrom"]
    spacing = recipe["grid_crosscheck"]["spacing_angstrom"]
    field = distance_grid(structure, porosity.zeo_radius, spacing=spacing)
    found = voids.classify(structure, field, porosity.zeo_radius, probe)
    area = voids.surface_area(structure, found, porosity.zeo_radius)
    volume = voids.volume(structure, field, found, occupiable=True)
    zeo_asa = zeo["surface_area"]["accessible_angstrom2"]
    zeo_poav = zeo["pore_volume"]["accessible_fraction"]
    # Grid values may be numpy scalars; JSON evidence needs plain Python types.
    grid_asa, grid_poav = float(area.accessible_area), float(volume.accessible_fraction)
    asa_relative = (abs(grid_asa - zeo_asa) / zeo_asa if zeo_asa > 0
                    else (0.0 if grid_asa == 0 else None))
    poav_absolute = abs(grid_poav - zeo_poav)
    agrees = (asa_relative is not None and asa_relative <= CROSSCHECK_ASA_RELATIVE
              and poav_absolute <= CROSSCHECK_POAV_ABSOLUTE)
    return {"method": "Crystal-Builder distance grid (xtal.analysis.voids)",
            "spacing_angstrom": spacing, "channel_count": int(found.n_channels),
            "pocket_count": int(found.n_pockets), "borderline_window": bool(found.borderline),
            "accessible_angstrom2": grid_asa, "accessible_fraction": grid_poav,
            "asa_relative_difference": asa_relative, "poav_absolute_difference": poav_absolute,
            "limits": {"asa_relative": CROSSCHECK_ASA_RELATIVE,
                       "poav_absolute": CROSSCHECK_POAV_ABSOLUTE},
            "agrees": agrees, "elapsed_seconds": time.monotonic() - started}


def flags(result, recipe):
    """Machine-readable cautions a downstream agent must not overlook."""
    raised = []
    probe_diameter = 2 * recipe["channel_radius_angstrom"]
    if result["channels"]["count"] == 0:
        raised.append("NO_PERCOLATING_CHANNEL")
    if result["diameters_angstrom"]["largest_free_sphere"] < probe_diameter:
        raised.append("FREE_SPHERE_BELOW_PROBE_DIAMETER")
    if result["surface_area"]["nonaccessible_m2_g"] > 0 or result["pore_volume"]["nonaccessible_cm3_g"] > 0:
        raised.append("INACCESSIBLE_POCKETS_PRESENT")
    if result["blocking"] is None and "INACCESSIBLE_POCKETS_PRESENT" in raised:
        raised.append("POCKETS_NOT_BLOCKED")
    check = result.get("grid_crosscheck")
    if check is not None:
        if not check["agrees"]:
            raised.append("GRID_CROSSCHECK_DISAGREES")
        if check["borderline_window"]:
            raised.append("WINDOW_NEAR_PROBE_SIZE")
        if (check["channel_count"] > 0) != (result["channels"]["count"] > 0):
            raised.append("GRID_ACCESSIBILITY_DISAGREES")
    return raised

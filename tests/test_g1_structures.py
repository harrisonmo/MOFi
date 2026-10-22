"""Scientific representation and leakage checks for the G1 baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen", reason="G1 structural tests require the isolated GNN environment")
pytest.importorskip("torch", reason="G1 model tests require the isolated GNN environment")

from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from g1_features import grouped_split, linker_family, periodic_graph
from g1_model import PeriodicMPNN, pack


def crystal():
    coords = [[x, y, z] for x in [0, .5] for y in [0, .5] for z in [0, .5]]
    return Structure(Lattice.cubic(4), ["Zn", "C", "C", "O", "H", "C", "O", "H"], coords)


def test_periodic_images_include_self_images_but_not_zero_self_edges():
    graph = periodic_graph(crystal())
    same = graph["src"] == graph["dst"]
    assert same.any()
    assert (np.linalg.norm(graph["shifts"][same], axis=1) > 0).all()
    assert graph["distance"].min() > 0


def test_graph_distances_are_invariant_to_translation_and_permutation():
    a = crystal()
    b = a.copy()
    b.translate_sites(range(len(b)), [.37, -.2, .7], frac_coords=True, to_unit_cell=True)
    c = Structure(b.lattice, list(reversed(b.species)), list(reversed(b.frac_coords)))
    ga, gc = periodic_graph(a), periodic_graph(c)
    np.testing.assert_allclose(sorted(ga["distance"]), sorted(gc["distance"]), atol=1e-6)


def test_coincident_atoms_fail_closed():
    s = crystal()
    s.append("C", s[0].frac_coords)
    with pytest.raises(ValueError, match="coincident"):
        periodic_graph(s)


def test_partial_occupancy_fails_closed():
    s = crystal()
    s.replace(0, {"Zn": .5})
    with pytest.raises(ValueError, match="partial"):
        periodic_graph(s)


def test_structure_matcher_identifies_supercell():
    a = crystal()
    b = a * (2, 1, 1)
    assert StructureMatcher(primitive_cell=True, attempt_supercell=True).fit(a, b)


def test_family_ignores_site_order_translation_and_metal_identity():
    # An isolated finite organic chain is sufficient to test the fingerprint contract.
    s = Structure(Lattice.cubic(20), ["Zn", "C", "C", "O"], [[10, 10, 10], [1, 1, 1], [2.4, 1, 1], [3.6, 1, 1]], coords_are_cartesian=True)
    t = s.copy()
    t.replace(0, "Cu")
    t.translate_sites(range(len(t)), [.8, .6, .9], to_unit_cell=True)
    t = Structure(t.lattice, list(reversed(t.species)), list(reversed(t.frac_coords)))
    assert linker_family(s)[0] == linker_family(t)[0]
    assert linker_family(s, scaffold=True)[0] == linker_family(t, scaffold=True)[0]


def test_family_groups_never_cross_splits_and_labels_do_not_matter():
    rows = [{"id": str(i), "family": str(i // 3), "target": i} for i in range(90)]
    splits = grouped_split(rows)
    lookup = {x["id"]: x["family"] for x in rows}
    sets = [{lookup[i] for i in ids} for ids in splits.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[:i])
    assert grouped_split([{**r, "target": -r["target"]} for r in rows]) == splits


def test_model_is_invariant_to_permutation_and_supercell():
    import torch
    torch.manual_seed(17)
    model = PeriodicMPNN().eval()
    a = crystal()
    b = Structure(a.lattice, list(reversed(a.species)), list(reversed(a.frac_coords)))
    graphs = [periodic_graph(x) for x in [a, b, a * (2, 1, 1)]]
    with torch.no_grad():
        y = model(pack(graphs, "cpu")).numpy()
    np.testing.assert_allclose(y, np.full(3, y[0]), atol=1e-6)


def test_scaffold_groups_terminal_substitutions():
    ring = [[5 + 1.4 * np.cos(t), 5 + 1.4 * np.sin(t), 5] for t in np.arange(6) * np.pi / 3]
    s = Structure(Lattice.cubic(20), ["Zn"] + ["C"] * 6, [[15, 15, 15]] + ring, coords_are_cartesian=True)
    t = s.copy()
    t.append("F", [7.7, 5, 5], coords_are_cartesian=True)
    assert linker_family(s)[0] != linker_family(t)[0]
    assert linker_family(s, scaffold=True)[0] == linker_family(t, scaffold=True)[0]


def test_baseline_runner_rejects_heldout_access():
    from run_g1_baselines import load_split
    with pytest.raises(ValueError, match="Heldout"):
        load_split("heldout", {})


def test_scoring_has_correct_units_and_constant_rank_is_undefined():
    from run_g1_baselines import metrics
    result = metrics(np.array([1., 3.]), np.array([2., 2.]), ["a", "b"])
    assert result["mae"] == result["rmse"] == 1.
    assert result["spearman"] is None

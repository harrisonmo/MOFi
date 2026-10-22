"""Contracts for the G4 models."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from g3_model import SmoothPeriodicMPNN, fit_normalizers as fit_g3, pack as pack_g3
from g4_models import AVOGADRO_CM3, fit_normalizers, forward, global_features, make_model, pack


def graph(z, cell_length=9.0, seed=3):
    rng = np.random.default_rng(seed)
    src, dst = np.meshgrid(range(len(z)), range(len(z)))
    keep = src.ravel() != dst.ravel()
    src, dst = src.ravel()[keep], dst.ravel()[keep]
    return {"z": np.array(z), "src": src.astype(np.int64), "dst": dst.astype(np.int64),
            "distance": rng.uniform(1.0, 7.5, len(src)).astype(np.float32), "cell": np.eye(3) * cell_length}


GRAPHS = [graph([6, 8, 30]), graph([6, 6, 8, 29, 1], cell_length=11.0, seed=4)]
NORMALIZERS = fit_normalizers(GRAPHS)


def build(arm, envelope=True, seed=17):
    torch.manual_seed(seed)
    return make_model(arm, NORMALIZERS, envelope).eval()


def run(arm, model, graphs, geometry=None):
    geometry = torch.zeros(len(graphs), 14) if geometry is None else geometry
    with torch.no_grad():
        return forward(model, arm, pack(graphs, "cpu"), geometry)


def test_global_features_match_hand_computation():
    g = graph([6, 8, 30], cell_length=10.0)
    mass = 12.011 + 15.999 + 65.38
    number, density = global_features(g)
    assert math.isclose(number, math.log(3 / 1000.0), rel_tol=1e-12)
    assert math.isclose(density, math.log(mass / 1000.0 * AVOGADRO_CM3), rel_tol=1e-6)


def test_normalizers_match_g3_for_the_shared_density_statistic():
    g3 = fit_g3(GRAPHS)
    assert NORMALIZERS["avg_degree"] == g3["avg_degree"]
    assert math.isclose(NORMALIZERS["global_mean"][0], g3["density_mean"], rel_tol=1e-12)
    assert math.isclose(NORMALIZERS["global_scale"][0], g3["density_scale"], rel_tol=1e-12)


def test_g3_arm_reproduces_the_g3_model_exactly():
    g3 = fit_g3(GRAPHS)
    torch.manual_seed(17)
    reference = SmoothPeriodicMPNN(**g3).eval()
    with torch.no_grad():
        expected = reference(pack_g3(GRAPHS, "cpu"))
    torch.testing.assert_close(run("g3", build("g3"), GRAPHS), expected, atol=1e-6, rtol=1e-6)


def test_g3_hard_arm_matches_the_best_ablation_cell():
    from g3_ablation_model import CELLS, AblationMPNN
    g3 = fit_g3(GRAPHS)
    torch.manual_seed(17)
    reference = AblationMPNN(**CELLS["c8-hard-sum"], **g3).eval()
    with torch.no_grad():
        expected = reference(pack_g3(GRAPHS, "cpu"))
    torch.testing.assert_close(run("g3_hard", build("g3_hard"), GRAPHS), expected, atol=1e-6, rtol=1e-6)


def test_hybrid_uses_geometry_and_has_expected_parameters():
    model = build("g4_hybrid")
    # G3 encoder without its 65 -> 64 readout layer, plus a 79 -> 64 -> 1 readout.
    assert sum(p.numel() for p in model.parameters()) == 81345
    geometry = torch.zeros(2, 14)
    shifted = geometry.clone(); shifted[:, 3] = 1.0
    assert not torch.allclose(run("g4_hybrid", model, GRAPHS, geometry), run("g4_hybrid", model, GRAPHS, shifted))


def test_graph_arm_sees_mass_density_beyond_number_density():
    light, heavy = graph([6, 8, 30]), graph([6, 8, 48])
    model = build("g4_graph")
    model.embedding.weight.data[48] = model.embedding.weight.data[30]
    first, second = run("g4_graph", model, [light, heavy])
    assert abs(first - second) > 1e-6


def test_graph_arm_volume_pooling_responds_to_cell_volume():
    model = build("g4_graph")
    small, large = graph([6, 8, 30], cell_length=9.0), graph([6, 8, 30], cell_length=12.0)
    first, second = run("g4_graph", model, [small, large])
    assert abs(first - second) > 1e-6


def test_graph_arm_envelope_flag_changes_messages():
    with_envelope, hard = run("g4_graph", build("g4_graph", True), GRAPHS), run("g4_graph", build("g4_graph", False), GRAPHS)
    assert not torch.allclose(with_envelope, hard)


def test_predictions_are_isolated_within_a_batch():
    for arm in ["g3", "g3_hard", "g4_hybrid", "g4_graph"]:
        model = build(arm)
        together = run(arm, model, GRAPHS)
        alone = torch.cat([run(arm, model, [g]) for g in GRAPHS])
        torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-5)

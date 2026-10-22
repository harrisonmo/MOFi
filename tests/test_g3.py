"""Contracts for the G3 smooth-cutoff, count-aware periodic encoder."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from g1_model import PeriodicMPNN, pack as pack_g1
from g3_model import SmoothPeriodicMPNN, cosine_envelope, fit_normalizers, pack

CELL = np.eye(3) * 10.0


def graph(z, edges, cell=CELL):
    src, dst, distance = zip(*edges) if edges else ((), (), ())
    return {"z": np.array(z), "src": np.array(src, dtype=np.int64), "dst": np.array(dst, dtype=np.int64),
            "distance": np.array(distance, dtype=np.float32), "cell": cell}


def model(seed=17):
    torch.manual_seed(seed)
    return SmoothPeriodicMPNN(avg_degree=2.0, density_mean=-6.0, density_scale=1.0).eval()


def predict(net, graphs):
    with torch.no_grad():
        return net(pack(graphs, "cpu"))


def test_envelope_has_zero_value_and_slope_at_cutoff():
    d = torch.tensor([0.0, 7.999, 8.0, 9.0], requires_grad=True)
    value = cosine_envelope(d, 8.0)
    value.sum().backward()
    assert value[0] == 1 and value[2] == 0 and value[3] == 0
    assert abs(value[1]) < 1e-6 and abs(d.grad[1]) < 1e-3


def test_output_is_continuous_when_an_edge_crosses_the_cutoff():
    base = [(0, 1, 2.0), (1, 0, 2.0), (1, 2, 2.0), (2, 1, 2.0)]
    far = [(0, 2, 8.0 - 1e-4), (2, 0, 8.0 - 1e-4)]
    net = model()
    with_edge, without_edge = predict(net, [graph([6, 30, 8], base + far), graph([6, 30, 8], base)])
    torch.testing.assert_close(with_edge, without_edge, atol=1e-6, rtol=0)


def test_g1_mean_aggregation_was_discontinuous_at_its_cutoff():
    base = [(0, 1, 2.0), (1, 0, 2.0), (1, 2, 2.0), (2, 1, 2.0)]
    far = [(0, 2, 5.0 - 1e-4), (2, 0, 5.0 - 1e-4)]
    torch.manual_seed(17)
    net = PeriodicMPNN().eval()
    with torch.no_grad():
        jump = net(pack_g1([graph([6, 30, 8], base + far), graph([6, 30, 8], base)], "cpu"))
    assert abs(jump[0] - jump[1]) > 1e-4


def test_neighbour_count_changes_prediction_unlike_mean_aggregation():
    single = graph([6, 6], [(0, 1, 1.5), (1, 0, 1.5)])
    double = graph([6, 6], [(0, 1, 1.5), (0, 1, 1.5), (1, 0, 1.5), (1, 0, 1.5)])
    torch.manual_seed(17)
    g1 = PeriodicMPNN().eval()
    with torch.no_grad():
        old = g1(pack_g1([single, double], "cpu"))
    torch.testing.assert_close(old[0], old[1])
    new = predict(model(), [single, double])
    assert abs(new[0] - new[1]) > 1e-4


def test_density_reaches_the_readout_and_receives_gradients():
    dense = graph([6, 6], [(0, 1, 1.5), (1, 0, 1.5)], cell=np.eye(3) * 5.0)
    sparse = graph([6, 6], [(0, 1, 1.5), (1, 0, 1.5)], cell=np.eye(3) * 20.0)
    net = model()
    out = predict(net, [dense, sparse])
    assert abs(out[0] - out[1]) > 1e-4
    net.train()
    net(pack([dense], "cpu")).sum().backward()
    assert net.readout[0].weight.grad[:, -1].abs().sum() > 0


def test_batching_matches_individual_predictions():
    a = graph([6, 30, 8], [(0, 1, 2.0), (1, 0, 2.0), (1, 2, 3.0), (2, 1, 3.0)])
    b = graph([6, 6], [(0, 1, 1.5), (1, 0, 1.5)], cell=np.eye(3) * 7.0)
    net = model()
    torch.testing.assert_close(predict(net, [a, b]), torch.cat([predict(net, [a]), predict(net, [b])]))


def test_normalizers_use_only_given_graphs_and_reject_invalid_values():
    a = graph([6, 6], [(0, 1, 1.5), (1, 0, 1.5)])
    stats = fit_normalizers([a, a])
    assert stats["avg_degree"] == 1.0 and stats["density_scale"] == 1.0
    assert stats["density_mean"] == pytest.approx(np.log(2 / 1000))
    with pytest.raises(ValueError):
        SmoothPeriodicMPNN(avg_degree=0.0, density_mean=0.0, density_scale=1.0)


def test_g3_graph_cutoff_and_scaled_edge_budget():
    pytest.importorskip("ase")
    pymatgen = pytest.importorskip("pymatgen.core")
    from g3_features import CUTOFF, MAX_EDGES, periodic_graph
    assert CUTOFF == 8.0 and MAX_EDGES == 204_800
    structure = pymatgen.Structure(np.eye(3) * 9.0, ["Zn", "C", "O", "H", "C", "O", "H", "C"],
                                   [[0, 0, 0], [.2, 0, 0], [0, .2, 0], [0, 0, .2],
                                    [.5, .5, .5], [.7, .5, .5], [.5, .7, .5], [.5, .5, .7]])
    result = periodic_graph(structure)
    assert result["distance"].max() < 8.0 and result["distance"].max() > 5.0
    with pytest.raises(ValueError, match="edge_budget"):
        periodic_graph(structure, max_edges=10)

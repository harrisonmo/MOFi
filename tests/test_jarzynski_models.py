"""Unexecuted model contracts for the synthetic harmonic graph benchmark."""
import math

import pytest

torch = pytest.importorskip("torch")

from rsi.jarzynski.models import FreeEnergyGNN, TransportGNN, interpolant_loss
from rsi.jarzynski.systems import make_graphs, take_graphs


def _transport():
    torch.manual_seed(101)
    model = TransportGNN(width=12, layers=2).double()
    # Exercise learned vector terms, not just the initial 0 and -x outputs.
    generator = torch.Generator().manual_seed(103)
    with torch.no_grad():
        for layer in (model.node_coefficients[-1], model.edge_coefficients[-1]):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator, dtype=torch.float64) * 0.05)
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator, dtype=torch.float64) * 0.05)
    return model


def _inputs():
    graphs = make_graphs(3, 4, 107)
    generator = torch.Generator().manual_seed(109)
    x0 = torch.randn((3, 4, 3), dtype=torch.float64, generator=generator)
    x1 = torch.randn((3, 4, 3), dtype=torch.float64, generator=generator)
    noise = torch.randn((3, 4, 3), dtype=torch.float64, generator=generator)
    t = torch.tensor([0.17, 0.48, 0.83], dtype=torch.float64)
    return graphs, x0, x1, noise, t


def _permute(graphs, permutation):
    return {
        key: (value.index_select(1, permutation).index_select(2, permutation)
              if key.startswith("k") else value.index_select(1, permutation)
              if key.startswith("a") else value)
        for key, value in graphs.items()
    }


@pytest.mark.parametrize("reflection", [False, True])
def test_transport_is_equivariant_under_rotation_and_reflection(reflection):
    model = _transport().eval()
    graphs, x, _, _, t = _inputs()
    angle = 0.43
    matrix = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, -1.0 if reflection else 1.0],
    ], dtype=torch.float64)
    velocity, score = model(x, t, graphs)
    rotated_velocity, rotated_score = model(x @ matrix.T, t, graphs)
    torch.testing.assert_close(rotated_velocity, velocity @ matrix.T, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(rotated_score, score @ matrix.T, rtol=1e-10, atol=1e-10)


def test_transport_is_equivariant_under_node_permutation():
    model = _transport().eval()
    graphs, x, _, _, t = _inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    velocity, score = model(x, t, graphs)
    permuted_velocity, permuted_score = model(x[:, permutation], t, _permute(graphs, permutation))
    torch.testing.assert_close(permuted_velocity, velocity[:, permutation], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(permuted_score, score[:, permutation], rtol=1e-10, atol=1e-10)


def test_scalar_free_energy_is_permutation_invariant_and_has_exact_offset():
    torch.manual_seed(113)
    model = FreeEnergyGNN(width=12, layers=2).double().eval()
    graphs, x, _, _, t = _inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    reference = model(graphs)
    torch.testing.assert_close(model(_permute(graphs, permutation)), reference, rtol=1e-10, atol=1e-10)
    modified = {key: value.clone() for key, value in graphs.items()}
    offset = torch.tensor([0.5, -1.2, 0.7], dtype=torch.float64)
    modified["c1"] += offset
    torch.testing.assert_close(model(modified), reference + offset, rtol=1e-10, atol=1e-10)
    transport = _transport().eval()
    for actual, expected in zip(transport(x, t, modified), transport(x, t, graphs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interpolant_loss_matches_fm_and_weighted_dsm_and_both_heads_receive_gradients():
    model = _transport()
    graphs, x0, x1, noise, t = _inputs()
    scale = 0.4
    time = t[:, None, None]
    gamma = scale * torch.sin(math.pi * time)
    derivative = scale * math.pi * torch.cos(math.pi * time)
    x = (1 - time) * x0 + time * x1 + gamma * noise
    velocity, score = model(x, t, graphs)
    flow = ((velocity - (x1 - x0 + derivative * noise)) ** 2).mean()
    dsm = ((gamma * score + noise) ** 2).mean()
    final_weight = model.node_coefficients[-1].weight
    flow_gradient, = torch.autograd.grad(flow, final_weight, retain_graph=True)
    score_gradient, = torch.autograd.grad(dsm, final_weight, retain_graph=True)
    assert bool(torch.isfinite(flow_gradient).all())
    assert bool(torch.isfinite(score_gradient).all())
    assert float(flow_gradient[0].abs().sum()) > 0
    assert float(score_gradient[1].abs().sum()) > 0
    torch.testing.assert_close(flow_gradient[1], torch.zeros_like(flow_gradient[1]))
    torch.testing.assert_close(score_gradient[0], torch.zeros_like(score_gradient[0]))
    total, parts = interpolant_loss(model, graphs, x0, x1, t, noise, gamma_scale=scale)
    torch.testing.assert_close(total, flow + dsm)
    torch.testing.assert_close(parts["flow"], flow.detach())
    torch.testing.assert_close(parts["score"], dsm.detach())
    total.backward()
    assert bool(torch.isfinite(final_weight.grad).all())
    assert bool((final_weight.grad.abs().sum(-1) > 0).all())


def test_models_do_not_mix_independent_graphs_in_a_batch():
    graphs, x, _, _, t = _inputs()
    transport = _transport().eval()
    torch.manual_seed(127)
    regressor = FreeEnergyGNN(width=12, layers=2).double().eval()
    full_velocity, full_score = transport(x, t, graphs)
    full_free_energy = regressor(graphs)
    for index in range(len(x)):
        one_graph = take_graphs(graphs, [index])
        velocity, score = transport(x[index:index + 1], t[index:index + 1], one_graph)
        torch.testing.assert_close(velocity, full_velocity[index:index + 1], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(score, full_score[index:index + 1], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(regressor(one_graph), full_free_energy[index:index + 1], rtol=1e-10, atol=1e-10)


def test_transport_accepts_scalar_time_without_cross_graph_broadcasting():
    model = _transport().eval()
    graphs, x, _, _, _ = _inputs()
    scalar_outputs = model(x, 0.3, graphs)
    vector_outputs = model(x, torch.full((len(x),), 0.3, dtype=torch.float64), graphs)
    for actual, expected in zip(scalar_outputs, vector_outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

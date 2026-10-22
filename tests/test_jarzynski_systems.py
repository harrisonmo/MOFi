"""Physical contracts for an analytical harmonic graph toy benchmark."""
import math

import pytest

torch = pytest.importorskip("torch")

from rsi.jarzynski.systems import (
    energy,
    energy_gradient,
    exact_delta_f,
    graph_hash,
    interpolated_score,
    make_graphs,
    precision,
    sample_endpoint,
    take_graphs,
)


def test_graphs_are_connected_spd_and_parameter_ranges_match_ood():
    for size, ood in ((4, False), (6, True)):
        graphs = make_graphs(30, size, seed=17, ood=ood)
        for endpoint in (0, 1):
            stiffness = precision(graphs, endpoint)
            assert bool((torch.linalg.eigvalsh(stiffness) > 0).all())
            torch.testing.assert_close(stiffness.sum(-1), graphs[f"a{endpoint}"])
        assert bool(((graphs["a0"] >= 0.6) & (graphs["a0"] <= 1.5)).all())
        assert bool((graphs["a1"].log() - graphs["a0"].log()).abs().le(0.7).all())
        edges = graphs["k0"] > 0
        assert bool(graphs["k0"][edges].ge(0.05).all())
        assert bool(graphs["k0"][edges].le(0.35).all())
        assert bool((graphs["k1"][edges] / graphs["k0"][edges]).log().abs().le(0.5).all())
        for node in range(size):
            assert bool(edges[:, node, (node + 1) % size].all())
    # The flag alone never shifts the distribution or random draw sequence.
    assert graph_hash(make_graphs(3, 4, 17)) == graph_hash(make_graphs(3, 4, 17, ood=True))


def test_exact_free_energy_matches_independent_diagonal_normalizer():
    graphs = make_graphs(3, 4, 29)
    graphs["k0"].zero_()
    graphs["k1"].zero_()
    beta, dimensions = 1.7, 3
    log_z = []
    for endpoint in (0, 1):
        a = graphs[f"a{endpoint}"]
        log_z.append(-beta * graphs[f"c{endpoint}"] + dimensions / 2 * (
            torch.log(2 * math.pi / (beta * a)).sum(dim=1)
        ))
    torch.testing.assert_close(exact_delta_f(graphs, dimensions, beta), -(log_z[1] - log_z[0]) / beta)
    # Physical energy does not acquire an unintended factor of beta.
    x = torch.ones((3, 4, 3), dtype=torch.float64)
    torch.testing.assert_close(energy(graphs, x, 1, beta), energy(graphs, x, 1))


def test_equilibrium_sampler_covariance_matches_inverse_precision():
    base = make_graphs(1, 3, 43)
    repeated = take_graphs(base, torch.zeros(30000, dtype=torch.long))
    beta = 1.6
    x = sample_endpoint(repeated, 1, torch.Generator().manual_seed(47), beta)
    # Independent Cartesian coordinates give 90000 independent Gaussian draws.
    samples = x.permute(0, 2, 1).reshape(-1, 3)
    assert bool(samples.mean(0).abs().lt(0.015).all())
    empirical = samples.T @ samples / len(samples)
    expected = torch.linalg.inv(precision(base, 1)[0]) / beta
    torch.testing.assert_close(empirical, expected, rtol=0.04, atol=0.008)


def test_analytical_energy_gradient_and_interpolated_score():
    graphs = make_graphs(5, 4, 59)
    x = sample_endpoint(graphs, 0, torch.Generator().manual_seed(61)).requires_grad_()
    for endpoint in (0, 1):
        automatic, = torch.autograd.grad(energy(graphs, x, endpoint).sum(), x)
        torch.testing.assert_close(energy_gradient(graphs, x, endpoint), automatic)
    time = torch.linspace(0, 1, len(x), dtype=x.dtype)
    beta = 0.7
    interpolated_energy = ((1 - time) * energy(graphs, x, 0) + time * energy(graphs, x, 1)).sum()
    automatic, = torch.autograd.grad(interpolated_energy, x)
    torch.testing.assert_close(interpolated_score(graphs, x, time, beta), -beta * automatic)
    torch.testing.assert_close(interpolated_score(graphs, x, 0, beta), -beta * energy_gradient(graphs, x, 0))


def test_hash_and_selection_are_deterministic_and_preserve_order():
    graphs = make_graphs(5, 4, 71)
    assert graph_hash(graphs) == graph_hash(make_graphs(5, 4, 71))
    assert graph_hash(graphs) != graph_hash(make_graphs(5, 4, 73))
    selected = take_graphs(graphs, [4, 0, 4])
    torch.testing.assert_close(selected["a1"], graphs["a1"][[4, 0, 4]])
    torch.testing.assert_close(take_graphs(graphs, slice(1, 3))["c1"], graphs["c1"][1:3])
    assert take_graphs(graphs, 2)["a0"].shape == (1, 4)
    assert graph_hash(graphs) != graph_hash(take_graphs(graphs, [4, 3, 2, 1, 0]))


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_invalid_beta_is_rejected(beta):
    graphs = make_graphs(2, 4, 79)
    x = torch.zeros((2, 4, 3), dtype=torch.float64)
    for operation in (
        lambda: energy(graphs, x, 0, beta),
        lambda: exact_delta_f(graphs, beta=beta),
        lambda: sample_endpoint(graphs, 0, torch.Generator(), beta),
        lambda: interpolated_score(graphs, x, 0.5, beta),
    ):
        with pytest.raises(ValueError, match="beta"):
            operation()


def test_invalid_physical_inputs_and_shapes_are_rejected():
    graphs = make_graphs(2, 4, 83)
    x = torch.zeros((2, 4, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="endpoint"):
        precision(graphs, 2)
    with pytest.raises(ValueError, match="Coordinates"):
        energy(graphs, x[:, :, :2], 0)
    with pytest.raises(ValueError, match="times"):
        interpolated_score(graphs, x, 1.1)
    with pytest.raises(ValueError, match="scalar"):
        interpolated_score(graphs, x, torch.zeros(3))
    with pytest.raises(ValueError, match="indices"):
        take_graphs(graphs, [-1])
    with pytest.raises(ValueError, match="integer"):
        take_graphs(graphs, [0.5])
    invalid = {key: value.clone() for key, value in graphs.items()}
    invalid["a0"][0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        precision(invalid, 0)
    invalid["a0"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        precision(invalid, 0)
    invalid = {key: value.clone() for key, value in graphs.items()}
    invalid["k0"][0, 0, 1] += 0.01
    with pytest.raises(ValueError, match="symmetric"):
        precision(invalid, 0)

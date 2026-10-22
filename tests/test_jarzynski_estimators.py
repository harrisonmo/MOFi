"""Independent numerical checks of FEAT work signs, kernels, and temperature."""

import math

import pytest

torch = pytest.importorskip("torch")

from rsi.jarzynski import estimators


@pytest.fixture
def harmonic_energy(monkeypatch):
    def energy(batch, x, endpoint, beta=1.0):
        # Physical endpoint energies deliberately do not multiply by beta.
        stiffness = batch.get(f"k{endpoint}", 1.0)
        offset = batch.get(f"c{endpoint}", 0.0)
        return 0.5 * stiffness * x.square().sum(dim=(1, 2)) + offset

    monkeypatch.setattr(estimators.systems, "energy", energy)
    return energy


def test_constant_work_sign_beta_and_batched_reduction():
    work = torch.tensor([[3.5] * 7, [-2.0] * 7], dtype=torch.float64)
    for beta in (0.5, 1.0, 3.0):
        torch.testing.assert_close(
            estimators.jarzynski(work, beta), torch.tensor([3.5, -2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            estimators.effective_sample_size(work, beta), torch.full((2,), 7.0, dtype=torch.float64)
        )


def test_extreme_log_weights_and_constant_offsets_are_stable():
    work = torch.tensor([-12000.0, -11999.0, -11998.0], dtype=torch.float64)
    expected = -12000.0 - math.log((1 + math.exp(-1) + math.exp(-2)) / 3)
    assert float(estimators.jarzynski(work)) == pytest.approx(expected)
    shifted = work + 1e8
    assert float(estimators.jarzynski(shifted) - estimators.jarzynski(work)) == pytest.approx(1e8)
    torch.testing.assert_close(
        estimators.effective_sample_size(shifted), estimators.effective_sample_size(work)
    )
    assert 1 <= float(estimators.effective_sample_size(work)) <= 3


def test_mean_work_is_an_upper_bound_for_each_finite_sample_estimate():
    work = torch.tensor([-2.0, 0.0, 1.0, 7.0])
    summary = estimators.work_summary(work, beta=1.7)
    assert summary["delta_f"] < summary["mean_work"]
    assert summary["mean_work"] == pytest.approx(1.5)
    assert summary["ess_fraction"] == pytest.approx(summary["ess"] / 4)
    assert summary["n_samples"] == 4


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_invalid_beta_is_rejected(beta):
    for function in (estimators.jarzynski, estimators.effective_sample_size, estimators.work_summary):
        with pytest.raises(ValueError, match="beta"):
            function(torch.ones(3), beta=beta)
    with pytest.raises(ValueError, match="beta"):
        estimators.simulate_forward({}, torch.zeros((1, 1, 1)), lambda *args: None,
                                    1, 0.1, generator=None, beta=beta)


@pytest.mark.parametrize("work", [[], [float("nan")], [0.0, float("inf")]])
def test_nonfinite_or_empty_work_is_never_filtered(work):
    with pytest.raises(ValueError):
        estimators.jarzynski(torch.tensor(work))


def test_one_step_matches_independent_gaussian_density_calculation(harmonic_energy):
    batch = {"k0": 1.3, "k1": 2.1, "c0": -0.4, "c1": 0.8}
    x0 = torch.tensor([[[0.7, -0.3], [-0.4, 0.1]]], dtype=torch.float64)
    original = x0.clone()
    beta, diffusion = 1.6, 0.12
    calls = []

    def drift(x, t, received_batch):
        assert received_batch is batch
        assert not torch.is_grad_enabled()
        calls.append(t.clone())
        time = t[:, None, None]
        return (0.3 + 0.1 * time) * x, -(0.8 + 0.2 * time) * x

    result = estimators.simulate_forward(
        batch, x0, drift, steps=1, diffusion=diffusion,
        generator=torch.Generator().manual_seed(29), beta=beta,
    )
    endpoint = result["endpoint"]
    forward_mean = x0 + (0.3 * x0 + diffusion * (-0.8 * x0))
    backward_mean = endpoint + (-0.4 * endpoint + diffusion * (-1.0 * endpoint))
    normal = torch.distributions.Normal
    forward_log = normal(forward_mean, math.sqrt(2 * diffusion)).log_prob(endpoint).sum()
    backward_log = normal(backward_mean, math.sqrt(2 * diffusion)).log_prob(x0).sum()
    delta_energy = harmonic_energy(batch, endpoint, 1) - harmonic_energy(batch, x0, 0)
    expected = delta_energy + (forward_log - backward_log) / beta
    torch.testing.assert_close(result["work"], expected)
    torch.testing.assert_close(result["dimensionless_work"], expected * beta)
    torch.testing.assert_close(result["log_kernel_ratio"], (forward_log - backward_log).reshape(1))
    torch.testing.assert_close(x0, original)
    assert [float(t.item()) for t in calls] == [0.0, 1.0]


def test_full_time_interval_and_energy_offset_are_exact(harmonic_energy):
    x0 = torch.tensor([[[0.3]], [[-0.8]]], dtype=torch.float64)
    observed = []

    def drift(x, t, batch):
        observed.append(float(t[0]))
        return torch.zeros_like(x), -0.7 * x

    options = dict(steps=4, diffusion=0.07, beta=2.0)
    base = estimators.simulate_forward(
        {}, x0, drift, generator=torch.Generator().manual_seed(71), **options
    )
    assert observed == [0.0, 0.25, 0.5, 0.75, 1.0]
    shifted = estimators.simulate_forward(
        {"c1": 5.25}, x0, drift, generator=torch.Generator().manual_seed(71), **options
    )
    torch.testing.assert_close(base["endpoint"], shifted["endpoint"])
    torch.testing.assert_close(shifted["work"] - base["work"], torch.full((2,), 5.25, dtype=torch.float64))
    assert float(estimators.jarzynski(shifted["work"], beta=2.0) -
                 estimators.jarzynski(base["work"], beta=2.0)) == pytest.approx(5.25)


def test_reversible_stationary_gaussian_kernel_has_constant_work(harmonic_energy):
    # An AR(1) Gaussian kernel y = a*x + sqrt(1-a^2)*z is exactly reversible
    # under N(0, 1).  Encode its mean and variance through the generic interface.
    # The drift score is deliberately imperfect; the discrete ratio still holds.
    a, steps = 0.8, 4
    diffusion = (1 - a * a) * steps / 2
    score_coefficient = (a - 1) * steps / diffusion
    x0 = torch.tensor([[[0.1]], [[2.3]], [[-1.7]]], dtype=torch.float64)

    def drift(x, t, batch):
        return torch.zeros_like(x), score_coefficient * x

    result = estimators.simulate_forward(
        {"c1": 2.7}, x0, drift, steps, diffusion,
        generator=torch.Generator().manual_seed(8),
    )
    torch.testing.assert_close(result["work"], torch.full((3,), 2.7, dtype=torch.float64))


def test_identity_energy_monte_carlo_jarzynski(harmonic_energy):
    # Euler Langevin is not exactly equilibrium-preserving.  The kernel ratio
    # must still recover the zero free-energy change without a Metropolis step.
    generator = torch.Generator().manual_seed(391)
    x0 = torch.randn((12000, 1, 1), dtype=torch.float64, generator=generator)

    def drift(x, t, batch):
        return torch.zeros_like(x), -x

    result = estimators.simulate_forward({}, x0, drift, 3, 0.45, generator)
    assert abs(float(estimators.jarzynski(result["work"]))) < 0.006
    assert bool((result["work"].abs() > 1e-5).any())


@pytest.mark.parametrize("diffusion", [0, -1, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_diffusion_is_rejected(diffusion):
    with pytest.raises(ValueError, match="diffusion"):
        estimators.simulate_forward({}, torch.zeros((1, 1, 1)), lambda *args: None,
                                    3, diffusion, generator=None)


def test_nonfinite_network_output_and_bad_shape_fail_closed(harmonic_energy):
    x0 = torch.zeros((1, 1, 1))
    for drift in (
        lambda x, t, batch: (torch.full_like(x, float("nan")), torch.zeros_like(x)),
        lambda x, t, batch: (torch.zeros(1), torch.zeros_like(x)),
    ):
        with pytest.raises(ValueError):
            estimators.simulate_forward({}, x0, drift, 2, 0.1, generator=None)


def test_real_system_energy_interface_supports_float64_work_from_float32_paths():
    batch = estimators.systems.make_graphs(3, 4, seed=15, dtype=torch.float32)
    generator = torch.Generator().manual_seed(89)
    x0 = estimators.systems.sample_endpoint(batch, 0, generator, beta=1.4)

    def drift(x, t, graph):
        return torch.zeros_like(x), estimators.systems.interpolated_score(graph, x, t, beta=1.4)

    result = estimators.simulate_forward(batch, x0, drift, 2, 0.05, generator, beta=1.4)
    assert result["endpoint"].dtype == torch.float32
    assert result["work"].dtype == torch.float64
    assert result["work"].shape == (3,)
    assert not result["work"].requires_grad
    assert bool(torch.isfinite(result["work"]).all())

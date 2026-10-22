"""Free-energy estimators and discrete FEAT forward-path importance weights.

The physical work convention is ``Delta F = F_1 - F_0``.  ``score`` means
``grad(log rho_t)`` and therefore already includes the inverse temperature.
The forward/backward Gaussian-kernel ratio implements Section 3.4 of
https://arxiv.org/abs/2504.11516v4.  It avoids continuous-time divergence
quadrature and remains an exact path-space identity at a finite step count,
up to floating-point arithmetic.  The finite-sample logarithmic free-energy
estimate is still biased; the identity applies to its exponential weights.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Callable

import torch
from torch import Tensor

from . import systems


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and strictly positive")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and strictly positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _finite(value: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values; no paths were discarded")
    return value


def _work_tensor(work: Tensor) -> Tensor:
    value = torch.as_tensor(work).to(dtype=torch.float64)
    if value.ndim == 0 or value.shape[-1] == 0 or value.numel() == 0:
        raise ValueError("work must contain at least one sample on its last axis")
    return _finite(value, "work")


def _centered_log_weights(work: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    log_weights = _finite(-beta * work, "dimensionless work")
    maximum = log_weights.max(dim=-1, keepdim=True).values
    centered = log_weights - maximum
    # A negative infinity from subtracting two finite, opposite extreme values
    # represents zero weight and is safe.  Positive infinities and NaNs are not.
    if bool(torch.isnan(centered).any() | torch.isposinf(centered).any()):
        raise ValueError("log weights cannot be represented in float64")
    return centered, maximum.squeeze(-1)


def jarzynski(work: Tensor, beta: float = 1.0) -> Tensor:
    """Return ``-log(mean(exp(-beta * work))) / beta`` on the last axis.

    Accumulation uses float64 and shifted log weights.  Leading dimensions are
    preserved, allowing ``[n_systems, n_paths]`` work arrays.
    """
    beta = _positive_scalar(beta, "beta")
    value = _work_tensor(work)
    centered, maximum = _centered_log_weights(value, beta)
    correction = torch.logsumexp(centered, dim=-1) - math.log(value.shape[-1])
    return _finite(-maximum / beta - correction / beta, "free-energy estimate")


def effective_sample_size(work: Tensor, beta: float = 1.0) -> Tensor:
    """Return importance-weight ESS, in [1, n_paths], on the last axis."""
    beta = _positive_scalar(beta, "beta")
    value = _work_tensor(work)
    centered, _ = _centered_log_weights(value, beta)
    log_ess = 2 * torch.logsumexp(centered, dim=-1) - torch.logsumexp(
        2 * centered, dim=-1
    )
    return log_ess.exp().clamp(min=1.0, max=float(value.shape[-1]))


def mean_work(work: Tensor) -> Tensor:
    """Return the uncorrected mean-work ablation on the last sample axis."""
    return _finite(_work_tensor(work).mean(dim=-1), "mean work")


def work_summary(work: Tensor, beta: float = 1.0) -> dict[str, float]:
    """Summarize a single system's one-dimensional work sample in plain floats."""
    value = _work_tensor(work)
    if value.ndim != 1:
        raise ValueError("work_summary requires one system's one-dimensional work")
    ess = effective_sample_size(value, beta)
    std = _finite(value.std(unbiased=False), "work standard deviation")
    return {
        "delta_f": float(jarzynski(value, beta)),
        "mean_work": float(mean_work(value)),
        "std_work": float(std),
        "ess": float(ess),
        "ess_fraction": float(ess / value.numel()),
        "n_samples": float(value.numel()),
    }


def _fields(
    drift_fn: Callable[[Tensor, Tensor, Any], tuple[Tensor, Tensor]],
    x: Tensor,
    t: Tensor,
    batch: Any,
) -> tuple[Tensor, Tensor]:
    fields = drift_fn(x, t, batch)
    if not isinstance(fields, (tuple, list)) or len(fields) != 2:
        raise ValueError("drift_fn must return (velocity, score)")
    checked = []
    for field, name in zip(fields, ("velocity", "score")):
        if not isinstance(field, Tensor) or field.shape != x.shape:
            raise ValueError(f"{name} must be a tensor with shape {tuple(x.shape)}")
        if field.device != x.device:
            raise ValueError(f"{name} and coordinates must be on the same device")
        checked.append(_finite(field.to(dtype=x.dtype), name))
    return checked[0], checked[1]


def _normal_log_density(value: Tensor, mean: Tensor, variance: float) -> Tensor:
    """Isotropic Gaussian density, with independent node/coordinate dimensions."""
    residual = value.to(torch.float64) - mean.to(torch.float64)
    dimension = value.shape[1] * value.shape[2]
    density = -0.5 * (
        residual.square().sum(dim=(1, 2)) / variance
        + dimension * math.log(2 * math.pi * variance)
    )
    return _finite(density, "Gaussian transition log density")


@torch.no_grad()
def simulate_forward(
    batch: Any,
    x0: Tensor,
    drift_fn: Callable[[Tensor, Tensor, Any], tuple[Tensor, Tensor]],
    steps: int,
    diffusion: float,
    generator: torch.Generator | None,
    beta: float = 1.0,
) -> dict[str, Tensor]:
    """Sample FEAT forward paths and compute their discrete physical work.

    ``x0`` has shape ``[n_paths, n_nodes, spatial_dim]``.  Each batch entry is
    one physical system and one path; repeat systems externally for multiple
    paths.  ``drift_fn(x, t, batch)`` receives a time vector of shape
    ``[n_paths]`` and returns same-shaped velocity and score fields.  Coordinates
    retain their input dtype; log densities and energy differences use float64.

    With ``epsilon = diffusion``, each forward transition has mean
    ``x + h * (velocity + epsilon * score)`` and variance ``2 * epsilon * h``.
    Its paired backward transition, evaluated at the *new* position and time,
    has mean ``x_new + h * (-velocity_new + epsilon * score_new)``.

    Returned ``work`` has physical energy units.  ``dimensionless_work`` is
    ``beta * (U_1(x_end) - U_0(x0)) + sum(log q_f - log q_b)``.  Real endpoint
    energies, rather than learned endpoint scores, fix the normalization.
    ``diffusion`` must be positive: deterministic ODEs require a separate
    Jacobian-based estimator and cannot use these Gaussian densities.
    """
    beta = _positive_scalar(beta, "beta")
    diffusion = _positive_scalar(diffusion, "diffusion")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("steps must be a strictly positive integer")
    if (
        not isinstance(x0, Tensor)
        or x0.ndim != 3
        or not x0.is_floating_point()
        or any(size == 0 for size in x0.shape)
    ):
        raise ValueError("x0 must be a nonempty floating tensor [paths, nodes, dims]")
    _finite(x0, "initial coordinates")
    x = x0.detach().clone()
    n_paths = x.shape[0]
    step_size = 1.0 / steps
    variance = _positive_scalar(2 * diffusion * step_size, "transition variance")
    std = math.sqrt(variance)
    log_ratio = torch.zeros(n_paths, dtype=torch.float64, device=x.device)

    def time_vector(index: int) -> Tensor:
        return x.new_full((n_paths,), index / steps)

    velocity, score = _fields(drift_fn, x, time_vector(0), batch)
    for index in range(steps):
        forward_mean = _finite(
            x + step_size * (velocity + diffusion * score), "forward mean"
        )
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        next_x = _finite(forward_mean + std * noise, "sampled coordinates")
        next_velocity, next_score = _fields(
            drift_fn, next_x, time_vector(index + 1), batch
        )
        backward_mean = _finite(
            next_x + step_size * (-next_velocity + diffusion * next_score),
            "backward mean",
        )
        log_ratio += _normal_log_density(next_x, forward_mean, variance)
        log_ratio -= _normal_log_density(x, backward_mean, variance)
        _finite(log_ratio, "accumulated path log-density ratio")
        x, velocity, score = next_x, next_velocity, next_score

    energies = []
    for endpoint, coordinates in ((0, x0), (1, x)):
        energy = systems.energy(batch, coordinates.to(torch.float64), endpoint, beta=beta)
        if not isinstance(energy, Tensor) or energy.shape != (n_paths,):
            raise ValueError("systems.energy must return one physical energy per path")
        if energy.device != x.device:
            raise ValueError("energies and coordinates must be on the same device")
        energies.append(_finite(energy.to(torch.float64), "endpoint energy"))
    delta_energy = energies[1] - energies[0]
    dimensionless_work = _finite(beta * delta_energy + log_ratio, "dimensionless work")
    work = _finite(dimensionless_work / beta, "physical work")
    return {
        "work": work,
        "endpoint": x,
        "dimensionless_work": dimensionless_work,
        "log_kernel_ratio": log_ratio,
        "delta_energy": delta_energy,
    }

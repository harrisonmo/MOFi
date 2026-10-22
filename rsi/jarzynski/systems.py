"""Analytically solvable trapped harmonic graphs, not molecular or MOF data.

For each Cartesian coordinate, the graph precision is a positive diagonal
trap plus a weighted graph Laplacian. The physical energy is
``U(x) = c + 0.5 * sum_d x_d.T @ K @ x_d``. Endpoint coordinates therefore
follow a zero-mean Gaussian with covariance ``(beta * K)^-1``.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import torch


FIELDS = ("a0", "a1", "k0", "k1", "c0", "c1")
GraphBatch = Mapping[str, torch.Tensor]


def _positive_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("beta must be finite and positive")
    return value


def _validate_batch(batch: GraphBatch) -> tuple[int, int]:
    if any(key not in batch for key in FIELDS):
        raise ValueError(f"Graph batches require fields {FIELDS}")
    reference = batch["a0"]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("a0 must be a floating tensor with shape [batch, nodes]")
    count, nodes = reference.shape
    if count < 1 or nodes < 1 or not reference.is_floating_point():
        raise ValueError("Graph batches must contain positive numbers of graphs and nodes")
    for key in FIELDS:
        value = batch[key]
        expected = (count, nodes, nodes) if key.startswith("k") else (
            (count,) if key.startswith("c") else (count, nodes)
        )
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{key} must have shape {expected}")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError("All graph tensors must have the same dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite")
        if key.startswith("a") and not bool((value > 0).all()):
            raise ValueError("All trap coefficients must be positive")
        if key.startswith("k"):
            if not bool((value >= 0).all()):
                raise ValueError("All spring coefficients must be nonnegative")
            if not torch.equal(value, value.transpose(-1, -2)):
                raise ValueError("Spring matrices must be symmetric")
            if not bool((value.diagonal(dim1=-2, dim2=-1) == 0).all()):
                raise ValueError("Spring matrix diagonals must be zero")
    return count, nodes


def _validate_endpoint(endpoint: int) -> None:
    if endpoint not in (0, 1):
        raise ValueError("endpoint must be 0 or 1")


def _validate_coordinates(batch: GraphBatch, x: torch.Tensor) -> None:
    count, nodes = _validate_batch(batch)
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != (count, nodes, 3):
        raise ValueError("Coordinates must have shape [batch, nodes, 3]")
    if not x.is_floating_point() or x.device != batch["a0"].device:
        raise ValueError("Coordinates must be floating point and share the graph device")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Coordinates must be finite")


def make_graphs(
    count: int,
    n_nodes: int,
    seed: int,
    ood: bool = False,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Generate connected rings with independent Bernoulli extra edges.

    ``ood`` is a descriptive flag; it deliberately changes no parameter
    ranges. The caller sets graph size (four nodes for ID, six for OOD in
    the registered experiment). Each potential is confining even without
    graph connectivity because all node trap coefficients are positive.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 2:
        raise ValueError("n_nodes must be an integer of at least two")
    if not isinstance(ood, bool):
        raise ValueError("ood must be a boolean")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, generator=generator, dtype=dtype)

    a0 = uniform((count, n_nodes), 0.6, 1.5)
    a1 = a0 * uniform((count, n_nodes), -0.7, 0.7).exp()
    mask = uniform((count, n_nodes, n_nodes), 0.0, 1.0) < 0.25
    mask = torch.triu(mask, diagonal=1)
    ring = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
    index = torch.arange(n_nodes)
    neighbor = (index + 1) % n_nodes
    ring[index, neighbor] = True
    ring[neighbor, index] = True
    mask = torch.triu(mask | ring, diagonal=1)
    upper_k0 = uniform((count, n_nodes, n_nodes), 0.05, 0.35) * mask
    upper_k1 = upper_k0 * uniform((count, n_nodes, n_nodes), -0.5, 0.5).exp()
    batch = {
        "a0": a0,
        "a1": a1,
        "k0": upper_k0 + upper_k0.transpose(-1, -2),
        "k1": upper_k1 + upper_k1.transpose(-1, -2),
        "c0": torch.zeros(count, dtype=dtype),
        "c1": uniform((count,), -0.4, 0.4),
    }
    _validate_batch(batch)
    return batch


def precision(batch: GraphBatch, endpoint: int) -> torch.Tensor:
    """Return physical stiffness K, before multiplication by beta."""
    _validate_batch(batch)
    _validate_endpoint(endpoint)
    a, k = batch[f"a{endpoint}"], batch[f"k{endpoint}"]
    return torch.diag_embed(a + k.sum(dim=-1)) - k


def energy(
    batch: GraphBatch, x: torch.Tensor, endpoint: int, beta: float = 1.0
) -> torch.Tensor:
    """Return physical U, not beta * U; beta is accepted and validated only."""
    _positive_beta(beta)
    _validate_coordinates(batch, x)
    stiffness = precision(batch, endpoint).to(dtype=x.dtype)
    return batch[f"c{endpoint}"].to(dtype=x.dtype) + 0.5 * (x * (stiffness @ x)).sum(dim=(-2, -1))


def energy_gradient(batch: GraphBatch, x: torch.Tensor, endpoint: int) -> torch.Tensor:
    """Return the analytical gradient of physical energy with respect to x."""
    _validate_coordinates(batch, x)
    return precision(batch, endpoint).to(dtype=x.dtype) @ x


def sample_endpoint(
    batch: GraphBatch,
    endpoint: int,
    generator: torch.Generator,
    beta: float = 1.0,
) -> torch.Tensor:
    """Draw one equilibrium configuration per graph using inverse Cholesky."""
    beta = _positive_beta(beta)
    stiffness = precision(batch, endpoint)
    count, nodes = batch["a0"].shape
    standard = torch.randn(
        (count, nodes, 3), generator=generator, dtype=stiffness.dtype, device=stiffness.device
    )
    lower = torch.linalg.cholesky(stiffness)
    return torch.linalg.solve_triangular(lower.transpose(-1, -2), standard, upper=True) / math.sqrt(beta)


def exact_delta_f(batch: GraphBatch, spatial_dim: int = 3, beta: float = 1.0) -> torch.Tensor:
    """Return F1-F0 = c1-c0 + d/(2 beta) * log(det(K1)/det(K0))."""
    beta = _positive_beta(beta)
    if isinstance(spatial_dim, bool) or not isinstance(spatial_dim, int) or spatial_dim < 1:
        raise ValueError("spatial_dim must be a positive integer")
    logdet0 = torch.linalg.slogdet(precision(batch, 0)).logabsdet
    logdet1 = torch.linalg.slogdet(precision(batch, 1)).logabsdet
    return batch["c1"] - batch["c0"] + (spatial_dim / (2 * beta)) * (logdet1 - logdet0)


def interpolated_score(
    batch: GraphBatch, x: torch.Tensor, t: torch.Tensor | float, beta: float = 1.0
) -> torch.Tensor:
    """Score of the equilibrium density for the linearly interpolated potential."""
    beta = _positive_beta(beta)
    _validate_coordinates(batch, x)
    time = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    count = len(x)
    if time.numel() == 1:
        time = time.reshape(1, 1, 1)
    elif tuple(time.shape) in ((count,), (count, 1), (count, 1, 1)):
        time = time.reshape(count, 1, 1)
    else:
        raise ValueError("t must be a scalar or one scalar per graph")
    if not bool(torch.isfinite(time).all()) or not bool(((time >= 0) & (time <= 1)).all()):
        raise ValueError("Interpolation times must be finite and in [0, 1]")
    stiffness = (1 - time) * precision(batch, 0).to(dtype=x.dtype) + time * precision(batch, 1).to(dtype=x.dtype)
    return -beta * (stiffness @ x)


def take_graphs(batch: GraphBatch, indices) -> dict[str, torch.Tensor]:
    """Select graphs, preserving a leading batch dimension and every field."""
    count, _ = _validate_batch(batch)
    if isinstance(indices, slice):
        index = torch.arange(count, device=batch["a0"].device)[indices]
    else:
        index = torch.as_tensor(indices, device=batch["a0"].device)
        if index.ndim == 0:
            index = index.reshape(1)
        if index.ndim != 1 or index.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("indices must be a slice or one-dimensional integer indices")
        index = index.to(dtype=torch.long)
    if index.numel() == 0 or not bool(((index >= 0) & (index < count)).all()):
        raise ValueError("Selection must contain valid, nonnegative graph indices")
    return {key: batch[key].index_select(0, index) for key in FIELDS}


def graph_hash(batch: GraphBatch) -> str:
    """Hash exact graph values, shapes, and dtypes in a stable field order."""
    _validate_batch(batch)
    digest = hashlib.sha256(b"trapped-harmonic-graphs-v1\n")
    for key in FIELDS:
        array = batch[key].detach().cpu().contiguous().numpy()
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        header = json.dumps(
            {"field": key, "shape": list(array.shape), "dtype": array.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(header.encode("ascii") + b"\n")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

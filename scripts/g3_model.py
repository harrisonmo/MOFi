"""Smooth-cutoff periodic message passing that retains neighbour count and density.

Changes relative to the frozen G1 PeriodicMPNN:
- Messages are multiplied by a cosine envelope, so each edge fades to zero
  value and zero slope at the cutoff instead of appearing abruptly.
- Messages are summed and divided by a fixed training-set average degree.
  Mean aggregation discarded neighbour counts (a porosity signal) and its
  per-node divisor jumped whenever an edge crossed the cutoff.
- The readout receives the standardized log number density (atoms per A^3).
"""
import math

import numpy as np
import torch
from torch import nn


def cosine_envelope(distance, cutoff):
    inside = (distance < cutoff).to(distance.dtype)
    return 0.5 * (torch.cos(math.pi * distance / cutoff) + 1) * inside


def log_density(graph):
    return math.log(len(graph["z"]) / abs(float(np.linalg.det(graph["cell"]))))


def fit_normalizers(training_graphs):
    """Return constructor statistics computed from training graphs only."""
    degrees = [len(g["src"]) / len(g["z"]) for g in training_graphs]
    densities = np.array([log_density(g) for g in training_graphs])
    scale = float(densities.std())
    return {"avg_degree": float(np.mean(degrees)), "density_mean": float(densities.mean()),
            "density_scale": scale if scale > 0 else 1.0}


class SmoothPeriodicMPNN(nn.Module):
    def __init__(self, avg_degree, density_mean, density_scale, cutoff=8.0, width=64, layers=3, radial=32):
        super().__init__()
        if not avg_degree > 0 or not density_scale > 0 or not cutoff > 0:
            raise ValueError("Normalizers and cutoff must be positive")
        self.cutoff = cutoff
        self.embedding = nn.Embedding(119, width)
        self.register_buffer("centers", torch.linspace(0, cutoff, radial))
        self.register_buffer("avg_degree", torch.tensor(float(avg_degree)))
        self.register_buffer("density_mean", torch.tensor(float(density_mean)))
        self.register_buffer("density_scale", torch.tensor(float(density_scale)))
        self.gaussian_width = cutoff / (radial - 1)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(width * 2 + radial, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch):
        z, src, dst, distance, graph_index, density, graph_count = batch
        h = self.embedding(z)
        radial = torch.exp(-((distance[:, None] - self.centers[None, :]) / self.gaussian_width) ** 2)
        envelope = cosine_envelope(distance, self.cutoff)[:, None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            edge = message(torch.cat([h[src], h[dst], radial], dim=1)) * envelope
            aggregate = torch.zeros_like(h).index_add(0, dst, edge) / self.avg_degree
            h = norm(h + torch.nn.functional.silu(update(torch.cat([h, aggregate], dim=1))))
        pooled = torch.zeros(graph_count, h.shape[1], device=h.device).index_add(0, graph_index, h)
        pooled = pooled / torch.bincount(graph_index, minlength=graph_count).to(h.dtype)[:, None]
        density = (density - self.density_mean) / self.density_scale
        return self.readout(torch.cat([pooled, density[:, None]], dim=1)).squeeze(-1)


def pack(graphs, device):
    src, dst, z, distance, groups = [], [], [], [], []
    offset = 0
    for i, graph in enumerate(graphs):
        z.append(torch.as_tensor(graph["z"], dtype=torch.long))
        src.append(torch.as_tensor(graph["src"], dtype=torch.long) + offset)
        dst.append(torch.as_tensor(graph["dst"], dtype=torch.long) + offset)
        distance.append(torch.as_tensor(graph["distance"], dtype=torch.float32))
        groups.append(torch.full((len(graph["z"]),), i, dtype=torch.long))
        offset += len(graph["z"])
    density = torch.tensor([log_density(g) for g in graphs], dtype=torch.float32)
    return tuple(torch.cat(xs).to(device) for xs in [z, src, dst, distance, groups]) + (density.to(device), len(graphs))

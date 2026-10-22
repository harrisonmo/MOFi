"""Factorial ablation of the three G3 representation changes.

Factors: cutoff radius (5 or 8 A), cosine envelope (off/on) and aggregation
("mean": divide by node degree, no density; "sum_density": divide by a fixed
training-set average degree and append standardized log number density).
Modules are created in the same order as the G1 PeriodicMPNN, so the
(5 A, off, mean) cell matches G1 and the (8 A, on, sum_density) cell matches
SmoothPeriodicMPNN under the same seed.
"""
import torch
from torch import nn

from g3_model import cosine_envelope

CELLS = {
    f"c{cutoff}-{'env' if envelope else 'hard'}-{'sum' if aggregation == 'sum_density' else 'mean'}":
        {"cutoff": float(cutoff), "envelope": envelope, "aggregation": aggregation}
    for cutoff in (5, 8) for aggregation in ("mean", "sum_density") for envelope in (False, True)
}


class AblationMPNN(nn.Module):
    def __init__(self, cutoff, envelope, aggregation, avg_degree, density_mean, density_scale,
                 width=64, layers=3, radial=32):
        super().__init__()
        if aggregation not in {"mean", "sum_density"}:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        if not avg_degree > 0 or not density_scale > 0 or not cutoff > 0:
            raise ValueError("Normalizers and cutoff must be positive")
        self.cutoff, self.envelope, self.aggregation = cutoff, envelope, aggregation
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
        extra = 1 if aggregation == "sum_density" else 0
        self.readout = nn.Sequential(nn.Linear(width + extra, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch):
        z, src, dst, distance, graph_index, density, graph_count = batch
        h = self.embedding(z)
        radial = torch.exp(-((distance[:, None] - self.centers[None, :]) / self.gaussian_width) ** 2)
        envelope = cosine_envelope(distance, self.cutoff)[:, None] if self.envelope else None
        if self.aggregation == "mean":
            divisor = torch.bincount(dst, minlength=len(z)).clamp_min(1).to(h.dtype)[:, None]
        else:
            divisor = self.avg_degree
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            edge = message(torch.cat([h[src], h[dst], radial], dim=1))
            if envelope is not None:
                edge = edge * envelope
            aggregate = torch.zeros_like(h).index_add(0, dst, edge) / divisor
            h = norm(h + torch.nn.functional.silu(update(torch.cat([h, aggregate], dim=1))))
        pooled = torch.zeros(graph_count, h.shape[1], device=h.device).index_add(0, graph_index, h)
        pooled = pooled / torch.bincount(graph_index, minlength=graph_count).to(h.dtype)[:, None]
        if self.aggregation == "sum_density":
            density = (density - self.density_mean) / self.density_scale
            pooled = torch.cat([pooled, density[:, None]], dim=1)
        return self.readout(pooled).squeeze(-1)

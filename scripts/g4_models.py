"""G4 models: the G3 encoder with geometry descriptors, and an improved pure-graph encoder.

g4_hybrid: the G3 SmoothPeriodicMPNN encoder (pooled atoms plus log number
density) concatenated with the 14 standardized G2 geometry descriptors, read
out by a 79 -> 64 -> 1 MLP, mirroring the G2 hybrid construction.

g4_graph: no precomputed descriptors. The G3 ablation showed that exposing
neighbour count and density (sum aggregation plus log number density) was the
dominant gain, so this encoder exposes more size-aware global information,
all derived from atoms and cell only:
- log mass density (g/cm^3), since the target is gravimetric (mmol/g);
- volume-normalized sum pooling, sum_i h_i / V scaled by the training mean
  number density, which equals mean pooling times relative density and lets
  the readout weight atom features by how densely they fill the cell.
"""
import math

import numpy as np
import torch
from ase.data import atomic_masses
from torch import nn

from g3_ablation_model import AblationMPNN
from g3_model import SmoothPeriodicMPNN, cosine_envelope

AVOGADRO_CM3 = 1.66053906660  # g/cm^3 for one dalton per cubic angstrom


def volume(graph):
    return abs(float(np.linalg.det(graph["cell"])))


def global_features(graph):
    """Log number density (atoms/A^3) and log mass density (g/cm^3) from the structure alone."""
    v = volume(graph)
    mass = float(atomic_masses[np.asarray(graph["z"])].sum())
    return [math.log(len(graph["z"]) / v), math.log(mass / v * AVOGADRO_CM3)]


def fit_normalizers(training_graphs):
    """Return constructor statistics computed from training graphs only."""
    degrees = [len(g["src"]) / len(g["z"]) for g in training_graphs]
    features = np.array([global_features(g) for g in training_graphs])
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    return {"avg_degree": float(np.mean(degrees)), "global_mean": features.mean(axis=0).tolist(),
            "global_scale": scale.tolist(),
            "number_density_mean": float(np.mean([len(g["z"]) / volume(g) for g in training_graphs]))}


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
    features = torch.tensor([global_features(g) for g in graphs], dtype=torch.float32)
    volumes = torch.tensor([volume(g) for g in graphs], dtype=torch.float32)
    return tuple(torch.cat(xs).to(device) for xs in [z, src, dst, distance, groups]) + (
        features.to(device), volumes.to(device), len(graphs))


def g3_batch(batch):
    """Adapt a G4 batch to the G3 encoder, whose density is global feature 0."""
    z, src, dst, distance, groups, features, _, count = batch
    return z, src, dst, distance, groups, features[:, 0], count


class G4Hybrid(nn.Module):
    def __init__(self, avg_degree, global_mean, global_scale, cutoff=8.0, width=64, **_):
        super().__init__()
        self.encoder = SmoothPeriodicMPNN(avg_degree, global_mean[0], global_scale[0], cutoff=cutoff, width=width)
        self.encoder.readout = nn.Identity()
        self.readout = nn.Sequential(nn.Linear(width + 1 + 14, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch, geometry):
        # With an identity readout the encoder returns the (graphs, width + 1) pooled
        # representation; its final squeeze(-1) is a no-op on that shape.
        pooled = self.encoder(g3_batch(batch))
        return self.readout(torch.cat([pooled, geometry], dim=1)).squeeze(-1)


class G4GraphMPNN(nn.Module):
    def __init__(self, avg_degree, global_mean, global_scale, number_density_mean, envelope,
                 cutoff=8.0, width=64, layers=3, radial=32):
        super().__init__()
        if not avg_degree > 0 or not number_density_mean > 0 or not cutoff > 0:
            raise ValueError("Normalizers and cutoff must be positive")
        self.cutoff, self.envelope = cutoff, envelope
        self.embedding = nn.Embedding(119, width)
        self.register_buffer("centers", torch.linspace(0, cutoff, radial))
        self.register_buffer("avg_degree", torch.tensor(float(avg_degree)))
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.tensor(global_scale, dtype=torch.float32))
        self.register_buffer("number_density_mean", torch.tensor(float(number_density_mean)))
        self.gaussian_width = cutoff / (radial - 1)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(width * 2 + radial, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(width * 2 + 2, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch):
        z, src, dst, distance, graph_index, features, volumes, graph_count = batch
        h = self.embedding(z)
        radial = torch.exp(-((distance[:, None] - self.centers[None, :]) / self.gaussian_width) ** 2)
        envelope = cosine_envelope(distance, self.cutoff)[:, None] if self.envelope else None
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            edge = message(torch.cat([h[src], h[dst], radial], dim=1))
            if envelope is not None:
                edge = edge * envelope
            aggregate = torch.zeros_like(h).index_add(0, dst, edge) / self.avg_degree
            h = norm(h + torch.nn.functional.silu(update(torch.cat([h, aggregate], dim=1))))
        summed = torch.zeros(graph_count, h.shape[1], device=h.device).index_add(0, graph_index, h)
        mean_pool = summed / torch.bincount(graph_index, minlength=graph_count).to(h.dtype)[:, None]
        volume_pool = summed / volumes[:, None] / self.number_density_mean
        features = (features - self.global_mean) / self.global_scale
        return self.readout(torch.cat([mean_pool, volume_pool, features], dim=1)).squeeze(-1)


def make_model(arm, normalizers, envelope):
    if arm == "g3":
        return SmoothPeriodicMPNN(normalizers["avg_degree"], normalizers["global_mean"][0], normalizers["global_scale"][0])
    if arm == "g3_hard":
        # Best G3 ablation cell (8 A, hard cutoff, sum + density): the control for g4_graph.
        return AblationMPNN(8.0, False, "sum_density", normalizers["avg_degree"],
                            normalizers["global_mean"][0], normalizers["global_scale"][0])
    if arm == "g4_hybrid":
        return G4Hybrid(**normalizers)
    if arm == "g4_graph":
        return G4GraphMPNN(**normalizers, envelope=envelope)
    raise ValueError(f"Unknown arm: {arm}")


def forward(model, arm, batch, geometry):
    if arm in {"g3", "g3_hard"}:
        return model(g3_batch(batch))
    return model(batch, geometry) if arm == "g4_hybrid" else model(batch)

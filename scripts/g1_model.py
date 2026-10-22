"""A compact periodic distance-message-passing regression baseline."""
import torch
from torch import nn


class PeriodicMPNN(nn.Module):
    def __init__(self, width=64, layers=3, radial=32):
        super().__init__()
        self.embedding = nn.Embedding(119, width)
        self.register_buffer("centers", torch.linspace(0, 5, radial))
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(width * 2 + radial, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch):
        z, src, dst, distance, graph_index, graph_count = batch
        h = self.embedding(z)
        radial = torch.exp(-((distance[:, None] - self.centers[None, :]) / (5 / 31)) ** 2)
        degree = torch.bincount(dst, minlength=len(z)).clamp_min(1).to(h.dtype)[:, None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            edge = message(torch.cat([h[src], h[dst], radial], dim=1))
            aggregate = torch.zeros_like(h).index_add(0, dst, edge) / degree
            h = norm(h + torch.nn.functional.silu(update(torch.cat([h, aggregate], dim=1))))
        pooled = torch.zeros(graph_count, h.shape[1], device=h.device).index_add(0, graph_index, h)
        pooled = pooled / torch.bincount(graph_index, minlength=graph_count).to(h.dtype)[:, None]
        return self.readout(pooled).squeeze(-1)


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
    return tuple(torch.cat(xs).to(device) for xs in [z, src, dst, distance, groups]) + (len(graphs),)

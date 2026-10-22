"""Graph regression and O(3)-equivariant stochastic-interpolant networks.

The harmonic benchmark has an origin-centred trap: rotation and permutation
equivariance apply, but translation equivariance does not. This is an original,
small implementation inspired by FEAT, not a copy of its molecular model.
"""

import math

import torch
from torch import nn


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class GraphEncoder(nn.Module):
    """Two-state graph conditioning, optionally including invariant coordinates."""

    def __init__(self, width=32, layers=2):
        super().__init__()
        self.nodes = mlp(7, width, width)
        self.messages = nn.ModuleList([mlp(2 * width + 3, width, width) for _ in range(layers)])
        self.updates = nn.ModuleList([mlp(2 * width, width, width) for _ in range(layers)])

    def forward(self, graph, x=None, t=None):
        a0, a1, k0, k1 = (graph[k] for k in ("a0", "a1", "k0", "k1"))
        batch, nodes = a0.shape
        if t is None:
            t = torch.zeros(batch, dtype=a0.dtype, device=a0.device)
        t = torch.as_tensor(t, dtype=a0.dtype, device=a0.device).expand(batch).reshape(batch, 1).expand(-1, nodes)
        if x is None:
            radius = torch.zeros_like(a0)
            distance = torch.zeros_like(k0)
        else:
            radius = x.square().sum(-1)
            distance = (x[:, :, None] - x[:, None, :]).square().sum(-1)
        h = self.nodes(torch.stack((a0, a1, k0.sum(-1), k1.sum(-1), t, t.square(), radius), -1))
        edge_features = torch.stack((k0, k1, distance), -1)
        mask = ((k0 + k1) > 0).unsqueeze(-1)
        degree = mask.sum(2).clamp_min(1)
        for message, update in zip(self.messages, self.updates):
            edge = torch.cat((h[:, :, None].expand(-1, -1, nodes, -1),
                              h[:, None, :].expand(-1, nodes, -1, -1), edge_features), -1)
            aggregate = (message(edge) * mask).sum(2) / degree
            h = (h + update(torch.cat((h, aggregate), -1))) / math.sqrt(2)
        return h, edge_features, mask, degree


class FreeEnergyGNN(nn.Module):
    """Ordinary supervised graph-to-scalar free-energy regressor.

    It receives exact training free-energy labels. The known endpoint constant
    offset is added explicitly, because it changes no forces or distributions.
    """

    def __init__(self, width=32, layers=2):
        super().__init__()
        self.encoder = GraphEncoder(width, layers)
        self.readout = mlp(width, width, 1)

    def forward(self, graph):
        h, _, _, _ = self.encoder(graph)
        return self.readout(h).squeeze(-1).sum(-1) + graph["c1"] - graph["c0"]


class TransportGNN(nn.Module):
    """Learn the velocity and score of a two-sided stochastic interpolant."""

    def __init__(self, width=32, layers=2):
        super().__init__()
        self.encoder = GraphEncoder(width, layers)
        self.node_coefficients = mlp(width, width, 2)
        self.edge_coefficients = mlp(2 * width + 3, width, 2)
        # A stable initial process: zero transport with a unit Gaussian score.
        for layer in (self.node_coefficients[-1], self.edge_coefficients[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, graph):
        h, edge_features, mask, degree = self.encoder(graph, x, t)
        nodes = h.shape[1]
        edge = torch.cat((h[:, :, None].expand(-1, -1, nodes, -1),
                          h[:, None, :].expand(-1, nodes, -1, -1), edge_features), -1)
        coefficients = self.edge_coefficients(edge) * mask
        difference = x[:, :, None] - x[:, None, :]
        aggregate = (coefficients.unsqueeze(-1) * difference.unsqueeze(-2)).sum(2) / degree.unsqueeze(-1)
        vector = self.node_coefficients(h).unsqueeze(-1) * x.unsqueeze(-2) + aggregate
        return vector[:, :, 0], vector[:, :, 1] - x


def interpolant_loss(model, graph, x0, x1, t, noise, gamma_scale=0.5):
    """Independent endpoint FM + weighted DSM; deliberately no TSM loss.

    gamma(t)=scale*sin(pi*t). Weighting DSM by gamma^2 avoids a singular
    target at the endpoints. The sampler still evaluates the model at t=0/1;
    exact discrete path-density ratios correct imperfect learned scores.
    """
    if not math.isfinite(gamma_scale) or gamma_scale <= 0:
        raise ValueError("gamma_scale must be finite and positive")
    if t.ndim != 1 or len(t) != len(x0):
        raise ValueError("Training times must have shape [batch]")
    if x0.shape != x1.shape or x0.shape != noise.shape or x0.ndim != 3:
        raise ValueError("Endpoint samples and noise must share [batch, nodes, dimensions]")
    t3 = t[:, None, None]
    gamma = gamma_scale * torch.sin(math.pi * t3)
    gamma_dot = gamma_scale * math.pi * torch.cos(math.pi * t3)
    x = (1 - t3) * x0 + t3 * x1 + gamma * noise
    target_velocity = x1 - x0 + gamma_dot * noise
    velocity, score = model(x, t, graph)
    flow_loss = (velocity - target_velocity).square().mean()
    score_loss = (gamma * score + noise).square().mean()
    return flow_loss + score_loss, {"flow": flow_loss.detach(), "score": score_loss.detach()}

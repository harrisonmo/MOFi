"""Geometry representation ablations that preserve the frozen G1 encoder."""
import numpy as np
import torch
from torch import nn
from g1_model import PeriodicMPNN


class GeometryScaler:
    def fit(self, training):
        training = np.asarray(training, dtype=np.float64)
        if training.ndim != 2 or training.shape[1] != 14 or not np.isfinite(training).all():
            raise ValueError("Expected finite training geometry with 14 columns")
        self.mean = training.mean(axis=0)
        self.scale = training.std(axis=0)
        self.scale[self.scale == 0] = 1
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
            raise ValueError("Expected finite geometry with 14 columns")
        return ((values - self.mean) / self.scale).astype(np.float32)


class HybridGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PeriodicMPNN()
        self.encoder.readout = nn.Identity()
        self.readout = nn.Sequential(nn.Linear(78, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, graph, geometry):
        pooled = self.encoder(graph)
        return self.readout(torch.cat([pooled, geometry], dim=1)).squeeze(-1)


class GeometryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(14, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, geometry):
        return self.layers(geometry).squeeze(-1)


def make_model(arm):
    if arm == "gnn":
        return PeriodicMPNN()
    if arm == "hybrid":
        return HybridGNN()
    if arm == "mlp":
        return GeometryMLP()
    raise ValueError(f"Unknown neural arm: {arm}")

"""Equivariant transport models with controlled velocity/score sharing."""
import torch
from torch import nn
from rsi.jarzynski.models import GraphEncoder, TransportGNN, mlp


class VectorHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.node = mlp(width, width, 1)
        self.edge = mlp(2 * width + 3, width, 1)
        for layer in [self.node[-1], self.edge[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, encoded, x):
        h, features, mask, degree = encoded
        nodes = h.shape[1]
        edge = torch.cat((h[:, :, None].expand(-1, -1, nodes, -1),
                          h[:, None, :].expand(-1, nodes, -1, -1), features), -1)
        relative = x[:, :, None] - x[:, None, :]
        return self.node(h) * x + (self.edge(edge) * mask * relative).sum(2) / degree


class BranchedTransport(nn.Module):
    def __init__(self, width=32, layers=2):
        super().__init__()
        self.encoder = GraphEncoder(width, layers)
        self.adapters = nn.ModuleList([mlp(width, width, width) for _ in range(2)])
        self.heads = nn.ModuleList([VectorHead(width) for _ in range(2)])

    def forward(self, x, t, graph):
        h, features, mask, degree = self.encoder(graph, x, t)
        outputs = [head(((h + adapter(h)) / 2**.5, features, mask, degree), x)
                   for adapter, head in zip(self.adapters, self.heads)]
        return outputs[0], outputs[1] - x


class IndependentTransport(nn.Module):
    def __init__(self, width=32, layers=2):
        super().__init__()
        self.encoders = nn.ModuleList([GraphEncoder(width, layers) for _ in range(2)])
        self.heads = nn.ModuleList([VectorHead(width) for _ in range(2)])

    def forward(self, x, t, graph):
        outputs = [head(encoder(graph, x, t), x) for encoder, head in zip(self.encoders, self.heads)]
        return outputs[0], outputs[1] - x


MODELS = {'shared': TransportGNN, 'branched': BranchedTransport, 'independent': IndependentTransport}

"""Nontrivial symmetry and gradient-isolation contracts for new architectures."""
import os
import sys
from pathlib import Path
import pytest
import torch

if 'QK_SOURCE' not in os.environ:
    pytest.skip('QK_SOURCE must identify the frozen Qiankun checkout', allow_module_level=True)
sys.path.insert(0, os.environ['QK_SOURCE'])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qk9_models import MODELS
from rsi.jarzynski.systems import make_graphs


def inputs():
    torch.manual_seed(991)
    return make_graphs(3, 4, 992), torch.randn(3, 4, 3, dtype=torch.float64), torch.tensor([.1, .5, .9], dtype=torch.float64)


def trained_like(name):
    model = MODELS[name](12, 2).double()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * .03)
    return model


@pytest.mark.parametrize('name', list(MODELS))
def test_initial_fields(name):
    g, x, t = inputs()
    v, s = MODELS[name](12, 2).double()(x, t, g)
    torch.testing.assert_close(v, torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(s, -x, rtol=0, atol=0)


@pytest.mark.parametrize('name', list(MODELS))
def test_rotation_reflection_permutation_and_batch_isolation(name):
    g, x, t = inputs()
    model = trained_like(name)
    v, s = model(x, t, g)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype))
    for sign in [1., -1.]:
        rotation = q * sign
        for actual, expected in zip(model(x @ rotation, t, g), [v @ rotation, s @ rotation]):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    perm = torch.tensor([2, 0, 3, 1])
    gp = {k: z[:, perm][:, :, perm] if k.startswith('k') else z[:, perm] if k.startswith('a') else z for k, z in g.items()}
    for actual, expected in zip(model(x[:, perm], t, gp), [v[:, perm], s[:, perm]]):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for i in range(3):
        for actual, expected in zip(model(x[i:i+1], t[i:i+1], {k:z[i:i+1] for k,z in g.items()}), [v[i:i+1], s[i:i+1]]):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('name', ['branched', 'independent'])
def test_private_branch_gradients_are_isolated(name):
    g, x, t = inputs()
    model = trained_like(name)
    private = model.adapters if name == 'branched' else model.encoders
    for field in [0, 1]:
        model.zero_grad(set_to_none=True)
        model(x, t, g)[field].square().mean().backward()
        assert sum(float(p.grad.abs().sum()) for p in private[field].parameters() if p.grad is not None) > 0
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in private[1-field].parameters())
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.heads[1-field].parameters())

"""Representation and preprocessing contracts for the frozen-data comparison."""
import sys
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from g2_models import GeometryScaler, HybridGNN, GeometryMLP
from g1_model import PeriodicMPNN
from run_g2_comparison import validate_config


def test_scaler_does_not_fit_validation_and_constant_columns_are_finite():
    x = np.ones((3, 14)); x[:, 0] = [1, 2, 3]
    scaler = GeometryScaler().fit(x)
    before = scaler.mean.copy()
    result = scaler.transform(np.full((2, 14), 1000))
    np.testing.assert_array_equal(scaler.mean, before)
    assert np.isfinite(result).all() and scaler.scale[1] == 1
    np.testing.assert_allclose(scaler.transform(x).mean(axis=0), 0, atol=1e-6)


def test_nonfinite_and_wrong_feature_count_are_rejected():
    with pytest.raises(ValueError):
        GeometryScaler().fit(np.ones((3, 15)))
    with pytest.raises(ValueError):
        GeometryScaler().fit(np.full((3, 14), np.nan))


def test_hybrid_can_reproduce_pure_gnn_with_zero_geometry_weights():
    torch.manual_seed(17); pure = PeriodicMPNN().eval()
    torch.manual_seed(17); hybrid = HybridGNN().eval()
    with torch.no_grad():
        hybrid.readout[0].weight[:, :64].copy_(pure.readout[0].weight)
        hybrid.readout[0].weight[:, 64:].zero_()
        hybrid.readout[0].bias.copy_(pure.readout[0].bias)
        hybrid.readout[2].load_state_dict(pure.readout[2].state_dict())
    graph = (torch.tensor([6, 30]), torch.tensor([0, 1]), torch.tensor([1, 0]),
             torch.tensor([2., 2.]), torch.tensor([0, 0]), 1)
    with torch.no_grad():
        torch.testing.assert_close(pure(graph), hybrid(graph, torch.randn(1, 14)))


def test_geometry_branch_receives_gradients():
    model = HybridGNN()
    graph = (torch.tensor([6, 30]), torch.tensor([0, 1]), torch.tensor([1, 0]),
             torch.tensor([2., 2.]), torch.tensor([0, 0]), 1)
    geometry = torch.randn(1, 14, requires_grad=True)
    model(graph, geometry).sum().backward()
    assert geometry.grad.abs().sum() > 0
    assert model.encoder.embedding.weight.grad.abs().sum() > 0


def test_mlp_preserves_batch_order():
    model = GeometryMLP().eval(); x = torch.randn(5, 14)
    torch.testing.assert_close(model(x.flip(0)), model(x).flip(0))


def test_config_cannot_silently_enable_heldout_or_expand_budget():
    import json
    c = json.loads((Path(__file__).parents[1] / "configs/g2-comparison.json").read_text())
    validate_config(c)
    for key, value in [("heldout_evaluation", True), ("max_epochs", 100), ("formal_seconds", 2000)]:
        with pytest.raises(ValueError):
            validate_config({**c, key: value})

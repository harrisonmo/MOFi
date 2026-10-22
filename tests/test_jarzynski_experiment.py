"""Unexecuted runner contracts; these tests require no CUDA hardware."""
import copy
import json
import math
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from rsi.jarzynski import experiment
from rsi.jarzynski.systems import make_graphs


@pytest.fixture
def config():
    return json.loads(experiment.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_registered_configuration_passes_without_mutation(config):
    original = copy.deepcopy(config)
    experiment.validate_config(config)
    assert config == original


@pytest.mark.parametrize("field,value", [
    ("train_graphs", 0),
    ("batch_size", -1),
    ("updates", True),
    ("layers", 1.5),
    ("nodes", 1),
    ("ood_nodes", 4),
    ("endpoint_samples_per_state_per_training_graph", 0),
    ("beta", 0),
    ("beta", 2.0),
    ("spatial_dim", 2),
    ("learning_rate", float("nan")),
    ("gradient_clip", float("inf")),
    ("gamma_scale", -0.5),
    ("diffusion", 0),
    ("training_time_min", 0),
    ("training_time_min", 0.5),
    ("training_time_min", float("nan")),
    ("calibration_seconds_limit", 0),
    ("training_seconds_per_arm_limit", -1),
    ("inference_seconds_per_seed_limit", float("inf")),
    ("total_seconds_limit", True),
    ("total_seconds_limit", 7201),
    ("updates", 5001),
    ("paths_per_graph", 4097),
    ("inference_steps", 513),
    ("seeds", []),
    ("seeds", [17, 17]),
    ("seeds", [True, 29]),
    ("seeds", [-1, 29]),
    ("arms", ["supervised_gnn", "feat_gnn_jarzynski"]),
])
def test_invalid_parameters_and_unregistered_budgets_are_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        experiment.validate_config(config)


def test_missing_required_budget_is_rejected(config):
    del config["calibration_seconds_limit"]
    with pytest.raises(ValueError, match="calibration_seconds_limit"):
        experiment.validate_config(config)


def test_require_cuda_rejects_unavailable_device_before_creating_one(monkeypatch):
    availability = Mock(return_value=False)
    device = Mock(side_effect=AssertionError("No device may be created after failed CUDA admission"))
    monkeypatch.setattr(experiment.torch.cuda, "is_available", availability)
    monkeypatch.setattr(experiment.torch, "device", device)
    with pytest.raises(RuntimeError, match="CUDA GPU is required.*No CPU experiment was started"):
        experiment.require_cuda()
    availability.assert_called_once_with()
    device.assert_not_called()


def test_metrics_use_graph_errors_and_each_graphs_actual_path_count():
    records = [
        {
            "split": "test", "truth": 2.0, "paths": 20,
            "predictions": {
                "supervised_gnn": 3.0, "feat_gnn_jarzynski": 2.5,
                "feat_gnn_mean_work": 4.0, "no_transport_jarzynski": 1.0,
            },
            "ess": {"feat_gnn_jarzynski": 3.0, "no_transport_jarzynski": 2.0},
        },
        {
            "split": "test", "truth": 5.0, "paths": 200,
            "predictions": {
                "supervised_gnn": 2.0, "feat_gnn_jarzynski": 5.5,
                "feat_gnn_mean_work": 7.0, "no_transport_jarzynski": 6.0,
            },
            "ess": {"feat_gnn_jarzynski": 15.0, "no_transport_jarzynski": 20.0},
        },
    ]
    results = experiment.metrics(records)["test"]
    expected = {
        "supervised_gnn": (2.0, math.sqrt(5.0), -1.0),
        "feat_gnn_jarzynski": (0.5, 0.5, 0.5),
        "feat_gnn_mean_work": (2.0, 2.0, 2.0),
        "no_transport_jarzynski": (1.0, 1.0, 0.0),
    }
    for arm, (mae, rmse, bias) in expected.items():
        assert results[arm]["mae"] == pytest.approx(mae)
        assert results[arm]["rmse"] == pytest.approx(rmse)
        assert results[arm]["signed_bias"] == pytest.approx(bias)
        assert results[arm]["graphs"] == 2
    feat = results["feat_gnn_jarzynski"]
    assert feat["ess_mean"] == pytest.approx(9.0)
    assert feat["ess_min"] == pytest.approx(3.0)
    assert feat["ess_fraction_mean"] == pytest.approx((3 / 20 + 15 / 200) / 2)
    assert feat["low_ess_graphs"] == 1
    # Exactly 10% ESS is not below the registered threshold.
    physical = results["no_transport_jarzynski"]
    assert physical["ess_fraction_mean"] == pytest.approx(0.1)
    assert physical["low_ess_graphs"] == 0
    assert "ess_mean" not in results["supervised_gnn"]
    assert "ess_mean" not in results["feat_gnn_mean_work"]


def test_endpoint_training_pool_uses_independent_reproducible_endpoint_draws():
    # Identical endpoints expose accidental reuse of the same random normals.
    # This is a small CPU sampling contract, not an experiment or benchmark.
    graphs = make_graphs(2, 4, 137)
    graphs["a1"] = graphs["a0"].clone()
    graphs["k1"] = graphs["k0"].clone()
    graphs["c1"] = graphs["c0"].clone()
    left, right = experiment.training_pool(graphs, samples=16, seed=139, beta=1.0)
    repeat_left, repeat_right = experiment.training_pool(graphs, samples=16, seed=139, beta=1.0)
    changed_left, changed_right = experiment.training_pool(graphs, samples=16, seed=149, beta=1.0)
    assert left.shape == right.shape == (2, 16, 4, 3)
    torch.testing.assert_close(left, repeat_left, rtol=0, atol=0)
    torch.testing.assert_close(right, repeat_right, rtol=0, atol=0)
    assert not torch.equal(left, right)
    assert not torch.equal(left, changed_left)
    assert not torch.equal(right, changed_right)

"""Contracts for the G3 factorial ablation model."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from g1_model import PeriodicMPNN, pack as pack_g1
from g3_ablation_model import CELLS, AblationMPNN
from g3_model import SmoothPeriodicMPNN, pack

NORMALIZERS = {"avg_degree": 2.0, "density_mean": -6.0, "density_scale": 1.0}


def graphs():
    rng = np.random.default_rng(3)
    out = []
    for atoms in (3, 5):
        src, dst = np.meshgrid(range(atoms), range(atoms))
        keep = src.ravel() != dst.ravel()
        src, dst = src.ravel()[keep], dst.ravel()[keep]
        out.append({"z": rng.choice([6, 8, 30], atoms), "src": src.astype(np.int64), "dst": dst.astype(np.int64),
                    "distance": rng.uniform(1.0, 4.9, len(src)).astype(np.float32), "cell": np.eye(3) * 9.0})
    return out


def ablation(cell, seed=17):
    torch.manual_seed(seed)
    return AblationMPNN(**CELLS[cell], **NORMALIZERS).eval()


def test_cells_cover_the_full_factorial_in_config_order():
    assert list(CELLS) == ["c5-hard-mean", "c5-env-mean", "c5-hard-sum", "c5-env-sum",
                           "c8-hard-mean", "c8-env-mean", "c8-hard-sum", "c8-env-sum"]
    assert len({tuple(v.values()) for v in CELLS.values()}) == 8


def test_baseline_cell_reproduces_g1_initialization_and_output():
    torch.manual_seed(17)
    g1 = PeriodicMPNN().eval()
    cell = ablation("c5-hard-mean")
    g1_state = g1.state_dict()
    for key, value in cell.state_dict().items():
        if key in g1_state:
            torch.testing.assert_close(value, g1_state[key], atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(cell(pack(graphs(), "cpu")), g1(pack_g1(graphs(), "cpu")), atol=0, rtol=0)


def test_full_cell_reproduces_g3_initialization_and_output():
    torch.manual_seed(17)
    g3 = SmoothPeriodicMPNN(**NORMALIZERS).eval()
    cell = ablation("c8-env-sum")
    for key, value in g3.state_dict().items():
        torch.testing.assert_close(cell.state_dict()[key], value, atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(cell(pack(graphs(), "cpu")), g3(pack(graphs(), "cpu")), atol=0, rtol=0)


def test_each_factor_changes_the_prediction():
    with torch.no_grad():
        base = ablation("c8-env-sum")(pack(graphs(), "cpu"))
        for other in ["c8-hard-sum", "c8-env-mean", "c5-env-sum"]:
            assert not torch.allclose(ablation(other)(pack(graphs(), "cpu")), base)


def test_parameter_counts_differ_only_by_the_density_input():
    counts = {cell: sum(p.numel() for p in ablation(cell).parameters()) for cell in CELLS}
    assert counts["c5-hard-mean"] == 80385 and counts["c8-env-sum"] == 80449
    assert {counts[c] for c in CELLS if c.endswith("mean")} == {80385}
    assert {counts[c] for c in CELLS if c.endswith("sum")} == {80449}

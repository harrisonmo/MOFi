"""Small independent invariance and rejection checks, charged to P0."""
from copy import deepcopy
import numpy as np

from .fixtures import transform
from .reference import verify, periodic_components
from .runtime import execute, manual_program, validate_structure, validate_program

NAMES = ["permutation", "origin", "wrap", "basis_identity", "basis_swap", "basis_shear",
         "edge_reverse", "periodic_chain", "finite_ring", "image_fault", "duplicate", "vacancy",
         "false_mapping", "false_image", "false_witness", "occupancy_change", "element_change",
         "position_change", "lattice_change", "unit", "nan", "singular", "shape", "edge_index",
         "fractional_image", "too_many_nodes", "cycle", "unknown_op", "manifest_bytes",
         "recursion", "array_shape", "table_ops", "report_binding", "empty_mapping", "malformed_cif"]


def toy():
    return dict(unit="angstrom", lattice=[[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]],
                frac=[[.1, .2, .3], [.4, .2, .3], [.8, .2, .3]], elements=["C", "C", "Zn"],
                occupancy=[1., 1., 1.], edges=[[0, 1, 0, 0, 0, 1.], [1, 2, 0, 0, 0, 1.], [2, 0, 1, 0, 0, 1.]])


def rejects(fn):
    try:
        fn()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError("Malformed input was accepted")


def check(name):
    a = toy()
    seed = {"basis_identity": 0, "basis_swap": 1, "basis_shear": 2}.get(name, 17)
    b, _ = transform(a, [0, 1, 2], seed)
    program = manual_program()
    if name in {"permutation", "origin", "wrap", "basis_identity", "basis_swap", "basis_shear", "edge_reverse"}:
        if name == "edge_reverse":
            b["edges"] = [[e[1], e[0], *[-x for x in e[2:5]], e[5]] for e in b["edges"]]
        output = execute(program, a, b)
        assert verify(a, b, output["evidence"])["verdict"] == "equivalent"
    elif name in {"periodic_chain", "finite_ring"}:
        if name == "finite_ring":
            a["edges"][-1][2] = 0
        assert periodic_components(a)[0]["periodic"] == (name == "periodic_chain")
        assert execute(program, a, a)["evidence"]["components"] == periodic_components(a)
    elif name in {"image_fault", "duplicate", "vacancy"}:
        b, _ = transform(a, [0, 2] if name == "vacancy" else [0, 1, 2], 17,
                         extra=0 if name == "duplicate" else None, fault=name == "image_fault")
        result = verify(a, b, execute(program, a, b)["evidence"], "A" if name == "image_fault" else "B")
        if name == "image_fault":
            assert result["verdict"] == "different" and result["removed_edges"] == result["added_edges"] == 1
        else:
            assert result["extra"] == (name == "duplicate") and result["missing"] == (name == "vacancy")
    elif name in {"false_mapping", "false_image", "false_witness", "occupancy_change", "element_change", "position_change", "lattice_change"}:
        evidence = execute(program, a, b)["evidence"]
        if name == "false_mapping":
            evidence["alignment"]["mapping"][0] = evidence["alignment"]["mapping"][1]
        elif name == "false_image":
            evidence["alignment"]["shifts"][0][0] += 1
        elif name == "false_witness":
            evidence["edge_delta"]["removed"] = a["edges"][:1]
        elif name == "occupancy_change":
            b["occupancy"][0] = .5
        elif name == "element_change":
            b["elements"][0] = "O"
        elif name == "position_change":
            b["frac"][0][0] += .001
        else:
            b["lattice"][0][0] += .1
        rejects(lambda: verify(a, b, evidence))
    elif name in {"unit", "nan", "singular", "shape", "edge_index", "fractional_image"}:
        if name == "unit":
            a["unit"] = "nm"
        elif name == "nan":
            a["frac"][0][0] = float("nan")
        elif name == "singular":
            a["lattice"][0] = [0, 0, 0]
        elif name == "shape":
            a["frac"].pop()
        elif name == "edge_index":
            a["edges"][0][0] = 3
        else:
            a["edges"][0][2] = .5
        rejects(lambda: validate_structure(a))
    elif name in {"too_many_nodes", "cycle", "unknown_op", "manifest_bytes", "recursion"}:
        if name == "too_many_nodes":
            program["nodes"] *= 2
        elif name == "cycle":
            program["nodes"][0]["inputs"][0] = "basis"
        elif name == "unknown_op":
            program["nodes"][0]["op"] = "eval"
        elif name == "manifest_bytes":
            program["nodes"][0]["config"] = " " * 32769
        else:
            program["nodes"][0]["op"] = "registered"
        rejects(lambda: validate_program(program, registered=deepcopy(program)))
    elif name == "array_shape":
        from .runtime import primitive
        rejects(lambda: primitive("distances", [dict(frac=[0] * 2048), dict(frac=[0] * 2048), {}, [0, 1]], {}))
    elif name == "table_ops":
        from .runtime import primitive
        table = [dict(k="a", v=-2), dict(k="b", v=3), dict(k="a", v=1)]
        assert primitive("project", [table], dict(fields=["k"])) == [{"k": x["k"]} for x in table]
        assert len(primitive("filter", [table], dict(field="k", equals="a"))) == 2
        assert primitive("sort", [table], dict(field="v"))[0]["v"] == -2
        assert primitive("group", [table], dict(field="k"))[0]["count"] == 2
        assert len(primitive("join", [table, table], dict(field="k"))) == 5
        assert primitive("map", [table], dict(field="v", operation="abs")) == [2, 3, 1]
    elif name in {"report_binding", "empty_mapping"}:
        from .scoring import score
        output = execute(program, a, b)
        pair = dict(A=a, B=b, case="A0", expected="equivalent")
        report = dict(verdict="equivalent", evidence={k: k for k in output["evidence"]},
                      missing=0, extra=0, removed_edges=0, added_edges=0)
        assert score(pair, output, report)["passed"]
        if name == "report_binding":
            report["evidence"]["alignment"] = "nonexistent"
        else:
            output["evidence"]["alignment"]["mapping"] = [-1] * 3
        assert not score(pair, output, report)["passed"]
    elif name == "malformed_cif":
        from pathlib import Path
        import tempfile
        from xtal.io import FORMATS
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.cif"
            path.write_text("This is not a crystallographic information file")
            try:
                FORMATS.read(path)
            except Exception:
                pass
            else:
                raise AssertionError("Malformed CIF was accepted")
    else:
        raise ValueError("Unknown check")
    return dict(name=name, passed=True)

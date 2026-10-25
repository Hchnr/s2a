import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coupled_gripper_preview_acceptance_checker import preview_mapping


def case_function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def preview_block(function):
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            starts = [
                i
                for i, n in enumerate(node.body)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "preview_index"
            ]
            if starts:
                stop = next(
                    i
                    for i, n in enumerate(node.body)
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "grip"
                )
                return node, starts[0], stop
    raise AssertionError("Explicit preview block missing")


def test_all_nonpreview_mechanical_checks_retained():
    old = case_function("scripts/gripper_preview_acceptance_checker.py")
    new = case_function("scripts/coupled_gripper_preview_acceptance_checker.py")
    old_loop, a, b = preview_block(old)
    new_loop, c, d = preview_block(new)
    text = ast.unparse(ast.Module(body=new_loop.body[c:d], type_ignores=[]))
    for name in (
        "position_span_max_m",
        "rotation_span_max_rad",
        "measured_position_max_m",
        "measured_rotation_max_rad",
    ):
        assert name in text
    assert "preview_mapping" in text and "measured @ h" in text
    new_loop.body[c:d] = old_loop.body[a:b]

    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "src/s2a/coupled_gripper_preview_simulation.py":
                node.value = "src/s2a/gripper_preview_simulation.py"
            return node

    new = Restore().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_nested_metadata_requires_exact_keys_boolean_and_integer_types():
    preview_mapping(
        {"guards": {"ready": True}, "index": 2, "error": 0.001}, {"guards": {"ready": True}, "index": 2, "error": 0.001}
    )
    for actual in (
        {"guards": {"ready": 1}, "index": 2, "error": 0.001},
        {"guards": {"ready": True}, "index": 2.0, "error": 0.001},
        {"guards": {"ready": True}, "index": 2, "error": 0.1},
    ):
        with pytest.raises(AssertionError):
            preview_mapping(actual, {"guards": {"ready": True}, "index": 2, "error": 0.001})

"""Keep every074 mechanical/state/task gate outside the080 planner replay."""

import ast
from pathlib import Path


def function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def index(nodes, name):
    return next(
        i
        for i, n in enumerate(nodes)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
    )


def test_only_planner_parameters_velocity_and_full_search_replay_change():
    old = function("scripts/residual_motor_acceptance_checker.py")
    new = function("scripts/velocity_model_acceptance_checker.py")
    runtime = index(new.body, "required_runtime")
    new.body[runtime] = old.body[index(old.body, "required_runtime")]
    import_index = next(
        i
        for i, n in enumerate(new.body)
        if isinstance(n, ast.ImportFrom) and n.module == "velocity_plan_independent_audit"
    )
    new.body[import_index] = next(
        n for n in old.body if isinstance(n, ast.ImportFrom) and n.module == "residual_motor_scalar_replay"
    )
    a, b = index(new.body, "envelope"), index(new.body, "previous")
    block = ast.unparse(ast.Module(body=new.body[a:b], type_ignores=[]))
    assert "evidence.check" in block and "gripper_algorithm" in block and "gripper_velocity_envelope" in block
    del new.body[a:b]
    old_loop = next(n for n in old.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    new_loop = next(n for n in new.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    a, b = index(new_loop.body, "measured_velocity"), index(new_loop.body, "command")
    block = ast.unparse(ast.Module(body=new_loop.body[a:b], type_ignores=[]))
    assert "audit_plan(" in block and "np.isfinite" in block and "row['measured_period_velocity']" in block
    assert "openings[step - chunk['step']:].tolist()" in block
    c, d = index(old_loop.body, "independent_grip"), index(old_loop.body, "command")
    new_loop.body[a:b] = old_loop.body[c:d]

    class Restore(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "response_parameters":
                node.id = "residual_parameters"
            return node

        def visit_Constant(self, node):
            if node.value == "response_parameters_ref":
                node.value = "residual_parameters_ref"
            return node

    new = Restore().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

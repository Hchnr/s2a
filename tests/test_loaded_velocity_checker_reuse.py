import ast
from pathlib import Path


def function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def test_all_nonrouting_control_state_and_task_checks_remain():
    old = function("scripts/velocity_model_acceptance_checker.py")
    new = function("scripts/loaded_velocity_acceptance_checker.py")

    def idx(body, name):
        return next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
        )

    r = idx(new.body, "required_runtime")
    assert "loaded_velocity_simulation.py" in ast.unparse(new.body[r]) and "loaded_velocity_control.py" in ast.unparse(
        new.body[r]
    )
    new.body[r] = old.body[idx(old.body, "required_runtime")]
    i = next(
        i
        for i, n in enumerate(new.body)
        if isinstance(n, ast.ImportFrom) and n.module == "loaded_velocity_independent_replay"
    )
    new.body[i] = next(
        n for n in old.body if isinstance(n, ast.ImportFrom) and n.module == "velocity_plan_independent_audit"
    )
    i = idx(new.body, "loaded_replay")
    assert "gripper_position_hold" in ast.unparse(new.body[i])
    del new.body[i]
    old_loop = next(n for n in old.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    new_loop = next(n for n in new.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    i = next(
        i
        for i, n in enumerate(new_loop.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "loaded_replay.step"
    )
    replay = ast.unparse(new_loop.body[i])
    assert "previous[-1]" in replay and "measured @ h" in replay and "chunk['effective_targets_tcp']" in replay
    new_loop.body[i] = next(
        n
        for n in old_loop.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "audit_plan"
    )

    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "081 native-close loading and frozen079 tracking":
                node.value = "080 frozen079 finite-command planner"
            return node

    new = Restore().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

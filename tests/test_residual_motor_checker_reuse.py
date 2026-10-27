"""Retain every inherited mechanical/state/task check outside residual replay."""

import ast
from pathlib import Path


def function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def test_checker_only_runtime_and_gripper_replay_change():
    old = function("scripts/motor_state_acceptance_checker.py")
    new = function("scripts/residual_motor_acceptance_checker.py")

    def idx(nodes, name):
        return next(
            i
            for i, n in enumerate(nodes)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
        )

    runtime = idx(new.body, "required_runtime")
    assert "residual_motor_simulation.py" in ast.unparse(new.body[runtime])
    new.body[runtime] = old.body[idx(old.body, "required_runtime")]
    before = idx(new.body, "gripper") + 1
    new.body.insert(before, old.body[idx(old.body, "scalar_gripper")])
    a = next(
        i
        for i, n in enumerate(new.body)
        if isinstance(n, ast.ImportFrom) and n.module == "residual_motor_scalar_replay"
    )
    b = idx(new.body, "previous")
    block = ast.unparse(ast.Module(body=new.body[a:b], type_ignores=[]))
    assert "evidence.check" in block and "residual_parameters_ref" in block and "source == 'oracle'" in block
    del new.body[a:b]
    old_loop = next(n for n in old.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    new_loop = next(n for n in new.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(step, row)")
    a, b = idx(new_loop.body, "independent_grip"), idx(new_loop.body, "command")
    replay = ast.unparse(ast.Module(body=new_loop.body[a:b], type_ignores=[]))
    assert "rows[step - 1]['opening_before']" in replay and "openings[step - chunk['step']:]" in replay
    assert "mapping" in replay and "row['command'][-1]" in replay
    c, d = idx(old_loop.body, "grip"), idx(old_loop.body, "command")
    new_loop.body[a:b] = old_loop.body[c:d]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

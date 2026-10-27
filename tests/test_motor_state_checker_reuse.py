import ast
from pathlib import Path


def load(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def assignment_index(function, name):
    return next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
    )


def test_only_initial_motor_admission_and_runtime_identity_change():
    old = load("scripts/training_recovery_acceptance_checker.py")
    new = load("scripts/motor_state_acceptance_checker.py")
    a, b = assignment_index(old, "target"), assignment_index(old, "dt")
    c, d = assignment_index(new, "initial_motor"), assignment_index(new, "dt")
    block = ast.unparse(ast.Module(body=new.body[c:d], type_ignores=[]))
    assert "native_initial_motor_target" in block and "preserved_motor" in block
    assert "np.isfinite" in block and "motor_initialization" in block
    assert "preserve current target; scalar reset broadcast only" in block
    assert len(new.body[c:d]) == 5
    new.body[c:d] = old.body[a:b]

    class Restore(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "src/s2a/motor_state_simulation.py":
                node.value = "src/s2a/training_recovery_simulation.py"
            return node

    new = Restore().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

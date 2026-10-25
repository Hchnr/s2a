import ast
from pathlib import Path


def test_only_motor_initialization_and_its_receipt_change():
    old = next(
        n
        for n in ast.parse(Path("src/s2a/training_recovery_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training_recovery_case"
    )
    new = next(
        n
        for n in ast.parse(Path("src/s2a/motor_state_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_motor_state_case"
    )
    new.name = old.name
    changes = []

    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "preserve_motor_target":
                assert ast.unparse(node.args[0]) == "native_initial_motor"
                changes.append("current_motor_target")
                return ast.parse("opening_position_target(initial_opening)", mode="eval").body
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == "preserve current target; scalar reset broadcast only":
                changes.append("receipt")
                node.value = "observed aperture alignment, not hidden expert motor state"
            return node

    new = Restore().visit(new)
    assert changes == ["current_motor_target", "receipt"]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

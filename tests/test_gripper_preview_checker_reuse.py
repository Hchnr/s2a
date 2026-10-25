import ast
from pathlib import Path


def case_function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def test_preview_verifier_keeps_full_mechanical_checks():
    old = case_function("scripts/training_recovery_acceptance_checker.py")
    new = case_function("scripts/gripper_preview_acceptance_checker.py")
    removed = []
    replacements = []

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in ("preview_index", "preview_stop", "preview_advance", "preview_selected", "command_opening")
            ):
                removed.append(node.targets[0].id)
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "close"
                and ast.unparse(node.value.args[0]) == "row['command_opening_reference']"
            ):
                removed.append("command_log_check")
                return None
            return self.generic_visit(node)

        def visit_Assert(self, node):
            if ast.unparse(node.test).startswith("row['gripper_preview'] == dict("):
                removed.append("provenance_check")
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) in ("gripper.update", "scalar_gripper.step"):
                assert ast.unparse(node.args[0]) == "command_opening"
                replacements.append(ast.unparse(node.func))
                node.args[0] = ast.parse("row['opening_target']", mode="eval").body
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == "src/s2a/gripper_preview_simulation.py":
                node.value = "src/s2a/training_recovery_simulation.py"
            return node

    new = Restore().visit(new)
    assert removed == [
        "preview_index",
        "preview_stop",
        "preview_advance",
        "preview_selected",
        "command_opening",
        "command_log_check",
        "provenance_check",
    ]
    assert replacements == ["gripper.update", "scalar_gripper.step"]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

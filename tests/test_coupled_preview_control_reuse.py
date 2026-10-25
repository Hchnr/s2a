import ast
from pathlib import Path


def test_coupling_changes_only_explicit_preview_selection():
    old = next(
        n
        for n in ast.parse(Path("src/s2a/gripper_preview_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_gripper_preview_case"
    )
    new = next(
        n
        for n in ast.parse(Path("src/s2a/coupled_gripper_preview_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_coupled_gripper_preview_case"
    )
    new.name = old.name
    calls = []

    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "coupled_opening_preview":
                calls.append(ast.unparse(node))
                return ast.parse(
                    "opening_preview(full_openings, index - 1, preview_steps=config['gripper_preview_steps'])",
                    mode="eval",
                ).body
            return self.generic_visit(node)

    new = Restore().visit(new)
    assert calls == [
        "coupled_opening_preview(full_effective_targets, full_openings, index - 1, measured_tcp, preview_steps=config['gripper_preview_steps'], **config['coupled_preview_bounds'])"
    ]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

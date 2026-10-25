"""Protect all original arm and gripper logic outside the preview seam."""

import ast
from pathlib import Path


def test_preview_changes_only_command_reference_and_explicit_log_fields():
    old = next(
        n
        for n in ast.parse(Path("src/s2a/training_recovery_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training_recovery_case"
    )
    new = next(
        n
        for n in ast.parse(Path("src/s2a/gripper_preview_simulation.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_gripper_preview_case"
    )
    new.name = old.name
    guards = []
    previews = []
    substitutions = []

    class Restore(ast.NodeTransformer):
        def visit_If(self, node):
            if (
                ast.unparse(node.test) in ("config['gripper_preview_steps'] not in (0, 6)", "variant['governor']")
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Raise)
            ):
                guards.append(ast.unparse(node.test))
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "preview":
                previews.append(ast.unparse(node.value))
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "gripper.update":
                assert ast.unparse(node.args[0]) == "preview['reference']"
                substitutions.append("command_reference")
                node.args[0] = ast.Name(id="desired_opening", ctx=ast.Load())
            if isinstance(node.func, ast.Name) and node.func.id == "dict":
                extra = [k for k in node.keywords if k.arg in ("command_opening_reference", "gripper_preview")]
                if extra:
                    assert [(k.arg, ast.unparse(k.value)) for k in extra] == [
                        ("command_opening_reference", "preview['reference']"),
                        ("gripper_preview", "preview"),
                    ]
                    substitutions.append("provenance_logs")
                    node.keywords = [k for k in node.keywords if k not in extra]
            return self.generic_visit(node)

    new = Restore().visit(new)
    assert guards == ["config['gripper_preview_steps'] not in (0, 6)", "variant['governor']"]
    assert previews == ["opening_preview(full_openings, index - 1, preview_steps=config['gripper_preview_steps'])"]
    assert substitutions == ["command_reference", "provenance_logs"]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

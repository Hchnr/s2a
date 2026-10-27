"""Prove the ablation retains083 arm execution and non-policy mechanical checks."""

import ast
from pathlib import Path


def function(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = "case"
        node.args.kwonlyargs = [
            x for x in node.args.kwonlyargs if x.arg not in ("response_parameters", "native_gripper_commands")
        ]
        node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        name = ast.unparse(node.targets[0])
        if name in (
            "loaded_controller",
            "previous_gripper_command",
            "grip_result",
            "grip_command",
            "native_gripper_commands",
            "loaded_replay",
            "response_parameters",
            "expected_native",
        ):
            return None
        if name == "required_runtime":
            node.value = ast.Constant(None)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        value = ast.unparse(node.value)
        if value.startswith(("VelocityModelPlanner(", "loaded_replay.step(", "evidence.json(parameter_ref[")):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if "native_gripper_commands" in ast.unparse(node.test):
            return None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        return None if node.module == "loaded_velocity_independent_replay" else node

    def visit_Assert(self, node):
        text = ast.unparse(node.test)
        if any(
            x in text
            for x in (
                "row['gripper']",
                "row['gripper_command_is_oracle']",
                "expected_native",
                "native_gripper_commands",
                "controller['gripper_algorithm']",
            )
        ):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [
            k
            for k in node.keywords
            if k.arg not in ("gripper", "gripper_command_is_oracle", "gripper_algorithm", "online_control_inputs")
        ]
        return self.generic_visit(node)


def normalized(path, name):
    n = Normalize().visit(function(path, name))
    return ast.dump(n, include_attributes=False)


def test_full_execution_preserves083_arm_observations_reference_and_endpoint_recording():
    assert normalized("src/s2a/pose_velocity_loaded_simulation.py", "run_pose_velocity_loaded_case") == normalized(
        "src/s2a/native_gripper_ablation_simulation.py", "run_native_gripper_ablation_case"
    )


def test_full_checker_preserves083_arm_labels_images_motor_and_task_checks():
    assert normalized("scripts/pose_velocity_loaded_acceptance_checker.py", "verify_case") == normalized(
        "scripts/native_gripper_ablation_acceptance_checker.py", "verify_case"
    )

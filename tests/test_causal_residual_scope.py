import ast
from pathlib import Path


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def test_control_preserves083_except_arm_converter_and_residual_logging():
    old = functions(Path("src/s2a/pose_velocity_loaded_simulation.py").read_text())
    text = Path("src/s2a/causal_residual_loaded_simulation.py").read_text()
    text = text.replace("run_causal_residual_loaded_case", "run_pose_velocity_loaded_case").replace(
        "CausalArmResidualOSC", "PoseVelocityInverseOSC"
    )
    text = text.replace(
        "085 lag1 causal response residual; 083 nominal inverse",
        "083 position-velocity dt-scaled translation rotation0.5",
    )
    new = functions(text)

    class StripLog(ast.NodeTransformer):
        def visit_Call(self, node):
            node.keywords = [
                k
                for k in node.keywords
                if k.arg not in ("nominal_predicted_linear_velocity", "nominal_predicted_motion", "response_residual")
            ]
            return self.generic_visit(node)

    new = {name: StripLog().visit(node) for name, node in new.items()}
    assert {k: ast.dump(v, include_attributes=False) for k, v in old.items()} == {
        k: ast.dump(v, include_attributes=False) for k, v in new.items()
    }


def test_checker_preserves_complete083_mechanical_checks():
    old = functions(Path("scripts/pose_velocity_loaded_acceptance_checker.py").read_text())["verify_case"]
    text = Path("scripts/causal_residual_loaded_acceptance_checker.py").read_text()
    text = text.replace("causal_arm_independent_replay", "pose_velocity_arm_independent_replay").replace(
        "CausalArmReplay", "PoseVelocityArmReplay"
    )
    text = text.replace(
        "085 lag1 causal response residual; 083 nominal inverse",
        "083 position-velocity dt-scaled translation rotation0.5",
    )
    new = functions(text)["verify_case"]
    required = next(
        n for n in old.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "required_runtime"
    )

    class StripAudit(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = ast.unparse(node.targets[0])
            if name == "required_runtime":
                return required
            if "saved_residual" in name:
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            text = ast.unparse(node)
            if text.startswith("close(row[") and any(
                x in text for x in ("nominal_predicted_linear_velocity", "nominal_predicted_motion")
            ):
                return None
            return self.generic_visit(node)

        def visit_Assert(self, node):
            if "saved_residual" in ast.unparse(node):
                return None
            return self.generic_visit(node)

        def visit_For(self, node):
            if ast.unparse(node.iter) == "expected_residual.items()":
                return None
            return self.generic_visit(node)

    new = StripAudit().visit(new)
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

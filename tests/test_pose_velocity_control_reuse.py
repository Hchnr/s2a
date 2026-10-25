import ast
from pathlib import Path


def test_complete_loop_changes_only_registered_arm_objective_and_reference_velocity():
    old = Path("src/s2a/translation_loaded_simulation.py").read_text()
    new = Path("src/s2a/pose_velocity_loaded_simulation.py").read_text()
    new = new.replace("def run_pose_velocity_loaded_case(", "def run_translation_loaded_case(").replace(
        "        converter = PoseVelocityInverseOSC(", "        converter = TranslationInverseOSC("
    )
    new = new.replace("083 position-velocity dt-scaled translation rotation0.5", "082 translation1 rotation0.5")
    block = """                velocity_reference = reference_linear_velocity(
                    full_effective_targets @ h_inv, index - 1, query_anchor @ h_inv, dt
                )
                converted = converter.convert(
                    measured, desired, twist, desired_linear_velocity=velocity_reference["velocity"]
                )"""
    assert new.count(block) == 1
    new = new.replace(block, "                converted = converter.convert(measured, desired, twist)")
    new = new.replace(
        '                    linear_velocity_reference=velocity_reference,\n                    predicted_linear_velocity=converted["predicted_linear_velocity"],\n',
        "",
    )

    def functions(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(new) == functions(old)

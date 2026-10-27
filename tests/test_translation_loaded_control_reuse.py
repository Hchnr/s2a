import ast
from pathlib import Path


def test_complete081_control_loop_changes_only_translation_inverse_and_receipt():
    old = Path("src/s2a/loaded_velocity_simulation.py").read_text()
    new = Path("src/s2a/translation_loaded_simulation.py").read_text()
    assert new.count("        converter = TranslationInverseOSC(") == 1
    new = new.replace("def run_translation_loaded_case(", "def run_loaded_velocity_case(").replace(
        "        converter = TranslationInverseOSC(", "        converter = OSCStateToAction("
    )
    new = new.replace('                arm_inverse_algorithm="082 translation1 rotation0.5",\n', "")

    def functions(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(new) == functions(old)

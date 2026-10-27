import ast
from pathlib import Path


def function(path):
    return next(
        n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
    )


def test_all_other081_mechanical_and_route_checks_remain():
    old = function("scripts/loaded_velocity_acceptance_checker.py")
    new = function("scripts/translation_loaded_acceptance_checker.py")

    def idx(body, name):
        return next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
        )

    i = idx(new.body, "required_runtime")
    assert "translation_inverse_osc.py" in ast.unparse(new.body[i])
    new.body[i] = old.body[idx(old.body, "required_runtime")]
    i = next(
        i
        for i, n in enumerate(new.body)
        if isinstance(n, ast.ImportFrom) and n.module == "translation_arm_independent_replay"
    )
    assert isinstance(new.body[i + 1], ast.Assert) and "arm_inverse_algorithm" in ast.unparse(new.body[i + 1])
    del new.body[i : i + 2]
    arm = new.body[idx(new.body, "arm")]
    assert ast.unparse(arm.value.func) == "TranslationArmReplay"
    arm.value.func.id = "OSCStateToAction"
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

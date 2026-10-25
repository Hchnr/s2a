import ast
from pathlib import Path

import numpy as np
from pose_velocity_loaded_acceptance_checker import independent_reference_velocity

from s2a.reference_linear_velocity import reference_linear_velocity


def test_reference_modes_match_independent_replay():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = np.arange(12).reshape(4, 3) ** 2 / 100
    anchor = np.eye(4)
    for n in (1, 4):
        for i in range(n):
            actual = reference_linear_velocity(poses[:n], i, anchor, 0.05)
            expected = independent_reference_velocity(poses[:n], i, anchor, 0.05)
            np.testing.assert_allclose(actual.pop("velocity"), expected.pop("velocity"))
            assert actual == expected


def test_existing_mechanical_checks_preserved():
    def function(path):
        return next(
            n
            for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "verify_case"
        )

    old = function("scripts/translation_loaded_acceptance_checker.py")
    new = function("scripts/pose_velocity_loaded_acceptance_checker.py")
    text = ast.unparse(new)
    text = text.replace("pose_velocity_arm_independent_replay", "translation_arm_independent_replay")
    text = text.replace("PoseVelocityArmReplay", "TranslationArmReplay")
    text = text.replace("083 position-velocity dt-scaled translation rotation0.5", "082 translation1 rotation0.5")
    new = ast.parse(text).body[0]
    for node in ast.walk(new):
        if not hasattr(node, "body") or not isinstance(node.body, list):
            continue
        body = node.body
        for i, item in enumerate(body):
            if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == "required_runtime":
                item.value = next(
                    x.value
                    for x in old.body
                    if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "required_runtime"
                )
            if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == "generated_source":
                end = next(
                    j
                    for j in range(i, len(body))
                    if isinstance(body[j], ast.Expr) and "predicted_linear_velocity" in ast.unparse(body[j])
                )
                body[i : end + 1] = ast.parse(
                    'reconstructed = arm.convert(measured, desired, row["measured_twist"])'
                ).body
                break
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)

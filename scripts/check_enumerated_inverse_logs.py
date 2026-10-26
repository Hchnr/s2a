"""161 offline counterfactual solve on saved current states; no physical replay."""

import json
from pathlib import Path

import numpy as np
from qualified_inverse_independent import verify_inverse

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.qualified_enumerated_inverse import inverse_translation


def main():
    root = REPO / "outputs/enumerated_inverse_logs_20261025"
    if root.exists():
        raise FileExistsError(root)
    parameters_path = REPO / "outputs/left_table_fit_20261025/parameters.json"
    parameters = json.loads(parameters_path.read_text())
    sources = [parameters_path, Path(__file__).resolve()]
    sources += [
        REPO / p
        for p in (
            "src/s2a/enumerated_response_inverse.py",
            "src/s2a/qualified_enumerated_inverse.py",
            "src/s2a/joint_wrench_condition_forward.py",
            "scripts/qualified_inverse_independent.py",
            "scripts/arm_joint_wrench_independent.py",
        )
    ]
    results = []
    for namespace in ("qualified_inverse_control_20261025", "qualified_inverse_native_precision_20261025"):
        files = list((REPO / "outputs" / namespace / "case00").rglob("steps.jsonl"))
        assert len(files) == 1
        path = files[0]
        controller_path = path.with_name("controller.json")
        sources += [path, controller_path]
        contract = json.loads(controller_path.read_text())["contract"]
        bounds = {k: np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")}
        worst_difference = 0.0
        count = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
            desired = np.asarray(row["desired_site"])[:3, 3] - data["x_site_pose"][0, :3, 3]
            velocity = np.asarray(row["site_velocity_reference"]["velocity"])
            previous = data["x_previous_command"][0, :3]
            solved = inverse_translation(data, parameters, desired, velocity, previous_command=previous, **bounds)
            checked = verify_inverse(data, parameters, contract, previous, desired, velocity, solved["command"])
            np.testing.assert_allclose(solved["squared_objective"], checked["objective"], atol=1e-12, rtol=0)
            old = row["qualified_forward_inverse"]["squared_objective"]
            assert checked["objective"] <= old + 1e-12
            worst_difference = max(worst_difference, abs(checked["objective"] - old))
            count += 1
        results.append(dict(namespace=namespace, checked_rows=count, max_objective_difference=worst_difference))
    assert [r["checked_rows"] for r in results] == [164, 90]
    publish(
        root / "verified.json",
        dict(
            checked=True,
            results=results,
            physics_steps=0,
            control_qualified=False,
            failed_solver_input_available=False,
            limitation="Only completed logged states available; not an exact reproduction of failed period90 input or a closed-loop qualification",
            source_sha256={str(p): sha(p) for p in sources},
        ),
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()

"""163 diagnose completed case00 condition extrapolation without new physics or fitting."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.joint_wrench_condition_forward import raw_condition
from s2a.qualified_enumerated_inverse import affine_response


def main():
    root = REPO / "outputs/inverse_control_departure_20261025"
    if root.exists():
        raise FileExistsError(root)
    case = REPO / "outputs/qualified_inverse_enumerated_20261025/case00"
    verified = case / "independent_verified.json"
    assert json.loads(verified.read_text())["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    controller_path = path.with_name("controller.json")
    controller = json.loads(controller_path.read_text())
    contract = controller["contract"]
    mapping = np.asarray(controller["source_from_site"])
    bounds = {k: np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")}
    parameter_path = REPO / "outputs/left_table_fit_20261025/parameters.json"
    parameters = json.loads(parameter_path.read_text())
    mean, scale = (np.asarray(parameters[k]) for k in ("state_mean", "state_scale"))
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
        z = ((raw_condition(data) - mean) / scale)[0]
        actual = (np.asarray(row["after_source"]) @ mapping)[:3, 3] - data["x_site_pose"][0, :3, 3]
        jacobian = np.asarray(row["qualified_forward_inverse"]["response_jacobian"])
        counterfactual = dict(data)
        counterfactual["x_response_residual"] = np.zeros((1, 12))
        no_history = affine_response(counterfactual, parameters, **bounds)["jacobian"]
        rows.append(
            dict(
                step=row["step"],
                position_error_m=float(np.linalg.norm(np.asarray(row["source_error"])[:3])),
                so3_error_rad=float(np.linalg.norm(np.asarray(row["source_error"])[3:])),
                motion_prediction_error_m=float(
                    np.linalg.norm(np.asarray(row["qualified_forward_inverse"]["predicted_motion"]) - actual)
                ),
                translation_jacobian_diagonal=jacobian[:3].diagonal().tolist(),
                zero_innovation_counterfactual_diagonal=no_history[:3].diagonal().tolist(),
                standardized_condition_max=float(np.max(np.abs(z))),
                standardized_top_indices=np.argsort(np.abs(z))[-5:][::-1].tolist(),
                standardized_history_max=float(np.max(np.abs(z[71:83]))),
                standardized_wrench_max=float(np.max(np.abs(z[83:89]))),
            )
        )
    sources = [Path(__file__).resolve(), path, controller_path, parameter_path, verified]
    sources += [
        REPO / p
        for p in (
            "src/s2a/qualified_enumerated_inverse.py",
            "src/s2a/joint_wrench_condition_forward.py",
            "src/s2a/history_conditioned_arm_forward.py",
            "src/s2a/wrench_conditioned_arm_forward.py",
            "src/s2a/state_conditioned_arm_forward.py",
        )
    ]
    publish(
        root / "result.json",
        dict(
            complete=True,
            rows=rows,
            source_sha256={str(p): sha(p) for p in sources},
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Zero-innovation substitution is a diagnostic counterfactual, not an admissible replacement controller or proof of causal physical derivative",
        ),
    )
    print(json.dumps([r for r in rows if r["step"] in (70, 74, 78, 80, 84, 90)]))


if __name__ == "__main__":
    main()

"""217 independent braking Jacobian, rate feasibility and actual endpoint audit."""

import json

import numpy as np
from arm_joint_wrench_independent import nominal_prediction
from audit_wam_position import Evidence
from late_clipped_arm_inverse_independent import features, inputs, nominal

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def load_case(e, root):
    verified = e.json(root / "case00/independent_verified.json")
    assert verified["complete"] and verified["mechanical_labels_images_passed"]
    case = next((root / "case00/simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    controller = e.json(case / "controller.json")
    return verified, rows, controller


def main():
    root = REPO / "outputs/late_clip_dynamic_departure_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/late_clipped_arm_inverse_independent.py",
        "scripts/arm_joint_wrench_independent.py",
        "plans/v0.0.1/217_late_clip_dynamic_departure_20261026.md",
    ):
        e.read(path)
    config = e.json("configs/late_clipped_inverse_control_registered_20261026.json")
    for key in ("direct_inverse_parameter_ref", "arm_forward_parameter_ref"):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(config["direct_inverse_parameter_ref"]["path"])
    physical = e.json(config["arm_forward_parameter_ref"]["path"])
    verified, rows, controller = load_case(e, REPO / "outputs/late_clipped_inverse_control_20261026")
    prior, prior_rows, _ = load_case(e, REPO / "outputs/causal_residual_loaded_control_20261025")
    assert len(rows) == len(prior_rows) == 164
    for row, old in zip(rows, prior_rows, strict=True):
        np.testing.assert_allclose(row["desired_source"], old["desired_source"], atol=1e-9, rtol=0)
    coefficient, scale, mean, frequency = (
        np.asarray(parameters[key])
        for key in ("coefficients", "normalization_scale", "normalization_mean", "frequencies")
    )
    lo, hi, outlo, outhi = (
        np.asarray(controller["contract"][key])[:3] for key in ("input_min", "input_max", "output_min", "output_max")
    )
    response = np.asarray(parameters["translation_response"])
    momentum = np.asarray(parameters["translation_momentum"])
    gain = (hi - lo) / (outhi - outlo) / response
    observations = []
    for index, row in enumerate(rows):
        data = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
        target = np.asarray(row["desired_site"])[None]
        grip = np.array([row["command"][-1]])
        value = inputs(data, target, grip)
        z = (value - mean) / scale
        phase = z @ frequency.T
        f = frequency[:, 37:40] / scale[37:40]
        jacobian = coefficient[38:41].T / scale[37:40]
        jacobian = jacobian + coefficient[90:154].T @ (-np.sin(phase[0, :, None]) * f / 8)
        jacobian += coefficient[154:218].T @ (np.cos(phase[0, :, None]) * f / 8)
        difference = np.empty((3, 3))
        for axis in range(3):
            before, after = {k: v.copy() for k, v in data.items()}, {k: v.copy() for k, v in data.items()}
            before["x_site_twist"][0, axis] -= 1e-7
            after["x_site_twist"][0, axis] += 1e-7
            difference[:, axis] = (
                (features(inputs(after, target, grip), parameters) - features(inputs(before, target, grip), parameters))
                @ coefficient
            )[0] / 2e-7
        np.testing.assert_allclose(jacobian, difference, atol=1e-6, rtol=0)
        net_velocity = jacobian - np.diag(gain * momentum)
        previous = data["x_previous_command"][0, :3]
        lower, upper = np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25)
        raw_nominal = nominal(data, target, controller["contract"])[0]
        nearest = np.minimum(upper, np.maximum(lower, raw_nominal))
        delta = outlo + (nearest - lo) * (outhi - outlo) / (hi - lo)
        pose, twist = data["x_site_pose"][0], data["x_site_twist"][0]
        minimum = np.linalg.norm(target[0, :3, 3] - pose[:3, 3] - momentum * twist[:3] - response * delta)
        model_prediction = nominal_prediction(data, physical)[0]
        after = np.asarray(row["after_source"]) @ np.asarray(controller["source_from_site"])
        actual_displacement = after[:3, 3] - pose[:3, 3]
        observations.append(
            dict(
                step=index,
                actual_endpoint_error_m=float(np.linalg.norm(row["tcp_error"][:3])),
                prior085_endpoint_error_m=float(np.linalg.norm(prior_rows[index]["tcp_error"][:3])),
                rate_limited=bool(row["direct_inverse"]["rate_limited"]),
                nominal_model_rate_feasible_minimum_error_m=float(minimum),
                nominal_displacement_prediction_error_m=float(
                    np.linalg.norm(actual_displacement - model_prediction[:3])
                ),
                net_partial_velocity_symmetric_eigenvalues=np.linalg.eigvalsh(
                    (net_velocity + net_velocity.T) / 2
                ).tolist(),
                max_velocity_jacobian_difference=float(np.max(np.abs(jacobian - difference))),
                raw_nominal=raw_nominal.tolist(),
                command=row["command"][:3],
            )
        )
    largest = sorted(observations, key=lambda row: row["actual_endpoint_error_m"], reverse=True)[:10]
    summary = dict(
        rate_limited_rows=sum(row["rate_limited"] for row in observations),
        actual_error_above_1cm_rows=sum(row["actual_endpoint_error_m"] > 0.01 for row in observations),
        largest_velocity_symmetric_eigenvalue=max(
            max(row["net_partial_velocity_symmetric_eigenvalues"]) for row in observations
        ),
        high_error_rate_limited_rows=sum(
            row["rate_limited"] and row["actual_endpoint_error_m"] > 0.01 for row in observations
        ),
        prior085_control=prior["control"],
        late_clip_control=verified["control"],
    )
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            summary=summary,
            peak_rows=largest,
            observations=observations,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            seed=42,
            closed_loop_cause_uniquely_identified=False,
        ),
    )
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith("control")}))


if __name__ == "__main__":
    main()

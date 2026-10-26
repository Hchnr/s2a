"""214 diagnose the executed direct inverse without fitting or changing control."""

import json

import numpy as np
from audit_wam_position import Evidence
from direct_arm_inverse_independent import features, inputs, nominal

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/direct_inverse_departure_20261026"
    assert not root.exists()
    e = Evidence()
    e.read(__file__)
    e.read("scripts/direct_arm_inverse_independent.py")
    run = REPO / "outputs/direct_inverse_control_20261026"
    config = e.json("configs/direct_inverse_control_registered_20261026.json")
    ref = config["direct_inverse_parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    parameters = e.json(ref["path"])
    verified = e.json(run / "case00/independent_verified.json")
    assert verified["complete"] and verified["mechanical_labels_images_passed"]
    assert not verified["performance_qualified"]
    case = next((run / "case00/simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    mean, scale, coef = (
        np.asarray(parameters[k]) for k in ("normalization_mean", "normalization_scale", "coefficients")
    )
    observations = []
    for row in rows:
        data = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
        desired = np.asarray(row["desired_site"])[None]
        value = inputs(data, desired, np.array([row["command"][-1]]))
        phi = features(value, parameters)
        residual = (phi @ coef)[0]
        baseline = nominal(data, desired, parameters["controller_contract"])[0]
        np.testing.assert_allclose(residual, row["direct_inverse"]["residual_command"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(baseline, row["direct_inverse"]["nominal_command"], atol=1e-9, rtol=0)
        command = np.asarray(row["command"])[:3]
        z = (value[0] - mean) / scale
        # Diagnostic: inversion relative to a saturated baseline, not a theorem
        # that every instantaneous opposing command is physically incorrect.
        opposing = (np.abs(baseline) >= 1 - 1e-12) & (baseline * command < 0)
        observations.append(
            dict(
                step=row["step"],
                endpoint_position_error_m=float(np.linalg.norm(row["tcp_error"][:3])),
                command=command.tolist(),
                nominal_command=baseline.tolist(),
                residual_command=residual.tolist(),
                residual_constant=coef[0].tolist(),
                residual_measured_and_history=(phi[0, 1:84] @ coef[1:84]).tolist(),
                residual_desired_motion=(phi[0, 84:90] @ coef[84:90]).tolist(),
                residual_fourier=(phi[0, 90:] @ coef[90:]).tolist(),
                max_abs_standardized_input=float(np.max(np.abs(z))),
                largest_standardized_input_index=int(np.argmax(np.abs(z))),
                saturated_baseline_opposing_executed_axes=np.flatnonzero(opposing).tolist(),
            )
        )
    assert len(observations) == 164
    first = {}
    for threshold in (0.005, 0.01, 0.05, 0.1):
        first[f"endpoint_error_above_{threshold:g}m"] = next(
            (r["step"] for r in observations if r["endpoint_position_error_m"] > threshold), None
        )
    first["residual_component_above_one"] = next(
        (r["step"] for r in observations if max(abs(v) for v in r["residual_command"]) > 1), None
    )
    first["saturated_baseline_opposes_executed_axis"] = next(
        (r["step"] for r in observations if r["saturated_baseline_opposing_executed_axes"]), None
    )
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            case=0,
            independently_recomputed_rows=164,
            first_events=first,
            observations=observations,
            source_sha256=e.hashes,
            parameter_ref=config["direct_inverse_parameter_ref"],
            physics_steps=0,
            new_fits=0,
            control_qualified=False,
            attribution="descriptive execution evidence; does not uniquely identify physical cause",
        ),
    )
    print(json.dumps(first))


if __name__ == "__main__":
    main()

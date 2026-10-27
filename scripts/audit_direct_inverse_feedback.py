"""215 analytic desired-position feedback bound and independent finite differences."""

import json

import numpy as np
from audit_wam_position import Evidence
from direct_arm_inverse_independent import features, inputs

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/direct_inverse_feedback_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/direct_arm_inverse_independent.py",
        "plans/v0.0.1/215_direct_inverse_feedback_audit_20261026.md",
    ):
        e.read(path)
    fit = e.json("outputs/direct_arm_inverse_fit_20261026/independent_verified.json")
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    p = e.json(ref["path"])
    coefficients, scale, mean, frequencies = (
        np.asarray(p[key]) for key in ("coefficients", "normalization_scale", "normalization_mean", "frequencies")
    )
    linear = coefficients[84:87].T / scale[83:86]
    position_frequencies = frequencies[:, 83:86] / scale[83:86]
    bound = sum(
        (np.linalg.norm(coefficients[90 + k]) + np.linalg.norm(coefficients[154 + k]))
        * np.linalg.norm(position_frequencies[k])
        / 8
        for k in range(64)
    )
    lo, hi, outlo, outhi = (
        np.asarray(p["translation_" + key]) for key in ("input_min", "input_max", "output_min", "output_max")
    )
    gain = (hi - lo) / (outhi - outlo) / np.asarray(p["translation_response"])
    lower_bound = float(np.linalg.eigvalsh(np.diag(gain) + (linear + linear.T) / 2)[0] - bound)
    case = next((REPO / "outputs/direct_inverse_control_20261026/case00/simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    checks = []
    for index in (0, 40, 48, 60, 67, 70, 80, 163):
        row = rows[index]
        data = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
        target = np.asarray(row["desired_site"])[None]
        grip = np.array([row["command"][-1]])
        value = inputs(data, target, grip)
        phase = ((value - mean) / scale) @ frequencies.T
        jacobian = linear + coefficients[90:154].T @ (-np.sin(phase[0, :, None]) * position_frequencies / 8)
        jacobian += coefficients[154:218].T @ (np.cos(phase[0, :, None]) * position_frequencies / 8)
        independent = np.empty((3, 3))
        for axis in range(3):
            minus, plus = target.copy(), target.copy()
            minus[0, axis, 3] -= 1e-7
            plus[0, axis, 3] += 1e-7
            independent[:, axis] = (
                (features(inputs(data, plus, grip), p) - features(inputs(data, minus, grip), p)) @ coefficients
            )[0] / 2e-7
        np.testing.assert_allclose(jacobian, independent, atol=1e-6, rtol=0)
        checks.append(dict(step=index, max_jacobian_difference=float(np.max(np.abs(jacobian - independent)))))
    assert lower_bound > 0
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            parameter_ref=ref,
            residual_position_linear_jacobian=linear.tolist(),
            residual_linear_symmetric_eigenvalues=np.linalg.eigvalsh((linear + linear.T) / 2).tolist(),
            physical_position_gain=gain.tolist(),
            fourier_jacobian_norm_upper=float(bound),
            late_clip_preprojection_symmetric_jacobian_lower=lower_bound,
            finite_difference_checks=checks,
            final_projection_strong_monotonicity_proven=False,
            closed_loop_stability_proven=False,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            seed=42,
        ),
    )
    print("215 preprojection uniform lower bound", lower_bound, "; no closed-loop stability claim")


if __name__ == "__main__":
    main()

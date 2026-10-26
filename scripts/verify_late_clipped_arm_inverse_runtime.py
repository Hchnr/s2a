"""212 independent recorded-state direct command replay; no counterfactual physics."""

import numpy as np
from arm_joint_wrench_independent import augment
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence
from late_clipped_arm_inverse_independent import features, inputs, nominal

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.late_clipped_arm_inverse_runtime import translation_command


def main():
    root = REPO / "outputs/late_clipped_arm_inverse_runtime_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/late_clipped_arm_inverse_runtime.py",
        "src/s2a/late_clipped_arm_inverse.py",
        "scripts/late_clipped_arm_inverse_independent.py",
        "plans/v0.0.1/216_late_clipped_direct_inverse_20261026.md",
        "outputs/direct_inverse_feedback_20261026/verified.json",
        "scripts/direct_arm_inverse_independent.py",
        "plans/v0.0.1/211_direct_arm_inverse_runtime_20261026.md",
    ):
        e.read(path)
    fit = REPO / "outputs/direct_arm_inverse_fit_20261026"
    execution = e.json(fit / "execution_result.json")
    assert execution["complete"] and execution["new_fits"] == 1
    proof = e.json(fit / "independent_verified.json")
    assert proof["complete"] and proof["numerically_verified"] and proof["rank"] == 218
    ref = proof["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    params = e.json(ref["path"])
    for path, signature in params["source_sha256"].items():
        e.check(path, signature)
    config = e.json("configs/left_table_forward_20261025.json")
    physical = e.json("outputs/left_table_fit_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    rows = []
    lo, hi, outlo, outhi = (
        np.asarray(params["controller_contract"][key])[:3]
        for key in ("input_min", "input_max", "output_min", "output_max")
    )
    coefficients = np.asarray(params["coefficients"])
    for entry in source["trajectories"]:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            data = augment({key: archive[key].copy() for key in archive.files}, physical)
        data = attach_current_wrench(data, entry, config, e)
        ref = entry["source"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            desired = archive["controller_site_pose"][1:].copy()
            actual = archive["sent_commands"].copy()
        previous = np.vstack([np.zeros((1, 7)), actual[:-1]])
        np.testing.assert_array_equal(data["x_previous_command"], previous)
        np.testing.assert_array_equal(data["x_previous_command_available"], np.arange(len(actual)) > 0)
        independent_input = inputs(data, desired, actual[:, -1])
        baseline = nominal(data, desired, params["controller_contract"])
        residual = features(independent_input, params) @ coefficients
        prediction = np.minimum(hi, np.maximum(lo, baseline + residual))
        # Scalar clipping is intentionally separate from the runtime producer.
        expected = np.array(
            [
                [min(hi[j], previous[i, j] + 0.25, max(lo[j], previous[i, j] - 0.25, value[j])) for j in range(3)]
                for i, value in enumerate(prediction)
            ]
        )
        delta = outlo + (expected - lo) * (outhi - outlo) / (hi - lo)
        produced = translation_command(data, desired, gripper_command=actual[:, -1], parameters=params)
        for key, value in (
            ("nominal_command", baseline),
            ("residual_command", residual),
            ("unconstrained_rate_command", prediction),
            ("command", expected),
            ("physical_delta", delta),
        ):
            np.testing.assert_allclose(produced[key], value, atol=1e-10, rtol=0)
        assert np.max(np.abs(produced["command"] - previous[:, :3])) <= 0.25 + 1e-15
        assert np.all(produced["command"] >= lo) and np.all(produced["command"] <= hi)
        rows.append(
            dict(
                native_index=entry["index"],
                split=entry["split"],
                predicted_action_rmse=float(np.sqrt(np.mean(np.sum((prediction - actual[:, :3]) ** 2, axis=1)))),
                rows=len(actual),
                rate_limited_rows=int(np.sum(produced["rate_limited"])),
                max_command_difference=float(np.max(np.abs(produced["command"] - expected))),
            )
        )
    assert len(rows) == 30 and sum(row["rows"] for row in rows) == 6180
    e.recheck()
    root.mkdir()
    publish(
        root / "independent_verified.json",
        dict(
            complete=True,
            independent_command_verified=True,
            trajectories=rows,
            rows=6180,
            seed=42,
            reference_kind="oracle next recorded SITE pose; recorded current feedback",
            evidence_level="recorded-state command replay only",
            physics_steps=0,
            new_fits=0,
            control_qualified=False,
            parameter_ref=proof["parameter_ref"],
            source_sha256=e.hashes,
        ),
    )
    print("216 late-clipped independent command replay passed:30 trajectories/6180 rows;no control claim")


if __name__ == "__main__":
    main()

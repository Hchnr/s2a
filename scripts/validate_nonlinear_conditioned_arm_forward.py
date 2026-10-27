"""100 heldout next-state and trained/untrained local response gates."""

import json
from pathlib import Path

import numpy as np
from arm_nonlinear_independent import INPUTS, augment, nominal_prediction, prediction
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.nonlinear_conditioned_arm_forward import nominal, predict, with_history
from s2a.paired_response_analysis import axis_response

ROOT = Path("outputs/nonlinear_conditioned_arm_forward_20261025")
CHANNELS = ("position", "rotation", "linear_velocity", "angular_velocity")


def rmse(error):
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1))))


def scores(values, data):
    relative = Rotation.from_rotvec(values[:, 3:6]) * Rotation.from_rotvec(data["y_rotation_increment"]).inv()
    return dict(
        position=rmse(values[:, :3] - data["y_motion"]),
        rotation=rmse(relative.as_rotvec()),
        linear_velocity=rmse(values[:, 6:9] / 0.05 - data["y_twist"][:, :3]),
        angular_velocity=rmse(values[:, 9:12] / 0.05 - data["y_twist"][:, 3:]),
    )


def endpoint(predicted, current_pose, command):
    pose = np.eye(4)
    pose[:3, 3] = current_pose[:3, 3] + predicted[:3]
    pose[:3, :3] = Rotation.from_rotvec(predicted[3:6]).as_matrix() @ current_pose[:3, :3]
    return dict(pose=pose, twist=predicted[6:] / 0.05, command=command.copy())


def matrices(packets, scale):
    axes = [axis_response(packets[0], packets[1 + 2 * a], packets[2 + 2 * a], a, scale[a]) for a in range(3)]
    return {
        k: np.column_stack([a["central"]["per_normalized_input"][k] for a in axes])
        for k in ("position", "linear_velocity", "so3_increment", "angular_velocity")
    }


def main():
    e = Evidence()
    for path in (
        __file__,
        "scripts/arm_nonlinear_independent.py",
        "src/s2a/nonlinear_conditioned_arm_forward.py",
        "src/s2a/paired_response_analysis.py",
    ):
        e.read(path)
    config = e.json("configs/nonlinear_conditioned_arm_forward_20261025.json")
    numeric = e.json(ROOT / "independent_fit_verified.json")
    assert numeric["complete"] and numeric["numerically_verified"]
    for path, signature in numeric["source_sha256"].items():
        e.check(path, signature)
    fit = e.json(ROOT / "fit_result.json")
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    parameters = e.json(ref["path"])
    for key in ("data_closure_ref", "data_result_ref", "paired_closure_ref", "paired_analysis_ref"):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    source = e.json(config["data_result_ref"]["path"])
    held = [p for p in source["trajectories"] if p["split"] == "holdout"]
    assert len(held) == 6
    held_results = []
    for entry in held:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        independent_data = augment(data, parameters)
        data = with_history(data, parameters)
        np.testing.assert_allclose(
            data["x_response_residual"], independent_data["x_response_residual"], rtol=0, atol=1e-10
        )
        fitted = predict(data, parameters)
        independent = prediction(independent_data, parameters)
        np.testing.assert_allclose(fitted, independent, rtol=1e-9, atol=1e-9)
        base = nominal_prediction(data, parameters)
        np.testing.assert_allclose(base, nominal(data, parameters), rtol=0, atol=1e-10)
        constant = np.c_[0.05 * data["x_site_twist"], 0.05 * data["x_site_twist"]]
        values = {
            "fitted": scores(independent, data),
            "nominal": scores(base, data),
            "zero_motion": scores(np.zeros_like(base), data),
            "constant_velocity": scores(constant, data),
        }
        gates = {key: values["fitted"][key] <= values["nominal"][key] for key in CHANNELS}
        path = ROOT / f"holdout{entry['index']:02d}_predictions.npz"
        with path.open("xb") as stream:
            np.savez(stream, fitted=fitted, nominal=base, constant_velocity=constant)
        held_results.append(
            dict(
                index=entry["index"],
                rows=len(base),
                metrics=values,
                gates=gates,
                predictions_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            )
        )
        print("holdout", entry["index"], "passed", sum(gates.values()), "/4", flush=True)
    paired = e.json(config["paired_closure_ref"]["path"])
    assert paired["complete"] and paired["data_qualified"]
    summary_ref = paired["summary_ref"]
    e.check(summary_ref["path"], summary_ref["sha256"])
    summary = e.json(summary_ref["path"])
    analysis = e.json(config["paired_analysis_ref"]["path"])
    assert analysis["complete"] and analysis["verified_anchor_groups"] == 6
    paired_config = e.json("configs/paired_arm_response_20261025.json")
    local_results = []
    for group in range(6):
        original = analysis["groups"][group]
        assert original["anchor_group"] == group
        entry_config = paired_config["profiles"][original["profile_index"]]
        entry = next(
            p for p in source["trajectories"] if p["source"]["sha256"] == entry_config["trajectory_ref"]["sha256"]
        )
        assert entry["split"] == "fit"
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        anchor = original["anchor_cycle"]
        with np.load(ref["path"], allow_pickle=False) as cache:
            raw = {k: cache[k].copy() for k in cache.files}
        full, independent_full = with_history(raw, parameters), augment(raw, parameters)
        np.testing.assert_allclose(
            full["x_response_residual"], independent_full["x_response_residual"], rtol=0, atol=1e-10
        )
        condition = {k: full[k][anchor : anchor + 1].copy() for k in INPUTS}
        independent_condition = {k: independent_full[k][anchor : anchor + 1].copy() for k in INPUTS}
        current_pose = condition["x_site_pose"][0]
        true_packets = []
        fitted_packets = []
        nominal_packets = []
        contract = None
        for offset in range(7):
            index = group * 7 + offset
            record = summary["cases"][index]
            assert record["case"] == index
            ref = record["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            collection = Path("outputs/paired_arm_response_20261025") / f"case{index:02d}" / "collection_result.json"
            e.check(collection, paired["source_sha256"][str(collection.resolve())])
            receipt = e.json(collection)
            assert receipt["arrays_ref"] == ref
            if contract is None:
                contract = receipt["controller_contract"]
            else:
                assert contract == receipt["controller_contract"]
            with np.load(ref["path"], allow_pickle=False) as actual:
                np.testing.assert_allclose(current_pose, actual["controller_site_pose"][anchor], rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    condition["x_site_twist"][0], actual["source_twist"][anchor], rtol=0, atol=1e-9
                )
                command = actual["sent_commands"][-1].copy()
                packet = dict(
                    pose=actual["controller_site_pose"][-1].copy(),
                    twist=actual["source_twist"][-1].copy(),
                    command=command,
                )
                selected = {k: v.copy() for k, v in condition.items()}
                selected["action_command"][0] = command
                selected["action_goal_delta"][0, :3] = actual["controller_goal_pos"][-1] - current_pose[:3, 3]
                selected["action_goal_delta"][0, 3:] = Rotation.from_matrix(
                    actual["controller_goal_ori"][-1] @ current_pose[:3, :3].T
                ).as_rotvec()
            predicted = predict(selected, parameters)
            independent_selected = {k: v.copy() for k, v in selected.items()}
            independent_selected["x_response_residual"] = independent_condition["x_response_residual"].copy()
            independent = prediction(independent_selected, parameters)
            np.testing.assert_allclose(predicted, independent, rtol=1e-9, atol=1e-9)
            base = nominal_prediction(selected, parameters)
            true_packets.append(packet)
            fitted_packets.append(endpoint(independent[0], current_pose, command))
            nominal_packets.append(endpoint(base[0], current_pose, command))
        scale = (np.array(contract["output_max"])[:3] - np.array(contract["output_min"])[:3]) / (
            np.array(contract["input_max"])[:3] - np.array(contract["input_min"])[:3]
        )
        actual = matrices(true_packets, scale)
        model = matrices(fitted_packets, scale)
        baseline = matrices(nominal_packets, scale)
        metrics = {}
        gates = {}
        for key in ("position", "linear_velocity", "so3_increment", "angular_velocity"):
            np.testing.assert_allclose(actual[key], original["central_per_normalized_input"][key], rtol=0, atol=1e-9)
            model_error = float(np.linalg.norm(model[key] - actual[key]))
            nominal_error = float(np.linalg.norm(baseline[key] - actual[key]))
            metrics[key] = dict(
                fitted_frobenius_error=model_error,
                nominal_frobenius_error=nominal_error,
                measured=actual[key].tolist(),
                predicted=model[key].tolist(),
                nominal=baseline[key].tolist(),
            )
            gates[key] = model_error <= nominal_error
        local_results.append(
            dict(
                anchor_group=group,
                used_for_contrast_fit=group in config["training_anchor_groups"],
                metrics=metrics,
                gates=gates,
            )
        )
        print("local response", group, "passed", sum(gates.values()), "/4", flush=True)
    prediction_gate = all(all(x["gates"].values()) for x in held_results)
    gradient_gate = all(all(x["gates"].values()) for x in local_results)
    e.recheck()
    result = dict(
        complete=True,
        numerical_fit_verified=True,
        holdout=held_results,
        local_response=local_results,
        holdout_gate_passed=prediction_gate,
        training_anchor_groups=config["training_anchor_groups"],
        validation_anchor_groups=config["validation_anchor_groups"],
        untrained_local_response_gate_passed=all(
            all(x["gates"].values()) for x in local_results if not x["used_for_contrast_fit"]
        ),
        training_constraints_gate_passed=all(
            all(x["gates"].values()) for x in local_results if x["used_for_contrast_fit"]
        ),
        local_response_gate_passed=gradient_gate,
        offline_forward_eligible=prediction_gate and gradient_gate,
        source_sha256=e.hashes,
        new_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        control_qualification=False,
    )
    with (ROOT / "validation_result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "100 complete;holdout",
        prediction_gate,
        "local response",
        gradient_gate,
        "offline eligible",
        result["offline_forward_eligible"],
    )


if __name__ == "__main__":
    main()

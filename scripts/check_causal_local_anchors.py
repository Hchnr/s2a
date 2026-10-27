"""096 fixed six-anchor validation with shared causal prefix and independent SVD."""

import json
from pathlib import Path

import numpy as np
from arm_history_independent import INPUTS, augment, nominal_prediction, prediction
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation
from validate_history_conditioned_arm_forward import endpoint, matrices

from s2a.causal_local_response import CausalLocalResponse
from s2a.history_conditioned_arm_forward import predict, with_history

ROOT = Path("outputs/causal_local_response_20261025")


def main():
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/causal_local_response.py",
        "scripts/arm_history_independent.py",
        "src/s2a/history_conditioned_arm_forward.py",
        "scripts/validate_history_conditioned_arm_forward.py",
    ):
        e.read(path)
    config = e.json("configs/history_conditioned_arm_forward_20261025.json")
    prior_root = Path("outputs/history_conditioned_arm_forward_20261025")
    closed = e.json(prior_root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(prior_root / "fit_result.json")
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    parameters = e.json(ref["path"])
    for key in ("data_result_ref", "paired_closure_ref", "paired_analysis_ref"):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    source = e.json(config["data_result_ref"]["path"])
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
        # Build only from endpoints available at this anchor, never probe labels.
        adapter = CausalLocalResponse()
        prefix = {k: v[:anchor].copy() for k, v in full.items()}
        independent_prefix = {k: v[:anchor].copy() for k, v in independent_full.items()}
        past_prior = predict(prefix, parameters)
        past_independent_prior = prediction(independent_prefix, parameters)
        np.testing.assert_allclose(past_prior, past_independent_prior, rtol=0, atol=1e-9)
        history = []
        for t in range(anchor):
            action = (full["action_goal_delta"][t, :3] - np.array(parameters["translation_mean"])) / np.array(
                parameters["translation_scale"]
            )
            before, now = full["x_site_pose"][t], full["x_site_pose"][t + 1]
            realized = np.r_[
                now[:3, 3] - before[:3, 3],
                Rotation.from_matrix(now[:3, :3] @ before[:3, :3].T).as_rotvec(),
                0.05 * full["x_site_twist"][t + 1],
            ]
            adapter.observe(action, past_prior[t], realized)
            independent_target = np.r_[
                independent_full["y_motion"][t],
                independent_full["y_rotation_increment"][t],
                0.05 * independent_full["y_twist"][t],
            ]
            np.testing.assert_allclose(realized, independent_target, rtol=0, atol=1e-10)
            residual = independent_target - past_independent_prior[t]
            residual[3:6] = Rotation.from_matrix(
                Rotation.from_rotvec(independent_target[3:6]).as_matrix()
                @ Rotation.from_rotvec(past_independent_prior[t, 3:6]).as_matrix().T
            ).as_rotvec()
            history.append((np.r_[1, action], residual))
        h = history[-20:]
        n = len(h)
        x = np.stack([v[0] for v in h]) / np.sqrt(n)
        y = np.stack([v[1] for v in h]) / np.sqrt(n)
        beta = np.linalg.lstsq(np.vstack([x, np.eye(4)]), np.vstack([y, np.zeros((4, 12))]), rcond=None)[0]
        np.testing.assert_allclose(adapter.coefficients(), beta, rtol=0, atol=1e-10)
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
            action = (selected["action_goal_delta"][0, :3] - np.array(parameters["translation_mean"])) / np.array(
                parameters["translation_scale"]
            )
            predicted = adapter.predict(action, predicted[0])[None, :]
            correction = np.r_[1, action] @ beta
            independent_value = independent[0] + correction
            independent_value[3:6] = Rotation.from_matrix(
                Rotation.from_rotvec(correction[3:6]).as_matrix()
                @ Rotation.from_rotvec(independent[0, 3:6]).as_matrix()
            ).as_rotvec()
            independent = independent_value[None, :]
            np.testing.assert_allclose(predicted, independent, rtol=0, atol=1e-9)
            assert adapter.updates == anchor
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
                online_updates=adapter.updates,
                probe_endpoint_updates=0,
                used_for_contrast_fit=group in config["training_anchor_groups"],
                metrics=metrics,
                gates=gates,
            )
        )
        print("local response", group, "passed", sum(gates.values()), "/4", flush=True)
    e.recheck()
    result = dict(
        complete=True,
        local_response=local_results,
        local_response_gate_passed=all(all(r["gates"].values()) for r in local_results),
        online_updates=sum(r["online_updates"] for r in local_results),
        independent_numerical_solves=6,
        probe_endpoint_updates=0,
        source_sha256=e.hashes,
        new_global_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        control_qualification=False,
    )
    with (ROOT / "local_response.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()

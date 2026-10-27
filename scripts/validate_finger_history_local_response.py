"""183 actual paired arm interventions versus aperture-response slopes."""

import json

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_measured_finger_history import independently_reconstruct
from audit_wam_position import Evidence
from finger_history_independent import INPUTS, feature_matrix
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.finger_contact_history import contact_history
from s2a.finger_history_forward import design
from s2a.joint_wrench_condition_forward import with_history


def main():
    out = REPO / "outputs/finger_history_local_response_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    for source_path in (
        "scripts/finger_history_independent.py",
        "src/s2a/finger_history_forward.py",
        "src/s2a/finger_contact_history.py",
        "scripts/audit_measured_finger_history.py",
    ):
        e.read(source_path)
    e.read("plans/v0.0.1/183_arm_finger_local_response_20261025.md")
    numeric = e.json("outputs/finger_history_fit_20261025/independent_verified.json")
    assert numeric["numerically_verified"]
    ref = numeric["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    coef = np.asarray(e.json(ref["path"])["coefficients"])
    config = e.json("configs/left_table_forward_20261025.json")
    parameters = e.json("outputs/finger_history_equations_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    paired = e.json(config["paired_closure_ref"]["path"])
    assert paired["complete"] and paired["data_qualified"]
    summary = e.json(paired["summary_ref"]["path"])
    registration = e.json("configs/paired_arm_response_20261025.json")
    results = []
    for group in range(6):
        native = registration["profiles"][group // 2]
        entry = next(p for p in source["trajectories"] if p["source"]["sha256"] == native["trajectory_ref"]["sha256"])
        assert entry["split"] == "fit"
        anchor = registration["trials"][group * 7]["anchor_cycle"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = with_history({k: cache[k].copy() for k in cache.files}, parameters)
        full = attach_current_wrench(data, entry, config, e)
        full["x_finger_contact_history"] = contact_history(full)
        np.testing.assert_allclose(
            full["x_finger_contact_history"], independently_reconstruct(full), atol=1e-12, rtol=0
        )
        condition = {k: full[k][anchor : anchor + 1].copy() for k in INPUTS}
        actual, predicted, commands = [], [], []
        for branch in range(7):
            ref = summary["cases"][7 * group + branch]["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as trace:
                np.testing.assert_allclose(trace["gripper_qpos"][-2], condition["x_finger_qpos"][0], atol=1e-12, rtol=0)
                q = trace["gripper_qpos"][-2:].copy()
                actual.append(float((q[1, 0] - q[1, 1]) - (q[0, 0] - q[0, 1])))
                row = {k: v.copy() for k, v in condition.items()}
                command = trace["sent_commands"][-1].copy()
                row["action_command"][0] = command
                current = condition["x_site_pose"][0]
                row["action_goal_delta"][0, :3] = trace["controller_goal_pos"][-1] - current[:3, 3]
                row["action_goal_delta"][0, 3:] = Rotation.from_matrix(
                    trace["controller_goal_ori"][-1] @ current[:3, :3].T
                ).as_rotvec()
            value = design(row, parameters) @ coef
            independent = feature_matrix(row, parameters) @ coef
            np.testing.assert_allclose(value, independent, atol=1e-10, rtol=0)
            predicted.append(float(independent[0]))
            commands.append(command)
        measured, model = [], []
        for axis in range(3):
            minus, plus = 1 + 2 * axis, 2 + 2 * axis
            delta = commands[plus][axis] - commands[minus][axis]
            assert delta > 0
            measured.append((actual[plus] - actual[minus]) / delta)
            model.append((predicted[plus] - predicted[minus]) / delta)
        model_error = float(np.linalg.norm(np.asarray(model) - measured))
        baseline_error = float(np.linalg.norm(measured))
        results.append(
            dict(
                group=group,
                used_for_fit=group in config["training_anchor_groups"],
                measured=measured,
                predicted=model,
                model_error=model_error,
                zero_coupling_error=baseline_error,
                passed=model_error <= baseline_error,
            )
        )
    e.recheck()
    publish(
        out / "result.json",
        dict(
            complete=True,
            results=results,
            passed=sum(r["passed"] for r in results),
            local_qualified=all(r["passed"] for r in results),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            control_qualified=False,
        ),
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()

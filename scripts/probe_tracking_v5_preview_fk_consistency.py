"""Check whether accepted preview Cartesian poses match FK of its own joint states."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse, pose_error
from s2a.native_environment import strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.xml_site_kinematics import XMLSiteKinematics


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_preview_fk_consistency_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_long_trajectory_caps_numeric_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed original fixed-horizon search")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0 and reap0",
    )
    producer_ref = owner["result_ref"]
    producer = read(producer_ref)
    entry = next(v for v in producer["entries"] if v["horizon"] == 8)
    answer = entry["answer"]
    require(answer["witness_found"] is True and answer["all_pose_caps_enforced"] is True, "Accepted all8 model witness")
    reg = read(producer["registration_ref"])
    control = read(reg["control_ref"])
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(
        read(reg["tool_ref"])["source_from_tcp"]
    )
    kin = XMLSiteKinematics(checked(reg["xml_ref"]).read_text(), control["contract"]["controller_site"])
    indices = {node.get("name"): value for node, value in kin.indices.items()}
    qids = np.asarray([indices[f"robot0_joint{k}"][0] for k in range(1, 8)])
    with np.load(checked(reg["states_ref"]), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    require(
        entry["step"] == 368 and entry["target_cursors"] == list(range(8)), "Same actual368 and original0..7 targets"
    )
    np.testing.assert_array_equal(state[1 + qids], entry["current_q"])
    np.testing.assert_allclose(kin.endpoint(state)[0], entry["current_site"], atol=1e-10, rtol=0)
    original = read(reg["producer_ref"])
    branch = next(v for v in original["branches"] if v["branch"] == "causal_error")
    dynamics = read(branch["steps_ref"])[0]["dynamics"]
    targets = np.asarray(read(reg["chunks_ref"])[46]["effective_targets_tcp"])
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            xml_ref=reg["xml_ref"],
            states_ref=reg["states_ref"],
            control_ref=reg["control_ref"],
            tool_ref=reg["tool_ref"],
            source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/xml_site_kinematics.py")],
            seed=42,
            step=368,
            horizon=8,
            finite_difference_step_rad=1e-6,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Internal Cartesian/joint prediction consistency, not actual native trajectory prediction or task qualification",
        ),
    )
    result = dict(
        complete=False, passed=False, registration_ref=registration, physics_steps=0, world_model_calls=0, fresh_cases=0
    )
    try:
        jacobian = []
        for column in range(7):
            plus, minus = state.copy(), state.copy()
            plus[1 + qids[column]] += 1e-6
            minus[1 + qids[column]] -= 1e-6
            jacobian.append(pose_error(kin.endpoint(plus)[0], kin.endpoint(minus)[0]) / 2e-6)
        jacobian = np.asarray(jacobian).T
        jacobian_difference = float(np.max(abs(jacobian - np.asarray(dynamics["jacobian"]))))
        require(jacobian_difference < 1e-5, "Independent actual-point world SITE Jacobian")
        rollout = answer["rollout"]
        rows = []
        for cursor, (q, pose) in enumerate(zip(rollout["joint_position"], rollout["pose"], strict=True)):
            changed = state.copy()
            changed[1 + qids] = q
            fk_tcp = kin.endpoint(changed)[0] @ site_from_tcp
            model_tcp = np.asarray(pose) @ site_from_tcp
            closure_error = pose_error(model_tcp, fk_tcp)
            model_error = pose_error(targets[cursor], model_tcp)
            fk_error = pose_error(targets[cursor], fk_tcp)
            np.testing.assert_allclose(model_error, answer["all_tcp_errors"][cursor], atol=1e-10, rtol=0)
            model_ok = bool(
                np.linalg.norm(model_error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(model_error[3:]) <= 0.05 + 1e-10
            )
            fk_ok = bool(np.linalg.norm(fk_error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(fk_error[3:]) <= 0.05 + 1e-10)
            require(model_ok, "Every declared accepted model endpoint meets caps")
            rows.append(
                dict(
                    cursor=cursor,
                    predicted_joint_position=q,
                    model_tcp=model_tcp,
                    fk_tcp=fk_tcp,
                    internal_closure_error=closure_error,
                    model_target_error=model_error,
                    fk_target_error=fk_error,
                    model_caps_satisfied=model_ok,
                    fk_caps_satisfied=fk_ok,
                )
            )
        result.update(
            complete=True,
            passed=True,
            rows_ref=publish(root / "rows.json", rows),
            actual_point_jacobian_max_difference=jacobian_difference,
            model_caps_satisfied_count=sum(v["model_caps_satisfied"] for v in rows),
            fk_caps_satisfied_count=sum(v["fk_caps_satisfied"] for v in rows),
            maximum_closure_position_m=max(float(np.linalg.norm(v["internal_closure_error"][:3])) for v in rows),
            maximum_closure_rotation_rad=max(float(np.linalg.norm(v["internal_closure_error"][3:])) for v in rows),
            maximum_fk_target_position_m=max(float(np.linalg.norm(v["fk_target_error"][:3])) for v in rows),
            maximum_fk_target_rotation_rad=max(float(np.linalg.norm(v["fk_target_error"][3:])) for v in rows),
            scope="Only internal consistency audit: XML FK of predicted q, not measured future state; no new physical/control qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

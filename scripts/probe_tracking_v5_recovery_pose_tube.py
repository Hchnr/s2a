"""Find actual-state anchored pose-tube transitions; XML-only geometric CPU audit."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.bounded_pose_tube import closest_pose_in_tube
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.full_coupled_pose_inverse import hat, right_inverse_jacobian
from s2a.geometry import inverse, pose_error, transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF
from s2a.xml_site_kinematics import XMLSiteKinematics


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_recovery_pose_tube_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    numeric_ref = reference(REPO / "outputs/tracking_v5_feasibility_first_numeric_20261028/verified.json")
    numeric = read(numeric_ref)
    require(numeric["complete"] is numeric["passed"] is True, "Closed two-phase recovery search")
    resume = read(numeric["baseline_ref"])
    hold = read(resume["hold_ref"])
    held = next(v for v in hold["branches"] if v["branch"] == "held")
    source = read(read(hold["registration_ref"])["producer_ref"])
    source_reg = read(source["registration_ref"])
    worker = read(source_reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    control_ref = reference(folder / "controller.json")
    control = read(control_ref)
    contract = control["contract"]
    xml_ref = reference(folder / "model.xml")
    manifest = read(source_reg["manifest_ref"])
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
    targets = np.asarray(read(source_reg["chunks_ref"])[46]["effective_targets_tcp"])
    with np.load(checked(held["states_ref"]), allow_pickle=False) as archive:
        state = archive["sim_states"][-1].copy()
    sources = [
        reference(Path(__file__)),
        reference(REPO / "src/s2a/bounded_pose_ik.py"),
        reference(REPO / "src/s2a/bounded_pose_tube.py"),
        reference(REPO / "src/s2a/xml_site_kinematics.py"),
    ]
    registration = publish(
        root / "registration.json",
        dict(
            numeric_ref=numeric_ref,
            actual_states_ref=held["states_ref"],
            xml_ref=xml_ref,
            control_ref=control_ref,
            tool_ref=tool_ref,
            chunks_ref=source_reg["chunks_ref"],
            source_refs=sources,
            seed=42,
            branches=["sequential_actual"],
            joint_margins_rad=[0.0, 0.05],
            target_cursors=[8, 9],
            total_tube_queries=4,
            max_tube_iterations=60,
            sampled_path_points_per_range=257,
            position_tolerance_m=0.01,
            rotation_tolerance_rad=0.05,
            max_nfev_per_solve=200,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Static bounded robot FK only; no contacts, dynamic/controller reachability, timing or task qualification",
            independent_check="Every returned pose and actual pose reconstructed with compiled XML scalar-joint FK, separate from MuJoCo; original .01m/.05rad tube/ranges rechecked",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(checked(xml_ref)))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
    require(all(v >= 0 for v in joints), "Actual robot joints")
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0, "Actual controller SITE")
    limits = model.jnt_range[joints]
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    actual = data.qpos[qids].copy()
    np.testing.assert_array_equal(actual, numeric["joint_position"])
    independent = XMLSiteKinematics(checked(xml_ref).read_text(), contract["controller_site"])
    independent_queries = []

    def fk(q):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        offset = site[:3, :3] @ site_from_tcp[:3, 3]
        return site @ site_from_tcp, np.vstack((jp[:, dofs] - hat(offset) @ jr[:, dofs], jr[:, dofs]))

    def xml_pose(q):
        changed = state.copy()
        changed[1 + qids] = q
        independent_queries.append(changed)
        return independent.endpoint(changed)[0] @ site_from_tcp

    result = dict(
        complete=False, passed=False, registration_ref=registration, physics_steps=0, world_model_calls=0, fresh_cases=0
    )
    try:
        current_pose, jac = fk(actual)
        np.testing.assert_allclose(current_pose, xml_pose(actual), atol=1e-10, rtol=0)
        derivative = np.vstack((-jac[:3], -right_inverse_jacobian(pose_error(targets[9], current_pose)[3:]) @ jac[3:]))
        errors = []
        for column in range(7):
            plus, minus = actual.copy(), actual.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            numeric_derivative = (
                pose_error(targets[9], xml_pose(plus)) - pose_error(targets[9], xml_pose(minus))
            ) / 2e-6
            errors.append(float(np.max(abs(numeric_derivative - derivative[:, column]))))
        require(max(errors) < 1e-5, "Independent XML residual finite-difference Jacobian")
        branches = []
        maximum_pose_difference = 0.0
        for margin in (0.0, 0.05):
            lower, upper = limits[:, 0] + margin, limits[:, 1] - margin
            for name, start in (("sequential_actual", actual),):
                q = start.copy()
                rows = []
                for cursor in (8, 9):
                    answer = closest_pose_in_tube(fk, targets[cursor], q, lower, upper, max_nfev=200, max_iterations=60)
                    independent_pose = xml_pose(answer["joints"])
                    difference = float(np.max(abs(independent_pose - answer["pose"])))
                    maximum_pose_difference = max(maximum_pose_difference, difference)
                    np.testing.assert_allclose(independent_pose, answer["pose"], atol=1e-10, rtol=0)
                    error = pose_error(targets[cursor], independent_pose)
                    np.testing.assert_allclose(error, answer["error"], atol=1e-10, rtol=0)
                    in_bounds = bool(
                        np.all(answer["joints"] >= lower - 1e-12) and np.all(answer["joints"] <= upper + 1e-12)
                    )
                    witness = bool(
                        in_bounds
                        and np.linalg.norm(error[:3]) <= 0.01 + 1e-10
                        and np.linalg.norm(error[3:]) <= 0.05 + 1e-10
                    )
                    require(
                        witness == answer["within_tube"],
                        "Independent tube label",
                    )
                    rows.append(
                        dict(
                            cursor=cursor,
                            answer=answer,
                            independent_pose=independent_pose,
                            independent_error=error,
                            actual_start_joint_distance_max_rad=float(np.max(abs(answer["joints"] - actual))),
                            previous_solution_distance_max_rad=float(np.max(abs(answer["joints"] - q))),
                            original_joint_margin_rad=float(
                                np.minimum(answer["joints"] - limits[:, 0], limits[:, 1] - answer["joints"]).min()
                            ),
                        )
                    )
                    q = answer["joints"].copy()
                path_samples = []
                goal = np.asarray(rows[-1]["answer"]["joints"])

                def within(error):
                    return bool(np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10)

                for fraction in np.linspace(0.0, 1.0, 257):
                    configuration = actual + fraction * (goal - actual)
                    pose = xml_pose(configuration)
                    previous_error = pose_error(targets[8], pose)
                    next_error = pose_error(targets[9], pose)
                    path_samples.append(
                        dict(
                            fraction=float(fraction),
                            joints=configuration,
                            previous_error=previous_error,
                            next_error=next_error,
                            within_previous=within(previous_error),
                            within_next=within(next_error),
                            original_joint_margin_rad=float(
                                np.minimum(configuration - limits[:, 0], limits[:, 1] - configuration).min()
                            ),
                        )
                    )
                branches.append(
                    dict(
                        joint_margin_rad=margin,
                        name=name,
                        start=start,
                        rows=rows,
                        sampled_path_ref=publish(root / f"path_margin_{margin:.2f}.json", path_samples),
                        all_sampled_points_in_union=all(v["within_previous"] or v["within_next"] for v in path_samples),
                        minimum_sampled_joint_margin_rad=min(v["original_joint_margin_rad"] for v in path_samples),
                        first_sample_within_next=next((v["fraction"] for v in path_samples if v["within_next"]), None),
                        scope="Only257 sampled joint-segment points; not a continuous path, collision, velocity or fixed-controller proof",
                    )
                )
        require(float(data.time) == state[0], "No physical time advance")
        np.testing.assert_array_equal(data.qvel, state[1 + model.nq :])
        for ref in sources:
            checked(ref)
        result.update(
            complete=True,
            passed=True,
            actual_joint=actual,
            joint_limits=limits,
            branches=branches,
            maximum_independent_pose_difference=maximum_pose_difference,
            maximum_independent_jacobian_difference=max(errors),
            independent_queries_ref=publish(root / "independent_queries.json", independent_queries),
            scope="Static pose existence only; no independent executable action/controller/task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

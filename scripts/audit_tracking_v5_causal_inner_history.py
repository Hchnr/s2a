"""Measure frozen joint prediction errors against independently replayed inner states."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_causal_inner_history_audit_20261028"
    require(not root.exists(), "No overwrite/retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent full-inner replay")
    check_ref = owner["result_ref"]
    require(read(check_ref)["complete"] is read(check_ref)["passed"] is True, "Verified all-inner states")
    producer_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_20261028/result.json")
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    control_ref = reference(folder / "controller.json")
    contract = read(control_ref)["contract"]
    numeric_ref = reference(REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028/result.json")
    initial_joint = np.asarray(read(numeric_ref)["initial_joint"])
    sources = {
        ref["path"]: ref["sha256"]
        for ref in [
            reference(Path(__file__)),
            *[reference(REPO / "src/s2a" / name) for name in ("interior_joint_preview.py", "joint_limit_response.py")],
        ]
    }
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            independent_owner_ref=owner_ref,
            independent_ref=check_ref,
            producer_ref=producer_ref,
            numeric_ref=numeric_ref,
            control_ref=control_ref,
            xml_ref=reference(folder / "model.xml"),
            seed=42,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            rendering_policy="XML-only CPU mj_forward; no renderer/context; OSMesa forcing unset before MuJoCo import",
            envelope="Per-joint running maximum of earlier completed-period absolute free-model residual; first period has no bound",
            scope="All actually executed periods; independent native-state causal-history/plan audit; mj_forward only, no task qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    # XML-only CPU dynamics: no OpenGL context or renderer is constructed.
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0 and all(i >= 0 for i in joints), "Actual robot joints/SITE")
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        branches=[],
    )
    try:
        for branch in producer["branches"]:
            rows = read(branch["steps_ref"])
            require(len(rows) == branch["continuation_steps"] and 0 < len(rows) <= 40, "All actual periods")
            with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
                states = archive["sim_states"]
            running = np.zeros(7)
            running_inner = np.zeros((25, 7))
            records, covered, tested = [], 0, 0
            for offset, row in enumerate(rows):
                step = 368 + offset
                require(row["step"] == step, "Ordered physical periods")
                state = states[step]
                data.time = state[0]
                data.qpos[:] = state[1 : 1 + model.nq]
                data.qvel[:] = state[1 + model.nq :]
                mujoco.mj_forward(model, data)
                mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
                d = row["dynamics"]
                np.testing.assert_allclose(mass, d["mass"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(jacobian, d["jacobian"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qpos[qids], d["robot_joint_position"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(data.qvel[dofs], d["robot_joint_velocity"], atol=1e-12, rtol=0)
                site = np.eye(4)
                site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
                site[:3, 3] = data.site_xpos[sid]
                local = build_joint_preview(
                    mass, jacobian, d["kp"], d["kd"], data.qpos[qids], initial_joint, null_kp=10.0, dt=0.002
                )
                delta = interface.decode(np.asarray(row["command"])[:6])[None]
                actual = np.asarray([v["q"] for v in row["actual_inner"]])
                require(actual.shape == (25, 7), "All25 actual inner positions")
                predictions = {}
                for label, constrained in (("free", False), ("ideal_limit", True)):
                    prediction = rollout_joint_preview(
                        local, data.qvel[dofs], delta, site, limits[:, 0], limits[:, 1], enforce_limits=constrained
                    )
                    q = prediction["inner_joint_position"]
                    require(q.shape == actual.shape, "Same25 predicted endpoints")
                    predictions[label] = dict(
                        inner_joint_position=q,
                        maximum_absolute_residual_by_joint=np.max(abs(actual - q), axis=0),
                        minimum_predicted_margin_rad=float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min()),
                    )
                residual = abs(actual - predictions["free"]["inner_joint_position"])
                previous_bound = running.copy() if offset else None
                if previous_bound is not None:
                    mask = residual <= previous_bound[None] + 1e-12
                    covered += int(mask.sum())
                    tested += int(mask.size)
                profile_before = np.maximum.accumulate(running_inner, axis=0)
                if branch["branch"] == "causal_error":
                    before = row["error_history_before"]
                    require(before["future_guarantee"] is False, "Empirical history is not a future guarantee")
                    require(before["completed_periods"] == offset, "Only prior actual candidate periods")
                    require(before["available"] == (offset > 0), "No prior at first candidate period")
                    require(
                        before["last_completed_step"] == (step - 1 if offset else None), "Causal last-completed index"
                    )
                    np.testing.assert_allclose(before["position_error_rad"], running, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(before["inner_position_error_rad"], profile_before, atol=1e-12, rtol=0)
                    planned = np.asarray(row["refinement"]["planned_commands"])
                    np.testing.assert_array_equal(planned[0], np.asarray(row["command"])[:6])
                    full = rollout_joint_preview(
                        local,
                        data.qvel[dofs],
                        np.asarray([interface.decode(c) for c in planned]),
                        site,
                        limits[:, 0],
                        limits[:, 1],
                        enforce_limits=False,
                    )
                    q = full["inner_joint_position"]
                    tightening = np.tile(running, (len(q), 1))
                    tightening[:25] = profile_before
                    require(
                        np.min(np.minimum(q - limits[:, 0] - 0.05 - tightening, limits[:, 1] - 0.05 - tightening - q))
                        >= -1e-10,
                        "Selected full plan retains causal tightened position constraints",
                    )
                    braking = full["joint_position"] + 0.1 * full["joint_velocity"]
                    require(
                        np.min(np.minimum(braking - limits[:, 0], limits[:, 1] - braking)) >= 0.05 - 1e-10,
                        "Original braking constraint retained",
                    )
                running = np.maximum(running, residual.max(axis=0))
                running_inner = np.maximum(running_inner, residual)
                profile_after = np.maximum.accumulate(running_inner, axis=0)
                if branch["branch"] == "causal_error":
                    np.testing.assert_allclose(
                        row["error_history_after"]["position_error_rad"], running, atol=1e-12, rtol=0
                    )
                    np.testing.assert_allclose(
                        row["error_history_after"]["inner_position_error_rad"], profile_after, atol=1e-12, rtol=0
                    )
                    require(
                        row["error_history_after"]["completed_periods"] == offset + 1,
                        "Exactly one update per completed period",
                    )
                records.append(
                    dict(
                        step=step,
                        route=row["refinement"]["route"],
                        optimizer=row["refinement"]["optimization"],
                        predictions=predictions,
                        actual_inner=actual,
                        minimum_actual_margin_rad=float(np.minimum(actual - limits[:, 0], limits[:, 1] - actual).min()),
                        causal_bound_before_period=previous_bound,
                        causal_bound_after_period=running.copy(),
                        causal_inner_profile_before=profile_before,
                        causal_inner_profile_after=profile_after,
                    )
                )
            if branch["stopped"] is not None:
                np.testing.assert_allclose(
                    branch["stopped"]["history_before"]["position_error_rad"], running, atol=1e-12, rtol=0
                )
                np.testing.assert_allclose(
                    branch["stopped"]["history_before"]["inner_position_error_rad"],
                    np.maximum.accumulate(running_inner, axis=0),
                    atol=1e-12,
                    rtol=0,
                )
            rows_ref = publish(root / branch["branch"] / "periods.json", records)
            result["branches"].append(
                dict(
                    branch=branch["branch"],
                    periods_ref=rows_ref,
                    maximum_absolute_free_residual_by_joint=running,
                    covered_coordinates=covered,
                    evaluated_coordinates=tested,
                    causal_empirical_coverage=None if not tested else covered / tested,
                    coverage_is_not_a_future_guarantee=True,
                )
            )
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        checked(producer_ref)
        checked(check_ref)
        result.update(
            complete=True,
            passed=True,
            scope="Independent causal inner-time error-profile and executed-plan constraint audit on saved native states; does not independently prove stopped-state infeasibility or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

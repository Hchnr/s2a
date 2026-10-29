"""Measure frozen joint prediction errors against independently replayed inner states."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.causal_inner_error import CausalInnerError
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.quality_capped_inner_joint_plan import solve_interior_plan
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_reference_hold_numeric_20261028"
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
    chunks = read(reg["chunks_ref"])
    chunk = chunks[46]
    require(chunk["step"] == 368, "Same already-generated reference chunk")
    manifest = read(reg["manifest_ref"])
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    control_ref = reference(folder / "controller.json")
    control = read(control_ref)
    contract = control["contract"]
    tool = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
    numeric_ref = reference(REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028/result.json")
    initial_joint = np.asarray(read(numeric_ref)["initial_joint"])
    source_rows = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    sources = {
        ref["path"]: ref["sha256"]
        for ref in [
            reference(Path(__file__)),
            *[
                reference(REPO / "src/s2a" / name)
                for name in (
                    "interior_joint_preview.py",
                    "joint_limit_response.py",
                    "causal_joint_error.py",
                    "causal_inner_error.py",
                    "past_inner_error_joint_plan.py",
                    "quality_capped_inner_joint_plan.py",
                    "past_error_joint_plan.py",
                    "velocity_braking_joint_plan.py",
                )
            ],
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
            comparison_policy="Same state/history/command constraints/precision; timed original targets versus held previous generated pose and zero twist for8-period forecast",
            rendering_policy="XML-only CPU mj_forward; no renderer/context; OSMesa forcing unset before MuJoCo import",
            envelope="Per-joint running maximum of earlier completed-period absolute free-model residual; first period has no bound",
            scope="All40 fixed-state numerical prediction comparisons; mj_forward only, no timeline rollout or controller qualification",
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
            require(len(rows) == branch["continuation_steps"] and 0 < len(rows) <= 40, "All actually executed periods")
            with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
                states = archive["sim_states"]
            running = np.zeros(7)
            history = CausalInnerError()
            searches = []
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
                snapshot = history.snapshot()
                if step in (376, 377, 378, 382):
                    cursor = row["cursor"]
                    poses = np.asarray(chunk["effective_targets_tcp"])
                    anchor = np.asarray(chunk["anchor_tcp"])
                    h = min(8, len(poses) - cursor)
                    seed = np.asarray(row["seed"]["preview"]["planned_commands"])
                    seed = np.vstack((seed, np.tile(seed[-1], (h - len(seed), 1))))
                    previous = np.asarray(source_rows[367]["command"] if offset == 0 else rows[offset - 1]["command"])[
                        :6
                    ]
                    twists = np.asarray(
                        [reference_world_twist(poses, k, anchor, 0.05)["twist"] for k in range(cursor, cursor + h)]
                    )
                    args = (
                        local,
                        data.qvel[dofs],
                        seed,
                        previous,
                        interface,
                        site,
                        tool,
                        poses[cursor : cursor + h],
                        twists,
                        limits[:, 0],
                        limits[:, 1],
                    )
                    baseline = solve_interior_plan(*args, past_inner_error_rad=snapshot["inner_position_error_rad"])
                    require(cursor > 0, "Previous generated reference available")
                    held_args = list(args)
                    held_args[7] = np.tile(poses[cursor - 1], (h, 1, 1))
                    held_args[8] = np.zeros((h, 6))
                    candidate = solve_interior_plan(
                        *held_args, past_inner_error_rad=snapshot["inner_position_error_rad"]
                    )
                    if candidate["witness_found"]:
                        require(
                            np.linalg.norm(candidate["first_tcp_error"][:3]) <= 0.01 + 1e-10
                            and np.linalg.norm(candidate["first_tcp_error"][3:]) <= 0.05 + 1e-10,
                            "Absolute first-period tracking accuracy",
                        )
                        q = candidate["rollout"]["inner_joint_position"]
                        error = candidate["requested_position_margin_rad"] - 0.05
                        require(
                            np.min(np.minimum(q - limits[:, 0] - 0.05 - error, limits[:, 1] - 0.05 - error - q))
                            >= -1e-10,
                            "All tightened inner endpoints",
                        )
                    searches.append(
                        dict(
                            step=step,
                            original_cursor=cursor,
                            held_cursor=cursor - 1,
                            held_target_tcp=poses[cursor - 1],
                            held_world_twist=np.zeros(6),
                            history_before=snapshot,
                            timed_quality=baseline,
                            held_quality=candidate,
                        )
                    )
                history.stage(step, predictions["free"]["inner_joint_position"])
                history.complete(step, actual)
                residual = abs(actual - predictions["free"]["inner_joint_position"])
                previous_bound = running.copy() if offset else None
                if previous_bound is not None:
                    mask = residual <= previous_bound[None] + 1e-12
                    covered += int(mask.sum())
                    tested += int(mask.size)
                running = np.maximum(running, residual.max(axis=0))
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
                    )
                )
            rows_ref = publish(root / branch["branch"] / "periods.json", records)
            searches_ref = publish(root / branch["branch"] / "searches.json", searches)
            result["branches"].append(
                dict(
                    branch=branch["branch"],
                    periods_ref=rows_ref,
                    searches_ref=searches_ref,
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
            scope="Same-state timed versus explicit held-previous-reference quality search; changed reference timing is not original timed tracking qualification; no physics",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

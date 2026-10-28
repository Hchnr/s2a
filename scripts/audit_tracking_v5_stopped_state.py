"""Measure frozen joint prediction errors against independently replayed inner states."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.inner_constraint_diagnostic import diagnose
from s2a.interior_joint_preview import build_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_stopped_state_diagnostic_20261028"
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
            *[
                reference(REPO / "src/s2a" / name)
                for name in ("interior_joint_preview.py", "joint_limit_response.py", "inner_constraint_diagnostic.py")
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
        for kind in ("causal_error", "causal_inner"):
            base = REPO / f"outputs/tracking_v5_{kind}_continuation_20261028"
            history_root = (
                "tracking_v5_causal_history_audit_execution_20261028"
                if kind == "causal_error"
                else "tracking_v5_causal_inner_history_audit_execution_20261028"
            )
            history_owner_ref = reference(REPO / "outputs" / history_root / "result.json")
            history_owner = read(history_owner_ref)
            require(history_owner["complete"] is history_owner["passed"] is True, "Closed independent causal history")
            history_ref = history_owner["result_ref"]
            require(read(history_ref)["complete"] is read(history_ref)["passed"] is True, "Independent causal values")
            case_ref = reference(base / "result.json")
            case = read(case_ref)
            branch = next(v for v in case["branches"] if v["branch"] == "causal_error")
            stop = branch["stopped"]
            require(stop is not None and stop["action_dispatched"] is False, "Explicit real stop")
            rows = read(branch["steps_ref"])
            require(stop["step"] == 368 + len(rows), "Actual stopped endpoint index")
            with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
                states = archive["sim_states"]
                require(len(states) == stop["step"] + 1, "Final stopped state saved")
                state = states[-1].copy()
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
            np.testing.assert_allclose(data.qpos[qids], rows[-1]["actual_joint_after"], atol=1e-12, rtol=0)
            local = build_joint_preview(
                mass,
                jacobian,
                rows[-1]["dynamics"]["kp"],
                rows[-1]["dynamics"]["kd"],
                data.qpos[qids],
                initial_joint,
                null_kp=10.0,
                dt=0.002,
            )
            site = np.eye(4)
            site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
            site[:3, 3] = data.site_xpos[sid]
            previous = np.asarray(rows[-1]["command"])[:6]
            before = stop["history_before"]
            error = (
                np.tile(before["position_error_rad"], (25, 1))
                if kind == "causal_error"
                else np.asarray(before["inner_position_error_rad"])
            )
            diagnostics = []
            for horizon in (1, 8):
                seed = np.tile(previous, (horizon, 1))
                value = diagnose(
                    local, data.qvel[dofs], seed, previous, interface, site, limits[:, 0], limits[:, 1], error
                )
                diagnostics.append(dict(horizon=horizon, result=value))
            result["branches"].append(
                dict(
                    kind=kind,
                    producer_ref=case_ref,
                    history_owner_ref=history_owner_ref,
                    history_ref=history_ref,
                    states_ref=branch["states_ref"],
                    stopped_step=stop["step"],
                    joint_position=data.qpos[qids].copy(),
                    joint_velocity=data.qvel[dofs].copy(),
                    past_error_profile=error,
                    diagnostics=diagnostics,
                )
            )
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(
            complete=True,
            passed=True,
            scope="Actual stopped-state frozen-model diagnostic; repeated prior command is an affine expansion seed, not the production seed or an executable plan; no physics or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

"""Search the original H12 trajectory with audited callback-direction backtracking."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_prefix_trust_control import admit_warm, run_case

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.prefix_callback_backtracking_evidence import (
    BACKTRACKING_REF,
    CPU_ENVIRONMENT,
    PLAN,
    REPO,
    SOURCES,
    case_root,
    checked,
    finish,
    load_prerequisites,
    model_registration,
    read,
    reference,
    require,
)
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF


def run_search(
    source, warm, interface, dynamics, *, solver_fn=None, rollout_fn=rollout_native_goal_trajectory, progress=None
):
    """Reuse exact native reset/accounting, explicitly selecting the new solver."""
    if solver_fn is None:
        from s2a.prefix_callback_backtracking_plan import solve_prefix_callback_backtracking_response

        solver_fn = solve_prefix_callback_backtracking_response

    def solve(*args, **kwargs):
        return solver_fn(*args, **kwargs, max_stages=60)

    return run_case(
        "candidate", source, warm, interface, dynamics, solver_fn=solve, rollout_fn=rollout_fn, progress=progress
    )


def main(dry=False):
    case_name = "candidate"
    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "Registered single-thread CPU environment",
    )
    source, source_reg, backtracking, refs = load_prerequisites()
    root = case_root(case_name)
    require(not root.exists(), "No overwrite or automatic retry")
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    warm = admit_warm(source, backtracking["cases"]["prefix"]["entries"][55]["request"]["commands"], interface)
    finish(refs)
    model_reg, model_proof = model_registration(source_reg, refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, case_name=case_name, source_count=len(refs), responses=0)),
            flush=True,
        )
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            case_name=case_name,
            experiment="prefix_callback_backtracking",
            asset_loading_proof=model_proof,
            backtracking_ref=BACKTRACKING_REF,
            plan_ref=reference(PLAN),
            seed=42,
            original_prefix_registration_ref=source["registration_ref"],
            source_refs=[reference(REPO / relative) for relative in SOURCES],
            input_refs=list(refs.values()),
            warm_locator="cases.prefix.entries[55].request.commands",
            endpoint=368,
            horizon=12,
            response_budget=4000,
            iteration_budget=60,
            stage_budget=60,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        case_name=case_name,
        registration_ref=registration,
        asset_loading_proof=model_proof,
        warm=warm,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        retiming_qualified=False,
        task_qualified=False,
        automatic_retry=False,
    )
    started = time.monotonic()
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        require(dynamics.model.opt.timestep == 0.002, "Original2ms XML")
        with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][368].copy()
        np.testing.assert_array_equal(source["current_q"], state[1 + dynamics.qids])
        np.testing.assert_array_equal(source["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
        np.testing.assert_array_equal(source["joint_limits"], dynamics.model.jnt_range[dynamics.joints])
        for name, fixed in source["fixed_gripper_state"].items():
            jid = dynamics.mj.mj_name2id(dynamics.model, dynamics.mj.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0, "Original actual368 finger coordinate")
            np.testing.assert_array_equal(
                fixed,
                [
                    state[1 + dynamics.model.jnt_qposadr[jid]],
                    state[1 + dynamics.model.nq + dynamics.model.jnt_dofadr[jid]],
                ],
            )
        run_search(source, warm, interface, dynamics, progress=result)
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True, passed=True, source_refs_unchanged=list(refs.values()), elapsed_s=time.monotonic() - started
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "verified.json", result)
    print(
        json.dumps(
            dict(complete=True, passed=True, case_name=case_name, witness_found=result["answer"]["witness_found"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.dry)

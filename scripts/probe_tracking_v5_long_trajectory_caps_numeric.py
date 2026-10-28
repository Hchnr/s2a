"""Compare8/12/16-point constrained previews from the same earliest actual state."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse
from s2a.interior_joint_preview import build_joint_preview
from s2a.long_trajectory_capped_joint_plan import solve_interior_plan as all_endpoints
from s2a.native_environment import strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.xml_site_kinematics import XMLSiteKinematics


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_long_trajectory_caps_numeric_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent physical replay")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    producer_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_20261028/result.json")
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    branch = next(v for v in producer["branches"] if v["branch"] == "causal_error")
    rows = read(branch["steps_ref"])
    by_step = {v["step"]: v for v in rows}
    with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    history_ref = reference(REPO / "outputs/tracking_v5_causal_inner_history_audit_20261028/verified.json")
    history = read(history_ref)
    require(history["complete"] is history["passed"] is True, "Independent own-history audit")
    history_branch = next(v for v in history["branches"] if v["branch"] == "causal_error")
    audited = {v["step"]: v for v in read(history_branch["periods_ref"])}
    limits = np.asarray(read(branch["inner_ref"])["joint_limits"])
    folder = Path(read(reg["worker_ref"])["physical_case_dir"])
    control_ref = reference(folder / "controller.json")
    control = read(control_ref)
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    manifest = read(reg["manifest_ref"])
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
    xml_ref = reference(folder / "model.xml")
    kinematics = XMLSiteKinematics(checked(xml_ref).read_text(), contract["controller_site"])
    robot_nodes = {node.get("name"): indices for node, indices in kinematics.indices.items()}
    qids = np.asarray([robot_nodes[f"robot0_joint{k}"][0] for k in range(1, 8)])
    dofs = np.asarray([robot_nodes[f"robot0_joint{k}"][1] for k in range(1, 8)])
    initial = np.asarray(read(reg["braking_numeric_ref"])["initial_joint"])
    original_rows = [json.loads(v) for v in checked(reg["rows_ref"]).read_text().splitlines()]
    chunk = read(reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Previously generated query46")
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    steps = [368]
    sources = [
        reference(Path(__file__)),
        *[
            reference(REPO / "src/s2a" / name)
            for name in (
                "trajectory_capped_joint_plan.py",
                "long_trajectory_capped_joint_plan.py",
                "feasibility_first_joint_plan.py",
                "xml_site_kinematics.py",
                "interior_joint_preview.py",
                "joint_limit_preview_objective.py",
            )
        ],
    ]
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            independent_owner_ref=owner_ref,
            history_ref=history_ref,
            states_ref=branch["states_ref"],
            steps_ref=branch["steps_ref"],
            control_ref=control_ref,
            tool_ref=tool_ref,
            xml_ref=xml_ref,
            chunks_ref=reg["chunks_ref"],
            source_refs=sources,
            steps=steps,
            horizons=[8, 12, 16],
            warm_start="Same original8-point actual-branch plan; repeat its last command to extend12/16; no alternative starts",
            seed=42,
            total_searches=3,
            per_search_response_budget=160,
            total_response_budget=480,
            per_search_iteration_budget=60,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Three fixed horizons at the same actual368 state; current measured robot dynamics, own past history and already generated40-point reference only",
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        entries=[],
    )
    try:
        for step in steps:
            row = by_step[step]
            d = row["dynamics"]
            state = states[step]
            site, twist = kinematics.endpoint(state)
            q, velocity = state[1 + qids], state[1 + kinematics.nq + dofs]
            np.testing.assert_array_equal(q, d["robot_joint_position"])
            np.testing.assert_array_equal(velocity, d["robot_joint_velocity"])
            np.testing.assert_allclose(np.asarray(d["jacobian"]) @ velocity, twist, atol=1e-10, rtol=0)
            profile = np.asarray(audited[step]["causal_inner_profile_before"])
            np.testing.assert_array_equal(profile, row["error_history_before"]["inner_position_error_rad"])
            previous = np.asarray(original_rows[367]["command"] if step == 368 else by_step[step - 1]["command"])
            old_warm = np.asarray(row["refinement"]["planned_commands"])
            require(old_warm.shape == (8, 6), "Original8-point warm plan")
            local = build_joint_preview(d["mass"], d["jacobian"], d["kp"], d["kd"], q, initial, null_kp=10.0, dt=0.002)
            cursor = step - 368
            for horizon in (8, 12, 16):
                seed = np.tile(previous[:6], (horizon, 1))
                warm = np.vstack((old_warm, np.tile(old_warm[-1], (horizon - 8, 1))))
                indices = list(range(cursor, cursor + horizon))
                desired = targets[indices]
                twists = np.asarray([reference_world_twist(targets, k, anchor, 0.05)["twist"] for k in indices])
                answer = all_endpoints(
                    local,
                    velocity,
                    seed,
                    previous[:6],
                    interface,
                    site,
                    site_from_tcp,
                    desired,
                    twists,
                    limits[:, 0],
                    limits[:, 1],
                    initial_guess=warm,
                    past_inner_error_rad=profile,
                )
                if answer["witness_found"]:
                    require(answer["minimum_tightened_inner_slack_rad"] >= -1e-10, "Tightened all-inner joint points")
                    errors = np.asarray(answer["all_tcp_errors"])
                    require(
                        errors.shape == (horizon, 6)
                        and np.all(np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
                        and np.all(np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10),
                        "Every original endpoint accuracy",
                    )
                result["entries"].append(
                    dict(
                        step=step,
                        cursor=cursor,
                        horizon=horizon,
                        target_cursors=indices,
                        current_q=q,
                        current_velocity=velocity,
                        current_site=site,
                        causal_history=profile,
                        previous_command=previous,
                        answer=answer,
                    )
                )
        for ref in sources:
            checked(ref)
        result.update(
            complete=True,
            passed=True,
            scope="Conditional finite-budget model search only; not a new physical trajectory or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

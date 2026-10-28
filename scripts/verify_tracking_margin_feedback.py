"""Independent native restoration of paired earlier-intervention trajectories."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_margin_feedback_verified_20261027"
    root.mkdir(exist_ok=False)
    owner_ref = reference(REPO / "outputs/tracking_margin_feedback_execution_20261027/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True and len(owner["stages"]) == 2, "Complete owner")
    registration = read(owner["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json"))
    old_verified = read(registration["baseline_verified_ref"])
    require(old_verified["complete"] is old_verified["passed"] is True, "Checked baseline")
    old_owner = read(old_verified["owner_ref"])
    for path, digest in read(old_owner["registration_ref"])["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    baseline_stages = [item for item in old_owner["stages"] if item["strategy"] == "baseline"]
    require(len(baseline_stages) == 2, "Exactly two baseline branches")
    stages = {}
    physics = 0
    for item in baseline_stages + owner["stages"]:
        stage, result = read(item["stage_ref"]), read(item["result_ref"])
        exit, release = read(stage["exit_ref"]), read(stage["release_ref"])
        require(stage["passed"] is stage["complete"] is result["passed"] is result["complete"] is True, "Stage")
        require(
            exit["actual_wait"] is release["reaped"] is True and exit["returncode"] == release["returncode"] == 0,
            "Wait/reap",
        )
        require(result["prefix_steps"] == 72 and result["branch_steps"] == 56, "Period counts")
        physics += 128
        ref = reference(Path(item["result_ref"]["path"]).parent / "steps.json")
        rows = read(ref)
        require(len(rows) == 56, "Complete rows")
        stages[item["case_index"], item["strategy"]] = (result, rows, ref)
    require(physics == 512 and registration["physics_budget"] == 256, "New/baseline budgets")
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import current_observation, make_environment, source_pose
    from s2a.native_training import state_sha256
    from s2a.osc import OSCInterface

    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        baseline = stages[index, "baseline"][0]
        require(
            baseline["branch_initial_state_sha256"] == stages[index, "tracking"][0]["branch_initial_state_sha256"],
            "Paired initial state",
        )
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunks = read(refs["chunks.json"])
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        tool = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))["contract"]
        interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        manifest = read(case["manifest_ref"])
        env = None
        metrics = {}
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            qids = np.asarray(env.robots[0]._ref_joint_pos_indexes)
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            for strategy in ("baseline", "tracking"):
                _, rows, rows_ref = stages[index, strategy]
                previous = np.asarray(original[71]["command"])[:6]
                errors, margins = [], []
                for j, row in enumerate(rows):
                    query = 9 + j // 8 if j < 16 else 11
                    cursor = j % 8 if j < 16 else j - 16
                    require((row["query_index"], row["reference_index"]) == (query, cursor), "Reference schedule")
                    command = np.asarray(row["command"])
                    require(
                        np.all(command[:6] >= interface.input_min - 1e-12)
                        and np.all(command[:6] <= interface.input_max + 1e-12),
                        "Box",
                    )
                    require(
                        np.max(abs(command[:6] - previous)) <= 0.25 + 1e-12 and abs(command[-1]) <= 1, "Rate/gripper"
                    )
                    plan = np.asarray(row["refinement"]["planned_commands"])
                    require(len(plan) == min(8 if strategy == "tracking" else 4, 40 - cursor), "Full plan")
                    require(np.max(abs(np.diff(np.vstack((previous, plan)), axis=0))) <= 0.25 + 1e-12, "Plan rate")
                    require(
                        np.all(plan >= interface.input_min - 1e-12) and np.all(plan <= interface.input_max + 1e-12),
                        "Plan box",
                    )
                    np.testing.assert_array_equal(plan[0], command[:6])
                    previous = command[:6]
                    if strategy == "baseline" and j < 16:
                        np.testing.assert_allclose(command, original[72 + j]["command"], atol=1e-8, rtol=0)
                    f = row["refinement"]
                    if f["optimization"] is not None:
                        opt = f["optimization"]
                        caps = np.asarray(opt["first_pose_caps"])
                        seed = np.asarray(f["plan_guard"]["seed_tcp_errors"])[0]
                        chosen = np.asarray(f["plan_guard"]["chosen_tcp_errors"])[0]
                        expected = np.maximum([np.linalg.norm(seed[:3]), np.linalg.norm(seed[3:])], [0.01, 0.05])
                        np.testing.assert_allclose(caps, expected, atol=1e-12, rtol=0)
                        passed = bool(
                            np.linalg.norm(chosen[:3]) <= caps[0] + 1e-10
                            and np.linalg.norm(chosen[3:]) <= caps[1] + 1e-10
                        )
                        if opt["first_pose_caps_enforced"]:
                            require(passed, "Enforced first caps")
                        else:
                            require(opt["selection"] == "interior_witness", "Only primary cap disabled")
                        if "original_first_pose_caps_satisfied" in opt:
                            require(passed == opt["original_first_pose_caps_satisfied"], "Reported old cap outcome")
                        require(opt["response_evaluations"] <= 640 and opt["iterations"] <= 60, "Optimizer budget")
                        np.testing.assert_allclose(
                            opt["objective"], opt["tracking_objective"] + opt["margin_objective"], atol=1e-10, rtol=0
                        )
                    if f["optimization"] is not None:
                        diag = f["optimization"]["interior_diagnostic"]
                        parts = [
                            f["optimization"][k]
                            for k in (
                                "hard_response_evaluations",
                                "adaptive_response_evaluations",
                                "refinement_response_evaluations",
                                "soft_response_evaluations",
                            )
                        ]
                        require(
                            all(0 <= part <= 160 for part in parts)
                            and sum(parts) == f["optimization"]["response_evaluations"],
                            "Separated budgets",
                        )
                        selection = f["optimization"]["selection"]
                        if selection == "interior_witness":
                            require(
                                f["optimization"]["terminal_constraint_enforced"] is True, "Enforced terminal condition"
                            )
                            signed = np.asarray(f["optimization"]["terminal_signed_velocities_rad_s"])
                            np.testing.assert_allclose(
                                signed, diag["terminal_signed_velocities_rad_s"], atol=1e-10, rtol=0
                            )
                            require(np.all(signed <= 1e-10), "Terminal signed velocity")
                            require(
                                diag["witness_found"] and diag["minimum_inner_margin_rad"] >= 0.05 - 1e-10,
                                "Logged full interior",
                            )
                        else:
                            require(not diag["witness_found"], "Explicit hard-stage failure")
                            require(
                                f["optimization"]["terminal_constraint_enforced"] is False,
                                "Fallback must not claim terminal enforcement",
                            )
                        if selection in ("interior_witness", "adaptive_margin_witness"):
                            require(f["guard_seed_model"] == "free", "Witness seed model")
                            minimum = f["optimization"]["minimum_inner_margin_rad"]
                            required = f["optimization"]["joint_margin_rad"]
                            require(minimum >= required - 1e-10, "Retained margin log")
                            require(
                                bool(minimum >= 0.05 - 1e-10) == f["optimization"]["requested_margin_satisfied"],
                                "Original margin classification",
                            )
                            if selection == "adaptive_margin_witness":
                                require(
                                    required > 0 and f["optimization"]["adaptive_diagnostic"]["witness_found"],
                                    "Positive adaptive witness",
                                )
                        else:
                            require(
                                selection == "preventive_fallback" and f["guard_seed_model"] == "ideal_limit",
                                "Soft route explicit",
                            )
                    state = np.asarray(row["sim_state"])
                    require(state_sha256(state) == row["state_sha256"], "State SHA")
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
                    error = pose_error(
                        np.asarray(chunks[query]["effective_targets_tcp"])[cursor],
                        source_pose(current_observation(env)) @ tool,
                    )
                    np.testing.assert_allclose(error, row["tcp_error"], atol=1e-8, rtol=0)
                    for key, tolerance in (("pos", 1e-9), ("ori", 1e-8)):
                        np.testing.assert_allclose(
                            row[f"controller_goal_{key}"], f[f"expected_goal_{key}"], atol=tolerance, rtol=0
                        )
                    q = env.sim.data.qpos[qids]
                    margins.append(np.minimum(q - limits[:, 0], limits[:, 1] - q))
                    errors.append(error)
                e, m = np.asarray(errors), np.asarray(margins)
                crossing = np.flatnonzero(m[:, 6] < 0)
                metrics[strategy] = dict(
                    rows_ref=rows_ref,
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1)))),
                    rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1)))),
                    first_q7_outside_step=None if not len(crossing) else 73 + int(crossing[0]),
                    q7_outside_periods=int(np.sum(m[:, 6] < 0)),
                    minimum_q7_margin_rad=float(m[:, 6].min()),
                    successful_steps=sum(r["success"] for r in rows),
                    grasp_contact_steps=sum(any(o["grasp_contact"] for o in r["objects"].values()) for r in rows),
                    route_counts={
                        route: sum(r["refinement"]["route"] == route for r in rows)
                        for route in sorted(set(r["refinement"]["route"] for r in rows))
                    },
                    fallback_reasons={
                        reason: sum(
                            r["refinement"]["optimization"] is not None
                            and r["refinement"]["optimization"]["interior_diagnostic"]["status"] == reason
                            for r in rows
                        )
                        for reason in sorted(
                            set(
                                r["refinement"]["optimization"]["interior_diagnostic"]["status"]
                                for r in rows
                                if r["refinement"]["optimization"] is not None
                            )
                        )
                    }
                    if strategy == "tracking"
                    else {},
                    primary_old_cap_violation_periods=sum(
                        r["refinement"]["optimization"] is not None
                        and not r["refinement"]["optimization"]["first_pose_caps_enforced"]
                        and not r["refinement"]["optimization"]["original_first_pose_caps_satisfied"]
                        for r in rows
                    ),
                    primary_old_cap_position_excess_m=[
                        max(
                            0.0,
                            float(np.linalg.norm(np.asarray(r["refinement"]["plan_guard"]["chosen_tcp_errors"])[0, :3]))
                            - r["refinement"]["optimization"]["first_pose_caps"][0],
                        )
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                        and not r["refinement"]["optimization"]["first_pose_caps_enforced"]
                    ],
                    primary_old_cap_rotation_excess_rad=[
                        max(
                            0.0,
                            float(np.linalg.norm(np.asarray(r["refinement"]["plan_guard"]["chosen_tcp_errors"])[0, 3:]))
                            - r["refinement"]["optimization"]["first_pose_caps"][1],
                        )
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                        and not r["refinement"]["optimization"]["first_pose_caps_enforced"]
                    ],
                    terminal_enforced_periods=sum(
                        bool(r["refinement"]["optimization"]["terminal_constraint_enforced"])
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                    ),
                    terminal_model_max_signed_velocities_rad_s=[
                        max(r["refinement"]["optimization"]["terminal_signed_velocities_rad_s"], default=0.0)
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                        and r["refinement"]["optimization"]["terminal_constraint_enforced"]
                    ],
                    adaptive_retained_margins_rad=[
                        r["refinement"]["optimization"]["joint_margin_rad"]
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                        and r["refinement"]["optimization"]["selection"] == "adaptive_margin_witness"
                    ],
                    original_margin_not_met_periods=sum(
                        not r["refinement"]["optimization"]["requested_margin_satisfied"]
                        for r in rows
                        if r["refinement"]["optimization"] is not None
                    ),
                    compute_mean_seconds=float(np.mean([r["refinement"]["compute_seconds"] for r in rows])),
                )
        finally:
            if env is not None:
                env.close()
        cases.append(dict(case=case["case"], metrics=metrics))
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    owner_ref=owner_ref,
                    source_ref=reference(Path(__file__)),
                    cases=cases,
                    physics_steps=256,
                    comparison_baseline_physics_steps=256,
                    verification_physics_steps=0,
                    control_qualified=False,
                    scope="earlier paired fixed-reference replay, not live WAM or stable fresh acceptance",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()

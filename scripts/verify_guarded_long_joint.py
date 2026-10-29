"""Independent lifecycle, common-prefix, command and saved-state TCP verification."""

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
    root = REPO / "outputs/guarded_long_joint_verified_20261027"
    require(not root.exists(), "No overwrite")
    owner_ref = reference(REPO / "outputs/guarded_long_joint_execution_20261027/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True and len(owner["stages"]) == 4, "All four stages required")
    registration = read(owner["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json"))
    stages = {}
    physics = 0
    for item in owner["stages"]:
        stage, result = read(item["stage_ref"]), read(item["result_ref"])
        exit_receipt, release = read(stage["exit_ref"]), read(stage["release_ref"])
        require(
            stage["passed"] is stage["complete"] is result["passed"] is result["complete"] is True, "Incomplete stage"
        )
        require(exit_receipt["actual_wait"] is release["reaped"] is True, "Actual wait and reap required")
        require(exit_receipt["returncode"] == release["returncode"] == 0, "Successful exit required")
        child = read(stage["launch_ref"])["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            require(int(fields[19]) != child["start_ticks"] or fields[0] in ("Z", "X"), "Child still live")
        case, horizon = item["case_index"], item["horizon"]
        require((case, horizon) not in stages, "Duplicate stage")
        require(
            result["prefix_steps"] == prepared["cases"][case]["replay_periods"] and result["branch_steps"] == horizon,
            "Physics budget differs",
        )
        physics += result["prefix_steps"] + horizon
        rows_ref = reference(Path(item["result_ref"]["path"]).parent / "steps.json")
        rows = read(rows_ref)
        require(len(rows) == horizon, "Incomplete physical rows")
        stages[case, horizon] = (result, rows, rows_ref)
    require(physics == registration["physics_budget"] == 306, "Fixed306 budget required")
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import current_observation, make_environment, source_pose
    from s2a.native_training import state_sha256
    from s2a.osc import OSCInterface

    tracking = read(reference(REPO / "outputs/preview_chunk_tracking_restored_20261027/endpoints.json"))
    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        short, a, _ = stages[index, 1]
        long, b, rows_ref = stages[index, 40]
        left, right = json.loads(json.dumps(a[0])), json.loads(json.dumps(b[0]))
        left["refinement"].pop("compute_seconds")
        right["refinement"].pop("compute_seconds")
        require(left == right, "Candidate first period differs except non-control wall timing")
        baseline_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{index:02d}_h40/result.json")
        baseline = read(baseline_ref)
        require(
            short["branch_initial_state_sha256"]
            == long["branch_initial_state_sha256"]
            == baseline["branch_initial_state_sha256"],
            "Paired branch initial states differ",
        )
        np.testing.assert_array_equal(long["branch_initial_motor"], baseline["branch_initial_motor"])
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets = np.asarray(chunk["effective_targets_tcp"])
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        manifest = read(case["manifest_ref"])
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        previous = np.asarray(original[case["replay_periods"] - 1]["command"])[:6]
        contract = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))["contract"]
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        errors = []
        env = None
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            for j, row in enumerate(b):
                command = np.asarray(row["command"])
                require(np.isfinite(command).all() and abs(command[-1]) <= 1, "Finite gripper command")
                require(
                    np.all(command[:6] >= interface.input_min) and np.all(command[:6] <= interface.input_max), "Arm box"
                )
                require(np.max(abs(command[:6] - previous)) <= 0.25 + 1e-12, "Arm rate")
                require(np.any(interface.decode(command[:6])[3:] != 0), "Native reset representative required")
                previous = command[:6]
                plan = np.asarray(row["refinement"]["planned_commands"])
                np.testing.assert_array_equal(command[:6], plan[0])
                require(len(plan) == min(8, 40 - j), "Full available preview horizon differs")
                require(
                    np.all(plan >= interface.input_min - 1e-12) and np.all(plan <= interface.input_max + 1e-12),
                    "Plan box",
                )
                prior = (
                    np.asarray(original[case["replay_periods"] - 1]["command"])[:6]
                    if j == 0
                    else np.asarray(b[j - 1]["command"])[:6]
                )
                require(np.max(abs(np.diff(np.vstack((prior, plan)), axis=0))) <= 0.25 + 1e-12, "Plan rate")
                pose_guard = row["refinement"]["plan_guard"]
                if pose_guard is not None:
                    for start, end in ((0, 3), (3, 6)):
                        require(
                            np.all(
                                np.linalg.norm(np.asarray(pose_guard["chosen_tcp_errors"])[:1, start:end], axis=1)
                                <= np.linalg.norm(np.asarray(pose_guard["seed_tcp_errors"])[:1, start:end], axis=1)
                                + 1e-10
                            ),
                            "Returned plan violates exact model pose guard",
                        )
                    optimization = row["refinement"]["optimization"]
                    require(
                        optimization["response_evaluations"] <= 160 and optimization["iterations"] <= 60,
                        "Optimizer budget",
                    )
                require(row["refinement"]["innovation_applied"] is False, "Unregistered innovation injection")
                state = np.asarray(row["sim_state"])
                require(state_sha256(state) == row["state_sha256"], "State hash")
                env.sim.set_state_from_flattened(state)
                obs = current_observation(env)
                error = pose_error(targets[j], source_pose(obs) @ h)
                np.testing.assert_allclose(error, row["tcp_error"], atol=1e-8, rtol=0)
                np.testing.assert_allclose(
                    row["controller_goal_pos"], row["refinement"]["expected_goal_pos"], atol=1e-9, rtol=0
                )
                np.testing.assert_allclose(
                    row["controller_goal_ori"], row["refinement"]["expected_goal_ori"], atol=1e-8, rtol=0
                )
                errors.append(error)
        finally:
            if env is not None:
                env.close()
        baseline_errors = np.asarray([r["error"] for r in tracking[index * 2 + 1]["rows"]])
        metrics = {}
        for name, e in (("baseline", baseline_errors), ("candidate", np.asarray(errors))):
            metrics[name] = dict(
                position_rmse_m=float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1)))),
                rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1)))),
            )
        cases.append(
            dict(
                case=case["case"],
                rows_ref=rows_ref,
                baseline_ref=baseline_ref,
                metrics=metrics,
                successful_steps=sum(r["success"] for r in b),
                refined_periods=sum(r["refinement"]["route"] == "preview_refined" for r in b),
                response_evaluations=sum(r["refinement"]["evaluations"] for r in b),
                retained_seed_periods=sum(r["refinement"]["route"] == "preview_retained_seed" for r in b),
                unsupported_latch_periods=sum(r["refinement"]["route"] == "unsupported_future_latch_seed" for r in b),
                compute_mean_seconds=float(np.mean([r["refinement"]["compute_seconds"] for r in b])),
                compute_max_seconds=max(r["refinement"]["compute_seconds"] for r in b),
            )
        )
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    owner_ref=owner_ref,
                    cases=cases,
                    physics_steps=physics,
                    source_ref=reference(__file__),
                    control_qualified=False,
                    fresh_authorized=False,
                    scope="two fixed local physical branches; not18 correct-reference qualification or600-step task acceptance",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()

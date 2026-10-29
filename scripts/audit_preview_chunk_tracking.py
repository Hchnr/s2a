"""Restore saved endpoints for TCP tracking; never advance simulation or query WAM."""

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
    output = REPO / "outputs/preview_chunk_tracking_restored_20261027"
    output.mkdir(exist_ok=False)
    verified_ref = reference(REPO / "outputs/preview_chunk_diagnostic_verified_20261027/verified.json")
    verified = read(verified_ref)
    require(verified["passed"] is True, "Paired branch verification required")
    for ref in verified["input_refs"]:
        checked(ref)
    prepared = read(verified["prepared_ref"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import aperture, current_observation, make_environment, source_pose
    from s2a.native_training import state_sha256

    summaries, all_rows = [], []
    for index, case in enumerate(prepared["cases"]):
        refs = {Path(ref["path"]).name: ref for ref in case["original_refs"]}
        chunks = read(refs["chunks.json"])
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        targets = np.asarray(chunks[case["event"]["query_index"]]["effective_targets_tcp"])
        manifest = read(case["manifest_ref"])
        tool = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        controller_ref = reference(Path(refs["chunks.json"]["path"]).parent / "controller.json")
        contract = read(controller_ref)["contract"]
        target_name = {8: "akita_black_bowl_1", 9: "wine_bottle_1"}[manifest["record"]["task_id"]]
        env = None
        try:
            env, _ = make_environment(manifest["runtime_config"], output / f"case{index:02d}", manifest["record"], 0)
            for horizon in (8, 40):
                rows_ref = reference(
                    REPO / f"outputs/preview_chunk_branches_20261027/case{index:02d}_h{horizon}/steps.json"
                )
                rows = read(rows_ref)
                previous = np.asarray(original[case["replay_periods"] - 1]["command"])[:6]
                errors, distances, records = [], [], []
                for j, row in enumerate(rows):
                    state = np.asarray(row["sim_state"])
                    require(state_sha256(state) == row["state_sha256"], "Saved endpoint hash mismatch")
                    env.sim.set_state_from_flattened(state)
                    np.testing.assert_array_equal(env.sim.get_state().flatten(), state)
                    obs = current_observation(env)
                    refreshed = env.sim.get_state().flatten()
                    np.testing.assert_allclose(refreshed, state, atol=1e-14, rtol=0)
                    refresh_difference = float(np.max(np.abs(refreshed - state)))
                    tcp = source_pose(obs) @ tool
                    error = pose_error(targets[j], tcp)
                    np.testing.assert_allclose(aperture(obs), row["opening"], atol=1e-12, rtol=0)
                    if j < 8:
                        np.testing.assert_allclose(
                            error, original[case["replay_periods"] + j]["tcp_error"], atol=1e-8, rtol=0
                        )
                    command = np.asarray(row["command"])
                    require(np.isfinite(command).all() and abs(command[6]) <= 1, "Invalid gripper command")
                    require(np.all(command[:6] >= np.asarray(contract["input_min"]) - 1e-12), "Lower box")
                    require(np.all(command[:6] <= np.asarray(contract["input_max"]) + 1e-12), "Upper box")
                    require(np.max(np.abs(command[:6] - previous)) <= 0.25 + 1e-12, "Command rate")
                    previous = command[:6]
                    center = np.asarray(row["objects"][target_name]["position"])
                    distance = [float(np.linalg.norm(p[:3, 3] - center)) for p in (targets[j], tcp)]
                    errors.append(error)
                    distances.append(distance)
                    records.append(
                        dict(
                            index=j,
                            tcp=tcp.tolist(),
                            refresh_state_max_abs=refresh_difference,
                            error=error.tolist(),
                            target_and_actual_center_distance_m=distance,
                        )
                    )
                errors = np.asarray(errors)
                close = next((j for j, row in enumerate(rows) if row["opening_target"] < 0.2), None)
                summaries.append(
                    dict(
                        case=case["case"],
                        horizon=horizon,
                        rows_ref=rows_ref,
                        controller_ref=controller_ref,
                        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                        max_position_error_m=float(np.max(np.linalg.norm(errors[:, :3], axis=1))),
                        first_close_index=close,
                        first_close_error=None if close is None else errors[close].tolist(),
                        first_close_center_distances_m=None if close is None else distances[close],
                    )
                )
                all_rows.append(dict(case_index=index, horizon=horizon, rows=records))
        finally:
            if env is not None:
                env.close()
        print(json.dumps(dict(completed_case=index)), flush=True)
    rows_ref = publish(output / "endpoints.json", all_rows)
    print(
        json.dumps(
            publish(
                output / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    verified_ref=verified_ref,
                    source_ref=reference(Path(__file__)),
                    cases=summaries,
                    endpoints_ref=rows_ref,
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                    scope="saved-state endpoint kinematics and executed command box/rate; no full dynamic or KKT replay",
                    center_distance_is_not_grasp_feasibility=True,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()

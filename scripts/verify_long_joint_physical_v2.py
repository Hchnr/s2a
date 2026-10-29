"""Restore native endpoints and independently measure long-plan tracking."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def metrics(errors):
    errors = np.asarray(errors)
    return dict(
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
    )


def main():
    strict_native_environment()
    root = REPO / "outputs/long_joint_physical_v2_verified_20261027"
    root.mkdir(exist_ok=False)
    owner_ref = reference(REPO / "outputs/long_joint_physical_recovery_20261027/completed.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Completed owner required")
    registration = read(reference(REPO / "outputs/long_joint_physical_v2_20261027/registration.json"))
    recovery_registration = read(owner["registration_ref"])
    for path, digest in recovery_registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    first_receipts = read(reference(REPO / "outputs/long_joint_physical_v2_20261027/receipts.json"))
    receipts = first_receipts + [dict(case=0, horizon=16, wait_returncode=None, reaped=None)] + owner["receipts"]
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json"))
    plan = read(reference(REPO / "outputs/long_joint_preview_probe_20261027/verified.json"))
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import current_observation, make_environment, source_pose

    summaries, total = [], 0
    for receipt, scheduled in zip(receipts, registration["schedule"], strict=True):
        if not (receipt["case"] == 0 and receipt["horizon"] == 16):
            require(receipt["wait_returncode"] == 0 and receipt["reaped"] is True, "Child wait receipt")
        index, horizon = scheduled["case"], scheduled["horizon"]
        require(receipt["case"] == index and receipt["horizon"] == horizon, "Schedule")
        folder = REPO / f"outputs/long_joint_physical_v2_branches_20261027/case{index:02d}_h{horizon}"
        result_ref, rows_ref = reference(folder / "result.json"), reference(folder / "steps.json")
        result, rows = read(result_ref), read(rows_ref)
        require(result["complete"] is result["passed"] is result["native_goals_verified"] is True, "Branch")
        case = prepared["cases"][index]
        require(
            result["prefix_steps"] == case["replay_periods"] and result["branch_steps"] == len(rows) == horizon,
            "Counts",
        )
        total += result["prefix_steps"] + horizon
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets = np.asarray(chunk["effective_targets_tcp"])
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        contract = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        tool = np.linalg.inv(contract["source_from_site"]) @ h
        planned = next(p for p in plan["cases"][index]["results"] if p["horizon"] == horizon)
        manifest = read(case["manifest_ref"])
        env = None
        actual_errors, prediction_errors = [], []
        try:
            env, _ = make_environment(
                manifest["runtime_config"], root / f"case{index:02d}_h{horizon}", manifest["record"], 0
            )
            for j, row in enumerate(rows):
                state = np.asarray(row["sim_state"])
                require(hashlib.sha256(state.tobytes()).hexdigest() == row["state_sha256"], "State digest")
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
                tcp = source_pose(current_observation(env)) @ h
                np.testing.assert_allclose(tcp, row["tcp"], atol=1e-12, rtol=0)
                error = pose_error(targets[j], tcp)
                prediction = pose_error(np.asarray(planned["rollout"]["pose"])[j] @ tool, tcp)
                np.testing.assert_allclose(error, row["tcp_error"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(prediction, row["prediction_error"], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(np.asarray(row["command"])[:6], planned["planned_commands"][j])
                actual_errors.append(error)
                prediction_errors.append(prediction)
        finally:
            if env is not None:
                env.close()
        original_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{index:02d}_h40/steps.json")
        original = read(original_ref)[:horizon]
        summaries.append(
            dict(
                case=index,
                horizon=horizon,
                result_ref=result_ref,
                rows_ref=rows_ref,
                tracking=metrics(actual_errors),
                prediction_error=metrics(prediction_errors),
                original_same_duration=metrics([r["tcp_error"] for r in original]),
                original_ref=original_ref,
                task_success_steps=sum(bool(r["success"]) for r in rows),
            )
        )
    require(total == registration["physics_budget"] == 272, "Physics budget")
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    owner_ref=owner_ref,
                    verifier_ref=reference(Path(__file__)),
                    cases=summaries,
                    physical_steps=total,
                    new_verification_physics_steps=0,
                    control_qualified=False,
                    lifecycle_complete=False,
                    lifecycle_gap="raw H16 child result complete but actual wait return code receipt missing after exclusive-create owner failure",
                    total_physics_including_failed_first_attempt=361,
                    scope="independent restored native endpoints of saved open-loop plans; no stable task qualification",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()

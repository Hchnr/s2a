"""Fixed-state numerical braking probe; no timeline rollout or model request."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF
from s2a.tracking_terminal_joint_plan import solve_interior_plan as original
from s2a.velocity_braking_joint_plan import solve_interior_plan as reentry


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028"
    require(not root.exists(), "No overwrite/retry")
    production = REPO / "outputs/tracking_preview_wam_development_v5_20261028"
    proof_ref = reference(REPO / "outputs/tracking_v5_case01_endpoints_20261028/verified.json")
    proof = read(proof_ref)
    require(
        proof["complete"] is proof["passed"] is True and proof["outside_before_periods"] == 197,
        "Actual boundary evidence",
    )
    worker_ref = reference(production / "development/cases/task0_init0_s42_terminal058/worker_result.json")
    worker = read(worker_ref)
    folder = Path(worker["physical_case_dir"])
    manifest_ref, control_ref, rows_ref, chunks_ref = [
        reference(folder / name) for name in ("manifest.json", "controller.json", "steps.jsonl", "chunks.json")
    ]
    manifest, control = read(manifest_ref), read(control_ref)
    rows = [json.loads(x) for x in checked(rows_ref).read_text().splitlines()]
    chunks = read(chunks_ref)
    sources = dict(read(reference(production / "prepared.json"))["source_sha256"])
    for path in (Path(__file__), REPO / "src/s2a/velocity_braking_joint_plan.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            endpoint_proof_ref=proof_ref,
            worker_ref=worker_ref,
            manifest_ref=manifest_ref,
            control_ref=control_ref,
            rows_ref=rows_ref,
            chunks_ref=chunks_ref,
            arrays_ref=worker["arrays_ref"],
            steps=[368, 375, 382],
            seed=42,
            original_future_joint_margin_rad=0.05,
            braking_time_s=0.1,
            braking_time_policy="Exactly two20Hz control periods, fixed before execution",
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Native environment initialization only; no timeline action/step; fixed-state local model search",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    env = None
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        initial = env.robots[0].controller.initial_joint.copy()
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        contract = control["contract"]
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
        tool = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
        entries = []
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            for step in (368, 375, 382):
                row = rows[step]
                d = row["coupled_dynamics"]
                site = archive["controller_site_pose"][step]
                model = build_joint_preview(
                    d["mass"],
                    d["jacobian"],
                    d["kp"],
                    d["kd"],
                    d["robot_joint_position"],
                    initial,
                    null_kp=10.0,
                    dt=0.002,
                )
                chunk = chunks[row["chunk_index"]]
                poses, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
                index = row["pose_preview_seed_inverse"]["reference_current_index"]
                h = min(8, len(poses) - index)
                seed = np.asarray(row["pose_preview_seed_inverse"]["preview"]["planned_commands"])
                seed = np.vstack((seed, np.tile(seed[-1], (h - len(seed), 1))))
                previous = np.asarray(rows[step - 1]["command"])[:6]
                twists = np.asarray(
                    [reference_world_twist(poses, k, anchor, 0.05)["twist"] for k in range(index, index + h)]
                )
                args = (
                    model,
                    d["robot_joint_velocity"],
                    seed,
                    previous,
                    interface,
                    site,
                    tool,
                    poses[index : index + h],
                    twists,
                    limits[:, 0],
                    limits[:, 1],
                )
                baseline = original(*args)
                if step == 382:
                    require(baseline["witness_found"], "Reproduce original interior witness")
                    np.testing.assert_allclose(
                        baseline["planned_commands"], row["tracking_refinement"]["planned_commands"], atol=1e-7, rtol=0
                    )
                candidate = reentry(*args)
                if candidate["witness_found"]:
                    planned = candidate["planned_commands"]
                    rollout = rollout_joint_preview(
                        model,
                        d["robot_joint_velocity"],
                        np.asarray([interface.decode(c) for c in planned]),
                        site,
                        limits[:, 0],
                        limits[:, 1],
                        enforce_limits=False,
                    )
                    q = rollout["inner_joint_position"]
                    require(
                        np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)) >= 0.05 - 1e-10,
                        "All future inner endpoints retain .05",
                    )
                    require(
                        np.max(abs(np.diff(np.vstack((previous, planned)), axis=0))) <= 0.25 + 1e-12,
                        "Original rate bounds",
                    )
                entries.append(dict(step=step, baseline=baseline, candidate=candidate))
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(
            complete=True,
            passed=True,
            entries=entries,
            initial_joint=initial,
            tool_ref=tool_ref,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Local numerical search only; no physical qualification or task success",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()

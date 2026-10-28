"""Compare positive joint buffers at fixed actual states; no temporal physics."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_buffer_tradeoff_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    producer_ref = reference(REPO / "outputs/committed_generated_chunk_20261027/result.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Closed physical producer")
    reg = read(producer["registration_ref"])
    manifest = read(reg["manifest_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    control = read(reference(folder / "controller.json"))["contract"]
    rows = read(producer["branches"][1]["steps_ref"])
    original = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            selected_cursors=[0, 8, 20, 30],
            margins_rad=[0.05, 0.01],
            source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/configurable_tracking_terminal_plan.py")],
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Fixed measured-state local numerical tradeoff; no temporal rollout or changed production policy",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.configurable_tracking_terminal_plan import solve_interior_plan
    from s2a.generated_reference_twist import reference_world_twist
    from s2a.geometry import inverse
    from s2a.interior_joint_preview import build_joint_preview
    from s2a.libero_simulation import make_environment
    from s2a.osc import OSCInterface

    env = None
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        controller = env.robots[0].controller
        interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        template = read(reference(Path(manifest["runtime_config"]["tool_contract"])))
        mapping = np.asarray(read(reference(folder / "controller.json"))["source_from_site"])
        site_from_tcp = inverse(mapping) @ np.asarray(template["source_from_tcp"])
        chunk = read(reg["chunks_ref"])[17]
        targets = np.asarray(chunk["effective_targets_tcp"])
        tcp_anchor = np.asarray(chunk["anchor_tcp"])
        results = []
        with np.load(checked(producer["branches"][1]["states_ref"]), allow_pickle=False) as archive:
            for cursor in (0, 8, 20, 30):
                row = rows[cursor]
                d = row["dynamics"]
                env.sim.set_state_from_flattened(archive["sim_states"][136 + cursor])
                env.sim.forward()
                site = np.eye(4)
                site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
                site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
                previous = np.asarray(original[135]["command"] if cursor == 0 else rows[cursor - 1]["command"])[:6]
                model = build_joint_preview(
                    d["mass"],
                    d["jacobian"],
                    d["kp"],
                    d["kd"],
                    d["robot_joint_position"],
                    controller.initial_joint,
                    null_kp=10.0,
                    dt=0.002,
                )
                seed = np.asarray(row["seed"]["preview"]["planned_commands"])
                horizon = min(8, 40 - cursor)
                seed = np.vstack((seed, np.tile(seed[-1], (horizon - len(seed), 1))))
                desired = targets[cursor : cursor + horizon]
                twists = np.asarray(
                    [
                        reference_world_twist(targets, j, tcp_anchor, 0.05)["twist"]
                        for j in range(cursor, cursor + horizon)
                    ]
                )
                entries = []
                for margin in (0.05, 0.01):
                    solved = solve_interior_plan(
                        model,
                        d["robot_joint_velocity"],
                        seed,
                        previous,
                        interface,
                        site,
                        site_from_tcp,
                        desired,
                        twists,
                        limits[:, 0],
                        limits[:, 1],
                        joint_margin_rad=margin,
                    )
                    require(solved["witness_found"], "Actual model witness at registered buffer")
                    if margin == 0.05:
                        expected = row["refinement"]["optimization"]
                        np.testing.assert_allclose(solved["objective"], expected["objective"], atol=1e-7, rtol=0)
                        np.testing.assert_allclose(
                            solved["planned_commands"], row["refinement"]["planned_commands"], atol=1e-7, rtol=0
                        )
                    require(
                        solved["minimum_inner_margin_rad"] >= margin - 1e-10,
                        "Original physical ranges plus positive buffer",
                    )
                    entries.append(
                        {
                            k: solved[k]
                            for k in (
                                "joint_margin_rad",
                                "objective",
                                "first_tcp_error",
                                "minimum_inner_margin_rad",
                                "terminal_signed_velocities_rad_s",
                                "planned_commands",
                                "response_evaluations",
                            )
                        }
                    )
                results.append(dict(cursor=cursor, variants=entries))
        print(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    registration_ref=registration,
                    results=results,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Fixed actual states, same targets/objective/action and physical ranges; smaller positive buffer model comparison only",
                ),
            ),
            flush=True,
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

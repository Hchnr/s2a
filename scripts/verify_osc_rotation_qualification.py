"""Independently audit saved rotation qualification physics and optional plots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import load_config, sha256


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metrics(desired, actual, config):
    position = np.linalg.norm(desired[:, :3, 3] - actual[:, :3, 3], axis=1)
    angles = np.linalg.norm(
        Rotation.from_matrix(desired[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)).as_rotvec(), axis=1
    )
    return (
        dict(
            steps=len(position),
            canonical_tcp_position_rmse_mm=float(1000 * np.sqrt(np.mean(position**2))),
            so3_rmse_rad=float(np.sqrt(np.mean(angles**2))),
            max_position_error_mm=float(1000 * max(position)),
            max_rotation_error_rad=float(max(angles)),
            within_both_tolerances_fraction=float(
                np.mean((position <= config["position_tolerance_m"]) & (angles <= config["rotation_tolerance_rad"]))
            ),
        ),
        position,
        angles,
    )


def check_metrics(saved, recalculated):
    assert set(saved) == set(recalculated)
    for key, value in recalculated.items():
        np.testing.assert_allclose(saved[key], value, atol=1e-10, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/osc_rotation_qualification_20261021_v2.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("audit requires explicit empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    config, root = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs")
    summary_path = root / "qualification_summary.json"
    summary = json.loads(summary_path.read_text())
    matrix_path = root / "matrix_manifest.json"
    matrix = json.loads(matrix_path.read_text())
    expected = {
        (r, m["name"], method) for r in config["records"] for m in config["motions"] for method in config["methods"]
    }
    assert summary["complete"] and summary["completed"] == summary["expected"] == len(expected)
    assert {(x["record"], x["motion"], x["method"]) for x in summary["results"]} == expected
    assert len(summary["results"]) == len(expected)
    inputs = dict(matrix["files_sha256"])
    for path in [
        summary_path,
        matrix_path,
        root / "frozen_config.json",
        root / "runtime_cpu_process_observation.json",
        Path(__file__),
    ]:
        inputs[str(path.resolve())] = sha256(path)
    assert json.loads((root / "frozen_config.json").read_text()) == config
    fixture = json.loads(Path(config["fixture"]).read_text())
    records = {r["name"]: r for r in fixture["records"]}
    checked, plotting, paired = [], {}, {}
    for saved in summary["results"]:
        result_path = Path(saved["path"])
        assert result_path == root / saved["record"] / saved["motion"] / saved["method"] / "result.json"
        assert sha256(result_path) == saved["sha256"]
        result = json.loads(result_path.read_text())
        assert result["error"] is None and result["steps"] == result["expected_steps"] == config["steps"]
        inputs[str(result_path)] = sha256(result_path)
        for path, digest in result["files_sha256"].items():
            assert sha256(path) == digest
            inputs[path] = digest
        case = result_path.parent
        case_manifest = json.loads((case / "manifest.json").read_text())
        assert case_manifest["cuda_visible_devices"] == "" and case_manifest["seed"] == config["seed"]
        initial = json.loads((case / "initial.json").read_text())
        physics_path = case / "initial_physics.npz"
        inputs[str(physics_path)] = sha256(physics_path)
        with np.load(physics_path) as data:
            state_sha = hashlib.sha256(data["state"].tobytes()).hexdigest()
        assert (
            state_sha == initial["restored_simulation_state_sha256"] == records[saved["record"]]["initial_state_sha256"]
        )
        rows = [json.loads(line) for line in (case / "steps.jsonl").read_text().splitlines()]
        assert len(rows) == config["steps"]
        assert [row["step"] for row in rows] == list(range(config["steps"]))
        assert all("objects" not in row and "success" not in row for row in rows)
        command = np.asarray([row["command"] for row in rows])
        contract = initial["controller"]
        low, high, outlow, outhigh = [
            np.asarray(contract[k]) for k in ("input_min", "input_max", "output_min", "output_max")
        ]
        assert np.all(command[:, :6] >= low - 1e-12) and np.all(command[:, :6] <= high + 1e-12)
        assert np.all(command[:, 6] == 0)
        dt = np.asarray([row["actual_dt"] for row in rows])
        np.testing.assert_allclose(dt, 1 / config["control_fps"], atol=1e-9, rtol=0)
        changes = np.diff(np.vstack([np.zeros(6), command[:, :6]]), axis=0)
        assert np.all(np.abs(changes) <= config["max_command_rate"] / config["control_fps"] + 1e-12)
        physical = (command[:, :6] - (high + low) / 2) * (outhigh - outlow) / (high - low) + (outhigh + outlow) / 2
        np.testing.assert_allclose(physical, np.asarray([row["physical_delta"] for row in rows]), atol=1e-12, rtol=0)
        expected_motor = np.asarray(initial["expected_gripper_control"])
        np.testing.assert_allclose(
            np.asarray([row["gripper_control"] for row in rows]),
            np.broadcast_to(expected_motor, (len(rows), len(expected_motor))),
            atol=1e-12,
            rtol=0,
        )
        h, q = np.asarray(initial["tool_source_from_tcp"]), np.asarray(initial["source_from_control_rotation"])
        desired = np.asarray([row["desired_tcp"] for row in rows])
        actual = np.asarray([row["after_tcp"] for row in rows])
        after = np.asarray([row["after_source"] for row in rows])
        measured = np.asarray([row["measured_source"] for row in rows])
        np.testing.assert_allclose(after @ h, actual, atol=1e-12, rtol=0)
        np.testing.assert_allclose(measured[1:], after[:-1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            measured[:, :3, :3] @ q, np.asarray([row["measured_control_rotation"] for row in rows]), atol=1e-6, rtol=0
        )
        for i, row in enumerate(rows):
            goal_before, goal_after = (
                np.asarray(row["controller_goal_rotation_before"]),
                np.asarray(row["controller_goal_rotation_after"]),
            )
            if np.all(physical[i, 3:] == 0):
                np.testing.assert_array_equal(goal_before, goal_after)
            else:
                expected_goal = Rotation.from_rotvec(physical[i, 3:]).as_matrix() @ np.asarray(
                    row["measured_control_rotation"]
                )
                np.testing.assert_allclose(goal_after, expected_goal, atol=1e-6, rtol=0)
        all_metrics, position, angles = metrics(desired, actual, config)
        check_metrics(result["overall"], all_metrics)
        check_metrics(result["last40"], metrics(desired[-40:], actual[-40:], config)[0])
        if result["motion"] == "large_rotation_step":
            for start, end in ((20, 80), (80, 140), (140, 180)):
                check_metrics(
                    result["large_rotation_hold_tail20"][str(start)],
                    metrics(desired[end - 20 : end], actual[end - 20 : end], config)[0],
                )
        reference_sha = hashlib.sha256(desired.tobytes()).hexdigest()
        key = (result["record"], result["motion"])
        comparison = dict(
            reference_sha256=reference_sha,
            initial=initial["source_pose"],
            controller_goal=initial["controller_goal_rotation"],
            gripper_target=initial["held_gripper_motor_target"],
            q=q.tolist(),
        )
        if key in paired:
            assert comparison == paired[key]
        else:
            paired[key] = comparison
        checked.append(
            dict(
                record=result["record"],
                motion=result["motion"],
                method=result["method"],
                state_sha256=state_sha,
                reference_sha256=reference_sha,
                overall=all_metrics,
                last40=result["last40"],
                large_rotation_hold_tail20=result.get("large_rotation_hold_tail20"),
                zero_rotation_steps=int(sum(np.all(physical[:, 3:] == 0, axis=1))),
                gripper_motor_target_verified=True,
            )
        )
        plotting[(result["record"], result["motion"], result["method"])] = (np.cumsum(dt), position, angles)
    for path, digest in inputs.items():
        assert sha256(path) == digest
    outputs = {}
    if args.plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory = root / "diagnostics"
        directory.mkdir(exist_ok=False)
        colors = {"baseline_rotation": "#777777", "response_rotation": "#2171b5", "bounded_rotation": "#d95f0e"}
        for record in config["records"]:
            figure, axes = plt.subplots(len(config["motions"]), 2, figsize=(11, 10), sharex=True)
            for index, motion in enumerate(config["motions"]):
                for method in config["methods"]:
                    t, position, angles = plotting[(record, motion["name"], method)]
                    axes[index, 0].plot(
                        t, position * 1000, label=method.replace("_rotation", ""), color=colors[method], linewidth=1.3
                    )
                    axes[index, 1].plot(t, angles, color=colors[method], linewidth=1.3)
                axes[index, 0].axhline(
                    config["position_tolerance_m"] * 1000, color="black", linestyle="--", linewidth=0.7
                )
                axes[index, 1].axhline(config["rotation_tolerance_rad"], color="black", linestyle="--", linewidth=0.7)
                axes[index, 0].set_ylabel(f"{motion['name']}\nTCP error [mm]")
                axes[index, 1].set_ylabel("SO(3) error [rad]")
                for axis in axes[index]:
                    axis.grid(alpha=0.2)
                    if "step_segments" in motion:
                        for start, _ in motion["step_segments"][1:]:
                            axis.axvline((start + 1) / config["control_fps"], color="black", alpha=0.2, linewidth=0.7)
            axes[0, 0].legend(fontsize=8)
            for axis in axes[-1]:
                axis.set_xlabel("Physical endpoint time [s]")
            figure.suptitle(
                f"{record}: independent MuJoCo tracking; seed {args.seed}\nSame absolute TCP references / gains / gripper motor targets; task success not scored",
                fontsize=11,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            for extension in ("png", "pdf"):
                path = directory / f"{record}_tracking.{extension}"
                figure.savefig(path, dpi=170)
                outputs[str(path.resolve())] = sha256(path)
            plt.close(figure)
    report = dict(
        seed=args.seed,
        complete=True,
        verified_cases=len(checked),
        verified_physical_steps=len(checked) * config["steps"],
        input_file_count=len(inputs),
        case_results=checked,
        paired_initial_controller_gripper_and_world_target_streams_exact=True,
        task_success_not_scored=True,
        limits="Independent physics tracking only, without WAM generation or task/contact inputs. Large steps are single-period infeasible. This does not establish WAM task success or universal controller dominance.",
        files_sha256=inputs,
        outputs_sha256=outputs,
    )
    path = root / "independent_verification.json"
    write_new(path, report)
    print(
        json.dumps(
            dict(
                path=str(path),
                sha256=sha256(path),
                cases=len(checked),
                physical_steps=report["verified_physical_steps"],
            )
        )
    )


if __name__ == "__main__":
    main()

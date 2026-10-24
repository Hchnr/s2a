"""Independent array comparison of the finite native hold intervention."""

import argparse
import json
from pathlib import Path

import numpy as np
from run_native_pose_hold_diagnostic import publish, read, read_prepared, reference, require


def arrays(case):
    ref = case["arrays"]
    require(reference(ref["path"]) == ref, "trajectory digest changed")
    with np.load(ref["path"], allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_goals(goals, data, n, method):
    require(len(goals) == n + 40, "actual policy goal count differs")
    terminal = data["controller_site_pose"][n]
    for index, goal in enumerate(goals):
        overridden = method == "latched_pose" and index >= n
        require(goal["step_index"] == index and goal["overridden"] is overridden, "goal phase differs")
        require(np.array_equal(goal["action"], data["sent_commands"][index, :6]), "goal command differs")
        if overridden:
            for key, expected in (("set_pos", terminal[:3, 3]), ("set_ori", terminal[:3, :3])):
                require(np.array_equal(goal[key], expected), "latch not measured native terminal SITE")
            require(
                np.array_equal(goal["actual_goal_pos"], terminal[:3, 3])
                and np.array_equal(goal["actual_goal_ori"], terminal[:3, :3]),
                "actual OSC goal not latched",
            )
        else:
            require(goal["set_pos"] is goal["set_ori"] is None, "native/baseline absolute intervention")


def verify(args):
    ref = dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256)
    prepared = read_prepared(ref)
    root = Path(prepared["root"])
    worker = root / "orchestration/worker"
    exit_record = json.loads((worker / "exit.json").read_text())
    release = json.loads((worker / "ownership/release.json").read_text())
    launch = json.loads((worker / "launch.json").read_text())
    require(
        exit_record["actual_wait"]
        and exit_record["returncode"] == 0
        and release["reaped"]
        and release["returncode"] == 0
        and not release["signal_sent"]
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        "actual worker normal wait/reap/CPU environment missing",
    )
    collected = json.loads((root / "collection/result.json").read_text())
    require(collected["passed"] and collected["complete"] and collected["prepared_ref"] == ref, "worker incomplete")
    require(
        collected["progress"] == dict(environment_initializations=8, physics_steps=1648, current_canvas_images=40),
        "full budget missing",
    )
    expected = [(method, r["name"]) for method in ("baseline", "latched_pose") for r in prepared["records"]]
    require([(r["method"], r["name"]) for r in collected["cases"]] == expected, "full eight-case roster missing")
    original_specs = {row["name"]: row for row in prepared["static_prepared"]["cases"]}
    summaries = []
    for row in collected["cases"]:
        case = read(row["case"])
        goals = read(row["goals"])
        original = read(original_specs[row["name"]]["case_result_ref"])
        data, gold = arrays(case), arrays(original)
        n = original["native_steps"]
        require(case["record"] == original["record"] and case["physics_steps"] == n + 40, "source changed")
        schedule = np.zeros((n + 40, 7))
        schedule[:n] = gold["native_commands"]
        schedule[n:, -1] = gold["native_commands"][-1, -1]
        require(np.array_equal(data["sent_commands"], schedule), "native/hold/gripper command changed")
        require(np.allclose(np.diff(data["sim_time"]), 0.05, atol=1e-9, rtol=0), "20Hz changed")
        expected_tcp = data["source_pose"] @ np.asarray(
            prepared["static_prepared"]["native_prepared"]["tool_source_from_tcp"]
        )
        require(np.allclose(data["tcp_pose"], expected_tcp, atol=1e-10, rtol=0), "measured TCP differs")
        inverse = np.tile(np.eye(4), (len(expected_tcp) - 1, 1, 1))
        inverse[:, :3, :3] = np.swapaxes(expected_tcp[:-1, :3, :3], -1, -2)
        inverse[:, :3, 3] = -np.einsum("nij,nj->ni", inverse[:, :3, :3], expected_tcp[:-1, :3, 3])
        delta = inverse @ expected_tcp[1:]
        require(np.allclose(data["delta_pose"], delta, atol=1e-10, rtol=0), "actual local transition differs")
        expected_action = np.zeros((n + 41, 128))
        expected_action[:-1, :3] = delta[:, :3, 3]
        expected_action[:-1, 3:6] = delta[:, :3, 0]
        expected_action[:-1, 6:9] = delta[:, :3, 1]
        expected_action[:-1, 9] = data["opening"][1:]
        require(np.allclose(data["action128"], expected_action, atol=1e-10, rtol=0), "transition labels differ")
        mask = np.zeros_like(expected_action, dtype=bool)
        mask[:-1, :10] = True
        require(np.array_equal(data["action_mask"], mask), "terminal action mask differs")
        if row["method"] == "baseline":
            require(data.keys() == gold.keys(), "baseline array roster changed")
            for key in gold:
                require(
                    data[key].dtype == gold[key].dtype
                    and data[key].shape == gold[key].shape
                    and data[key].tobytes() == gold[key].tobytes(),
                    f"baseline full exact regression: {key}",
                )
        else:
            keys = [k for k in gold if k.startswith("controller_")]
            keys += ["sim_states", "source_pose", "opening", "native_success", "sim_time", "gripper_qpos"]
            for key in keys:
                require(np.array_equal(data[key][: n + 1], gold[key][: n + 1]), f"native prefix changed: {key}")
        for actual_anchor, gold_anchor in zip(case["anchors"], original["anchors"], strict=True):
            require(actual_anchor["frame_index"] == gold_anchor["frame_index"], "render schedule changed")
            require(actual_anchor["canvas_rgb_sha256"] == gold_anchor["canvas_rgb_sha256"], "prefix image changed")
        verify_goals(goals, data, n, row["method"])
        success = data["native_success"][-40:]
        require(case["native_quality_passed"] is bool(success.all()), "quality summary differs")
        summaries.append(
            dict(
                method=row["method"],
                name=row["name"],
                stable=bool(success.all()),
                successful_hold_endpoints=int(success.sum()),
                complete_hold_success=success.tolist(),
                terminal_tcp_displacement_mm=(
                    (data["tcp_pose"][-1, :3, 3] - data["tcp_pose"][n, :3, 3]) * 1000
                ).tolist(),
                terminal_opening=[float(data["opening"][n]), float(data["opening"][-1])],
            )
        )
    read_prepared(ref)
    review = root / "review"
    review.mkdir(exist_ok=False)
    publish(
        review / "verified.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=ref,
            cases=summaries,
            baseline_exact=True,
            native_prefix_exact=True,
            training_authorized=False,
            fresh_authorized=False,
            new_model_calls=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()

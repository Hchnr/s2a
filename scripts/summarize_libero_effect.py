"""Verify delay-free WAM execution and privileged component interventions."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.collision_bounds import separation_bound
from s2a.prediction_audit import audit_action_response


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def segment_distance(points, center):
    starts, ends = points[:-1], points[1:]
    direction = ends - starts
    squared = np.sum(direction**2, axis=1)
    ratio = np.sum((center - starts) * direction, axis=1) / np.maximum(squared, 1e-30)
    nearest = starts + np.clip(ratio, 0, 1)[:, None] * direction
    return float(np.min(np.linalg.norm(nearest - center, axis=1)))


def integrate(anchor, actions):
    """Independent homogeneous composition; no call to the running decoder."""
    result, current = [], np.asarray(anchor).copy()
    for action in actions:
        first, second = action[3:6].copy(), action[6:9].copy()
        first /= np.linalg.norm(first)
        second -= np.dot(first, second) * first
        second /= np.linalg.norm(second)
        delta = np.eye(4)
        delta[:3, :3] = np.column_stack([first, second, np.cross(first, second)])
        delta[:3, 3] = action[:3]
        current = current @ delta
        result.append(current.copy())
    return np.asarray(result)


def verify_quasistatic(config, root, component_root):
    matrix = json.loads((root / "matrix_quasistatic.json").read_text())
    expected = {
        (source, trial)
        for source in ("oracle_response", "wam_affine", "wam_position_oracle_rotation_opening")
        for trial in config["trials"]
    }
    require(matrix["complete"] and len(matrix["cases"]) == len(expected), "quasistatic matrix incomplete")
    require({(r["source"], r["trial"]) for r in matrix["cases"]} == expected, "quasistatic cases differ")
    results = []
    for item in matrix["cases"]:
        require(item["exit_code"] == 0, "quasistatic process failed")
        dest = root / "quasistatic" / item["source"] / f"trial{item['trial']}"
        manifest = json.loads((dest / "manifest.json").read_text())
        for path, digest in manifest["inputs_sha256"].items():
            require(sha256(path) == digest, "frozen reference input changed")
        name = "scripts/run_libero_endpoint_diagnostic.py"
        executed = subprocess.check_output(["git", "show", f"{manifest['code']['commit']}:{name}"])
        require(
            hashlib.sha256(executed).hexdigest() == manifest["code"]["files_sha256"][name], "quasistatic source differs"
        )
        source_root = component_root if item["source"] == "wam_position_oracle_rotation_opening" else root
        source = source_root / "sync" / item["source"] / f"trial{item['trial']}" / "steps.jsonl"
        original = [json.loads(line) for line in source.read_text().splitlines()][: config["oracle_steps"]]
        references = json.loads((dest / "references.json").read_text())
        rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
        result = json.loads((dest / "result.json").read_text())
        require(result["error"] is None and len(rows) == result["physical_steps"], "quasistatic budget failed")
        require(
            len(references) == len(original) == result["references"] == config["oracle_steps"],
            "reference budget failed",
        )
        grouped, previous = defaultdict(list), np.zeros(6)
        for step, row in enumerate(rows):
            index = row["reference_index"]
            require(row["step"] == step and 0 <= index < len(references), "quasistatic sequence failed")
            close(row["actual_dt"], 1 / config["control_fps"])
            close(row["simulation_time"] - rows[0]["simulation_time"], step / config["control_fps"])
            close(references[index]["pose"], original[index]["desired_source"])
            close(references[index]["opening"], original[index]["opening_target"])
            close(row["desired_source"], references[index]["pose"])
            close(row["opening_target"], references[index]["opening"])
            if step:
                close(row["measured_source"], rows[step - 1]["after_source"])
            command = np.asarray(row["command"])
            require(np.max(abs(command)) <= 1 + 1e-9, "quasistatic command bound failed")
            require(
                np.max(abs(command[:6] - previous)) <= config["max_command_rate"] / config["control_fps"] + 1e-9,
                "quasistatic command rate failed",
            )
            previous = command[:6]
            target, actual = np.asarray(row["desired_source"]), np.asarray(row["after_source"])
            error = np.r_[
                target[:3, 3] - actual[:3, 3], Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            ]
            close(row["error"], error)
            p = config["endpoint_diagnostic"]
            reached = (
                np.linalg.norm(error[:3]) <= p["position_tolerance_m"]
                and np.linalg.norm(error[3:]) <= p["rotation_tolerance_rad"]
            )
            require(row["reached"] == reached, "endpoint reach criterion differs")
            grouped[index].append(row)
        require(set(grouped) == set(range(len(references))), "endpoint skipped")
        endpoints = []
        for group in grouped.values():
            require([r["substep"] for r in group] == list(range(len(group))), "substep sequence differs")
            require(not any(r["reached"] for r in group[:-1]), "reference held after convergence")
            require(
                group[-1]["reached"] or len(group) == config["endpoint_diagnostic"]["max_substeps"],
                "early reference termination",
            )
            endpoints.append(group[-1])
        e = np.asarray([r["error"] for r in endpoints])
        close(result["endpoint_position_rmse_m"], np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1))))
        close(result["endpoint_rotation_rmse_rad"], np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1))))
        require(result["reached_endpoints"] == sum(r["reached"] for r in endpoints), "reached count differs")
        require(
            result["success"] == any(r["success"] for r in rows) and result["terminal_success"] == rows[-1]["success"],
            "quasistatic task result differs",
        )
        result["directory"] = str(dest)
        result["evidence"] = manifest["evidence"]
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--component-config", default="configs/libero_wam_components_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", action="append", choices=["controls", "wam", "components"])
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    cc = json.loads(Path(args.component_config).read_text())
    require(args.seed == c["seed"] == cc["seed"], "seed differs")
    require(not args.case or args.stage, "case-level verification must be explicitly partial")
    out = Path(c["output_dir"])
    fixture = json.loads(Path(c["fixture"]).read_text())
    bridge = json.loads(Path(c["published_bridge"]).read_text())["trajectories"]
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    hi, dt, horizon = np.linalg.inv(h), 1 / c["control_fps"], c["execution_horizon"]
    bounds = json.loads((out / "contact_bounds/bounds.json").read_text())
    workspace = json.loads((out / "workspace_bound/bound.json").read_text())
    gripper_xml = []
    for record in fixture["episodes"]:
        xml = ET.fromstring(record["model_xml"])
        subtree = next(n for n in xml.iter("body") if n.get("name") == "gripper0_right_gripper")
        gripper_xml.append(hashlib.sha256(ET.tostring(subtree)).hexdigest())
    require(len(set(gripper_xml)) == 1, "geometry bound cannot be shared across different grippers")
    specification = yaml.safe_load((Path(c["checkpoint_run"]) / "config.yaml").read_text())
    specification = specification["model"]["config"]["action128_spec"]
    spec = specification["transforms"][specification["registry"]["domains"].index(c["action_domain"])]
    inputs, revisions, artifacts, contracts, results = {}, {}, {}, [], []
    contact_support_checks = 0
    core = [
        "src/s2a/libero_effect.py",
        "src/s2a/action_reference.py",
        "src/s2a/geometry.py",
        "src/s2a/libero_simulation.py",
        "src/s2a/gripper.py",
        "src/s2a/osc.py",
        "src/s2a/cameras.py",
    ]
    for config, stage, expected_cases in (
        (c, "controls", [r for r in c["effect_cases"] if r["source"] == "oracle"]),
        (c, "wam", [r for r in c["effect_cases"] if r["source"] == "wam"]),
        (cc, "components", cc["component_cases"]),
    ):
        if args.stage and stage not in args.stage:
            continue
        if args.case:
            expected_cases = [r for r in expected_cases if r["name"] in args.case]
            if not expected_cases:
                continue
        root = Path(config["output_dir"])
        matrix_path = root / f"matrix_{stage}.json"
        if matrix_path.is_file():
            matrix = json.loads(matrix_path.read_text())
            require(matrix["complete"], "matrix failed")
        elif args.case:
            log_name = "components_matrix.log" if stage == "components" else f"{stage}_matrix.log"
            ledger = [json.loads(line) for line in (root / log_name).read_text().splitlines() if line.startswith("{")]
            matrix = dict(cases=[r for r in ledger if "exit_code" in r])
        else:
            raise ValueError("full matrix has not finished")
        if args.case:
            matrix["cases"] = [r for r in matrix["cases"] if r["case"]["name"] in args.case]
        expected = {(case["name"], trial) for case in expected_cases for trial in config["trials"]}
        require({(r["case"]["name"], r["trial"]) for r in matrix["cases"]} == expected, "case matrix differs")
        require(len(matrix["cases"]) == len(expected), "duplicate cases")
        for item in matrix["cases"]:
            require(item["exit_code"] == 0, "simulation process failed")
            dest = Path(item["directory"])
            manifest = json.loads((dest / "manifest.json").read_text())
            for name, digest in manifest["inputs_sha256"].items():
                if name not in inputs:
                    inputs[name] = sha256(name)
                require(inputs[name] == digest, f"input changed: {name}")
            for name in core:
                key = manifest["code"]["commit"], name
                if key not in revisions:
                    revisions[key] = hashlib.sha256(
                        subprocess.check_output(["git", "show", f"{key[0]}:{name}"])
                    ).hexdigest()
                require(revisions[key] == manifest["code"]["files_sha256"][name], f"executed source differs: {key}")
            require(manifest["case"] == item["case"], "case differs from manifest")
            trial, case = item["trial"], item["case"]
            require(
                manifest["config"] == {**config, "server_url": config["server_urls"][trial]}, "configuration differs"
            )
            reference = bridge[fixture["episodes"][trial]["demo"]]
            oracle_source, oracle_opening = np.asarray(reference["source_poses"]), np.asarray(reference["openings"])
            budget = (
                config["oracle_steps"]
                if stage == "controls"
                else config["component_steps"]
                if stage == "components"
                else config["max_steps"]
            )
            rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
            chunks = json.loads((dest / "chunks.json").read_text())
            result = json.loads((dest / "result.json").read_text())
            require(result["error"] is None and len(rows) == result["steps"] == budget, "physics budget incomplete")
            require(len(chunks) == result["chunks"] == (budget + horizon - 1) // horizon, "prediction count differs")
            expected_targets, expected_openings = [], []
            used_rotation_angles, used_raw_openings = [], []
            for number, chunk in enumerate(chunks):
                step = number * horizon
                require(chunk["sequence"] == number and chunk["step"] == step, "chunk sequence differs")
                require(
                    chunk["simulation_time_before"] == chunk["simulation_time_after"],
                    "physics advanced during inference",
                )
                require(
                    chunk["state_sha256_before"] == chunk["state_sha256_after"],
                    "physical state changed during inference",
                )
                close(chunk["simulation_time_before"], rows[step]["simulation_time"])
                close(chunk["anchor_tcp"], np.asarray(rows[step]["measured_source"]) @ h)
                actions = np.asarray(chunk["actions"])
                require(actions.shape == (config["action_chunk_size"], 128), "model horizon differs")
                raw_targets = integrate(chunk["anchor_tcp"], actions[:horizon])
                previous_tcp = np.concatenate([np.asarray(chunk["anchor_tcp"])[None], raw_targets[:-1]])
                relative_rotation = previous_tcp[:, :3, :3].swapaxes(-1, -2) @ raw_targets[:, :3, :3]
                used_rotation_angles.extend(Rotation.from_matrix(relative_rotation).magnitude())
                used_raw_openings.extend(actions[:horizon, 9])
                opening = np.clip(actions[:horizon, 9], 0, 1)
                if stage != "controls":
                    path = dest / f"response_{number:04d}.json"
                    response = json.loads(path.read_text())
                    require(response["status"] == "success", "WAM request failed")
                    content = response["outputs"][0]["content"]
                    checkpoint = Path(c["checkpoint_run"]) / "checkpoints" / c["checkpoint_iteration"] / "model"
                    require(
                        Path(content["action128"]["checkpoint_path"]).resolve() == checkpoint.resolve(),
                        "checkpoint differs",
                    )
                    audit = audit_action_response(content, spec, domain=c["action_domain"], fps=c["control_fps"])
                    require(audit["denormalization_max_abs"] < 1e-6, "physical action scaling differs")
                    contracts.append(audit)
                    close(content["action"], actions)
                    request = json.loads((dest / f"request_{number:04d}.json").read_text())
                    require(request["model_mode"] == "wam" and request["seed"] == c["seed"], "model mode/seed differs")
                    for request_key, config_key in (
                        ("num_steps", "sampling_steps"),
                        ("guidance", "guidance"),
                        ("shift", "shift"),
                        ("action_chunk_size", "action_chunk_size"),
                        ("fps", "control_fps"),
                        ("prompt", "task_text"),
                    ):
                        require(
                            request[request_key] == config[config_key], f"inference parameter differs: {request_key}"
                        )
                    pixels = np.asarray(Image.open(dest / f"input_{number:04d}.png").convert("RGB"))
                    require(
                        hashlib.sha256(pixels.tobytes()).hexdigest() == chunk["observation_sha256"], "camera changed"
                    )
                targets = raw_targets @ hi
                indices = np.minimum(step + np.arange(horizon), len(oracle_source) - 1)
                if stage == "controls":
                    close(targets, oracle_source[indices])
                    close(opening, oracle_opening[indices])
                if stage == "components":
                    close(chunk["raw_targets_tcp"], raw_targets)
                    close(chunk["raw_openings"], opening)
                    for component in case["oracle_components"]:
                        if component == "position":
                            targets[:, :3, 3] = oracle_source[indices, :3, 3]
                        elif component == "rotation":
                            targets[:, :3, :3] = oracle_source[indices, :3, :3]
                        elif component == "opening":
                            opening = oracle_opening[indices]
                close(chunk["targets_tcp"], targets @ h)
                expected_targets.extend(targets)
                expected_openings.extend(opening)
            previous, errors = np.zeros(6), []
            for step, row in enumerate(rows):
                require(
                    row["step"] == step
                    and row["chunk_sequence"] == step // horizon
                    and row["target_index"] == step % horizon,
                    "reference not executed at prescribed tick",
                )
                close(row["actual_dt"], dt)
                close(row["simulation_elapsed_s"], step * dt)
                close(row["desired_source"], expected_targets[step])
                close(row["opening_target"], expected_openings[step])
                close(row["after_source"], row["physical_after_source"])
                if step:
                    close(row["measured_source"], rows[step - 1]["after_source"])
                command = np.asarray(row["command"])
                require(np.isfinite(command).all() and np.max(abs(command)) <= 1 + 1e-9, "command bound failed")
                require(
                    np.max(abs(command[:6] - previous)) <= config["max_command_rate"] * dt + 1e-9, "command rate failed"
                )
                previous = command[:6]
                target, actual = np.asarray(row["desired_source"]), np.asarray(row["after_source"])
                error = np.r_[
                    target[:3, 3] - actual[:3, 3], Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
                ]
                close(row["error"], error)
                if row["contacts"]["finger_bowl"]:
                    observed_bound = separation_bound(
                        np.repeat(actual[None], 2, axis=0),
                        row["bowl_position"],
                        bounds["bowl_radius_m"],
                        bounds["collision_geometries"],
                    )
                    require(
                        observed_bound["endpoint_lower_bound_m"] <= 1e-9,
                        "collision envelope excludes an observed physical contact",
                    )
                    contact_support_checks += 1
                errors.append(error)
            errors = np.asarray(errors)
            close(result["position_rmse_m"], np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
            close(result["rotation_rmse_rad"], np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
            require(
                result["success"] == any(r["success"] for r in rows)
                and result["terminal_success"] == rows[-1]["success"],
                "task result differs",
            )
            bowl_center = np.asarray(rows[0]["bowl_position"])
            bowl_excursion = float(
                np.max(np.linalg.norm(np.array([r["bowl_position"] for r in rows]) - bowl_center, axis=1))
            )
            desired_points = np.r_[
                np.asarray(rows[0]["measured_source"])[None, :3, 3], np.asarray(expected_targets[:budget])[:, :3, 3]
            ]
            gap = (
                segment_distance(desired_points, bowl_center)
                - bounds["finger_radius_m"]
                - bounds["bowl_radius_m"]
                - bowl_excursion
            )
            reach = next(r for r in workspace["cases"] if r["trial"] == trial)
            excess = np.linalg.norm(desired_points - reach["center_world"], axis=1) - reach["radius_m"]
            ideal_poses = np.concatenate(
                [np.asarray(rows[0]["measured_source"])[None], np.asarray(expected_targets[:budget])]
            )
            gripper_gap = separation_bound(
                ideal_poses, bowl_center, bounds["bowl_radius_m"] + bowl_excursion, bounds["collision_geometries"]
            )
            result.update(
                stage=stage,
                directory=str(dest),
                privileged=stage != "wam",
                bowl_excursion_m=bowl_excursion,
                ideal_finger_bowl_separation_lower_bound_m=gap,
                max_bowl_lift_m=float(max(r["bowl_position"][2] for r in rows) - rows[0]["bowl_position"][2]),
                desired_source_min_z_m=float(np.min(desired_points[:, 2])),
                outside_conservative_workspace_endpoints=int(np.sum(excess[1:] > 1e-9)),
                max_workspace_radius_excess_m=float(np.max(excess)),
                ideal_gripper_endpoint_lower_bound_m=gripper_gap["endpoint_lower_bound_m"],
                ideal_gripper_linear_slerp_lower_bound_m=gripper_gap["linear_slerp_lower_bound_m"],
                raw_local_rotation_mean_executed_rad=float(np.mean(used_rotation_angles[:budget])),
                raw_opening_outside_executed_fraction=float(
                    np.mean((np.asarray(used_raw_openings[:budget]) < 0) | (np.asarray(used_raw_openings[:budget]) > 1))
                ),
            )
            results.append(result)
            for path in dest.glob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".png"):
                    artifacts[str(path)] = sha256(path)
    require(
        not args.case or {r["case"]["name"] for r in results} == set(args.case), "unknown or unverified selected case"
    )
    groups = defaultdict(list)
    for row in results:
        groups[row["case"]["name"]].append(row)
    grouped = []
    for name, rows in groups.items():
        grouped.append(
            dict(
                case=name,
                episodes=len(rows),
                successes=sum(r["success"] for r in rows),
                terminal_successes=sum(r["terminal_success"] for r in rows),
                position_rmse_mm=float(np.mean([r["position_rmse_m"] for r in rows]) * 1000),
                rotation_rmse_rad=float(np.mean([r["rotation_rmse_rad"] for r in rows])),
                finger_contact_steps=sum(r["finger_contact_steps"] for r in rows),
                ideal_finger_bowl_gap_min_mm=1000 * min(r["ideal_finger_bowl_separation_lower_bound_m"] for r in rows),
                raw_local_rotation_mean_executed_rad=float(
                    np.mean([r["raw_local_rotation_mean_executed_rad"] for r in rows])
                ),
                raw_opening_outside_executed_fraction=float(
                    np.mean([r["raw_opening_outside_executed_fraction"] for r in rows])
                ),
                ideal_gripper_linear_slerp_gap_min_mm=1000
                * min(r["ideal_gripper_linear_slerp_lower_bound_m"] for r in rows),
            )
        )
    summary = dict(
        complete=args.stage is None,
        integrity_passed=True,
        seed=c["seed"],
        verification_code=code_fingerprint(),
        formal_cases=len(results),
        formal_physics_steps=sum(r["steps"] for r in results),
        model_responses=len(contracts),
        groups=grouped,
        cases=results,
        worst_denormalization_error=max((r["denormalization_max_abs"] for r in contracts), default=None),
        primary_artifacts_sha256=artifacts,
        checked_revision_files=len(revisions),
        inputs_sha256=inputs,
        collision_bounds=bounds,
        conservative_workspace_bound=workspace,
        fixture_gripper_xml_sha256=gripper_xml[0],
        actual_contact_support_checks=contact_support_checks,
        inference_latency_excluded=True,
        hardware_executed=False,
        limits="Frozen checkpoint, one task and five known scenes. Privileged interventions are not pure WAM successes. A positive ideal finger separation concerns the prescribed endpoint path, not all possible robot motions.",
    )
    if args.stage:
        write_json(out / ("verification_" + "_".join(args.stage + (args.case or [])) + ".json"), summary)
    else:
        pairing = []
        for trial in c["trials"]:
            roots = [out / "sync" / case / f"trial{trial}" for case in ("wam_affine", "wam_response")]
            pixels = [np.asarray(Image.open(p / "input_0000.png")) for p in roots]
            actions = [
                np.asarray(json.loads((p / "response_0000.json").read_text())["outputs"][0]["content"]["action"])
                for p in roots
            ]
            require(np.array_equal(*pixels), "pure WAM comparison initial cameras differ")
            pairing.append(
                dict(
                    trial=trial,
                    initial_canvas_max_abs=0,
                    first_action_max_abs=float(np.max(abs(actions[0] - actions[1]))),
                )
            )
        summary["pure_wam_initial_pairing"] = pairing
        diagnostic = verify_quasistatic(c, out, Path(cc["output_dir"]))
        summary["quasistatic_cases"] = diagnostic
        summary["quasistatic_physics_steps"] = sum(r["physical_steps"] for r in diagnostic)
        write_json(out / "summary.json", summary)
        write_json("reports/libero_wam_effect_20261019.json", summary)
    print(json.dumps({k: summary[k] for k in ("formal_cases", "formal_physics_steps", "model_responses", "groups")}))


if __name__ == "__main__":
    main()

"""Independently reconstruct cached-endpoint fit certificates and validation scores."""

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.research.wam.training.clean.clean_training_aggregation import admit_aggregation
from s2a.research.wam.training.clean.clean_training_runtime import publish, reference
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require

CONFIG = Path("configs/wam/training/cached_endpoint_calibration_20261111.json")


def rotation(columns):
    columns = np.asarray(columns, dtype=np.float64)
    first, second = columns[..., :3], columns[..., 3:]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def integrate(current, action):
    position, attitude = current[:3].copy(), rotation(current[3:9])
    positions, attitudes = [], []
    for row in action:
        position = position + attitude @ row[:3]
        attitude = attitude @ rotation(row[3:9])
        positions.append(position.copy())
        attitudes.append(attitude.copy())
    return np.asarray(positions), np.asarray(attitudes), np.clip(action[:, 9], 0, 1)


def design(current, position, attitude, opening, config):
    base = rotation(current[3:9])
    u = (position - current[:3]) @ base
    v = Rotation.from_matrix(base.T @ attitude).as_rotvec()
    require(np.max(np.linalg.norm(v, axis=1)) < np.pi - config["pi_margin_rad"], "ambiguous rotation")
    fraction = np.arange(1, 41) / 40
    x = np.column_stack((u / config["position_scale_m"], v / config["rotation_scale_rad"], fraction))
    o = np.clip(opening, 0, 1)
    g = np.column_stack(
        (
            np.ones(40),
            o,
            np.maximum(o - 0.25, 0),
            np.maximum(o - 0.5, 0),
            np.maximum(o - 0.75, 0),
            np.full(40, current[9]),
            fraction,
        )
    )
    return x, g, u, v


def training_rows(evidence, config):
    registration = evidence.json(
        {"path": str(Path(config["fit_registration"]).resolve()), "sha256": config["fit_registration_sha256"]}
    )
    _, prepared, _, collections = admit_aggregation(registration, read_json=evidence.json)
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    normal = evidence.json(collections[0]["normalization_ref"])["transform"]
    mask = np.asarray(normal["normalization_mask"][:10], dtype=bool)
    scale, bias = (np.asarray(normal[k][:10]) for k in ("scale", "bias"))
    archived, xs, ys, gs, os_, truths = {}, [], [], [], [], []
    for request in collections[0]["requests"]:
        anchor = anchors[request["anchor_name"]]
        if anchor["name"] not in archived:
            with np.load(evidence.path(anchor["trajectory_ref"]), allow_pickle=False) as z:
                frame = anchor["anchor_frame"]
                archived[anchor["name"]] = z["state128"][frame : frame + 41].copy()
        states = archived[anchor["name"]]
        ledger = evidence.json(request["compact_ledger_ref"])
        point = evidence.json(ledger["points"][-1])
        with np.load(evidence.path(point["arrays_ref"]), allow_pickle=False) as z:
            raw = z["sample_active10_fp32"].astype(float) - point["metadata"]["sigma_actual"] * z[
                "raw_velocity_active10"
            ].astype(float)
            require(np.array_equal(raw, z["c0_active10"]), "raw terminal reconstruction changed")
        raw[:, mask] = (raw[:, mask] - bias[mask]) / scale[mask]
        pos, rot, opening = integrate(states[0], raw)
        x, g, u, v = design(states[0], pos, rot, opening, config)
        base = rotation(states[0, 3:9])
        true_u = (states[1:, :3] - states[0, :3]) @ base
        true_v = Rotation.from_matrix(base.T @ rotation(states[1:, 3:9])).as_rotvec()
        y = np.column_stack(((true_u - u) / config["position_scale_m"], (true_v - v) / config["rotation_scale_rad"]))
        xs.append(x)
        ys.append(y)
        gs.append(g)
        os_.append(opening)
        truths.append(states[1:, 9])
    require(len(archived) == 150 and len(xs) == 300, "training denominator differs")
    return tuple(map(np.concatenate, (xs, ys, gs, os_, truths)))


def check_fit(candidate, data, config):
    x, y, g, opening, truth = data
    augmented = np.vstack((x / np.sqrt(len(x)), np.sqrt(config["pose_ridge"]) * np.eye(7)))
    augmented_target = np.vstack((y / np.sqrt(len(x)), np.zeros((7, 6))))
    independent = np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
    pose_difference = float(np.max(abs(independent - np.asarray(candidate["pose_coefficients"]))))
    require(pose_difference < 1e-10, "independent augmented pose least squares differs")
    coef = np.asarray(candidate["opening_coefficients"])
    first, second = np.split(np.asarray(candidate["opening_lp"]["inequality_dual"]), 2)
    require(first.shape == second.shape == (len(g),), "LP dual length differs")
    residual_target = truth - opening
    primal = float(np.mean(abs(g @ coef - residual_target)) + config["opening_l1"] * np.sum(abs(coef)))
    dual = float(residual_target @ (first - second))
    sign_violation = max(0.0, float(np.max(np.r_[first, second])))
    coefficient_violation = max(0.0, float(np.max(abs(g.T @ (first - second))) - config["opening_l1"]))
    slack_violation = max(0.0, float(np.max(-first - second) - 1 / len(g)))
    require(
        abs(primal - candidate["opening_lp"]["objective"]) < 1e-8
        and abs(primal - dual) < 1e-7
        and max(sign_violation, coefficient_violation, slack_violation) < 1e-8,
        "opening primal/dual optimality certificate differs",
    )
    return dict(
        pose_coefficient_max_abs=pose_difference,
        opening_primal=primal,
        opening_dual=dual,
        opening_primal_dual_gap=abs(primal - dual),
        opening_dual_constraint_max_violation=max(sign_violation, coefficient_violation, slack_violation),
    )


def corrected(current, position, attitude, opening, candidate, config):
    x, g, u, v = design(current, position, attitude, opening, config)
    fraction = np.arange(1, 41) / 40
    delta = x @ np.asarray(candidate["pose_coefficients"])
    dp, dv = delta[:, :3] * config["position_scale_m"], delta[:, 3:] * config["rotation_scale_rad"]
    for values, limit in ((dp, config["position_cap_m"]), (dv, config["rotation_cap_rad"])):
        length = np.linalg.norm(values, axis=1)
        values *= np.minimum(1, limit * fraction / np.maximum(length, 1e-30))[:, None]
    base = rotation(current[3:9])
    position = current[:3] + (u + dp) @ base.T
    attitude = base @ Rotation.from_rotvec(v + dv).as_matrix()
    aperture = np.clip(opening + np.clip(g @ np.asarray(candidate["opening_coefficients"]), -0.25, 0.25), 0, 1)
    return position, attitude, aperture


def errors(states, position, attitude, opening):
    actual = states[1:]
    return dict(
        position=np.linalg.norm(position - actual[:, :3], axis=1),
        rotation=Rotation.from_matrix(rotation(actual[:, 3:9]) @ attitude.transpose(0, 2, 1)).magnitude(),
        opening=opening - actual[:, 9],
    )


def compare(left, right, tolerance=1e-10):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), "independent result keys differ")
        return max((compare(left[key], right[key], tolerance) for key in left), default=0.0)
    if isinstance(left, (list, tuple, np.ndarray)):
        a, b = np.asarray(left), np.asarray(right)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "result array differs")
        difference = float(np.max(abs(a - b))) if a.size else 0.0
        require(difference <= tolerance, "independent result array differs")
        return difference
    if isinstance(left, bool) or isinstance(right, bool):
        require(left is right, "independent boolean differs")
        return 0.0
    difference = abs(float(left) - float(right))
    require(difference <= tolerance, "independent scalar differs")
    return difference


def pool(rows):
    result = {}
    for horizon in (8, 16, 40):
        result[str(horizon)] = {}
        for method in ("raw", "endpoint_calibration", "hold", "constant_velocity"):
            pieces = {
                k: np.concatenate([r[method][k][:horizon] for r in rows]) for k in ("position", "rotation", "opening")
            }
            p, r, g = (pieces[k] for k in ("position", "rotation", "opening"))
            result[str(horizon)][method] = dict(
                valid_endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.mean(p**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
                opening_mae=float(np.mean(abs(g))),
                joint_score=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2)),
            )
    return result


def gate(groups):
    pooled = groups["pooled"]
    components = {}
    for h in (8, 16, 40):
        raw, candidate = (pooled[str(h)][k] for k in ("raw", "endpoint_calibration"))
        for field, additive in (("position_rmse_m", 0.002), ("rotation_rmse_rad", 0.01), ("opening_mae", 0.01)):
            components[f"h{h}_{field}"] = (
                candidate[field] <= raw[field] * 1.05 and candidate[field] <= raw[field] + additive
            )
    episodes = {
        name: all(
            group[str(h)]["endpoint_calibration"]["joint_score"] <= 1.05 * group[str(h)]["raw"]["joint_score"]
            for h in (8, 16)
        )
        for name, group in groups["by_episode"].items()
    }
    checks = dict(
        short_joint_improvement=all(
            pooled[str(h)]["endpoint_calibration"]["joint_score"] <= 0.90 * pooled[str(h)]["raw"]["joint_score"]
            for h in (8, 16)
        ),
        strictly_beats_hold_and_constant_velocity=all(
            pooled[str(h)]["endpoint_calibration"]["joint_score"] < pooled[str(h)][b]["joint_score"]
            for h in (8, 16)
            for b in ("hold", "constant_velocity")
        ),
        all_components_protected=all(components.values()),
        at_least_five_episodes=sum(episodes.values()) >= 5,
        no_episode_short_outlier=all(
            g[str(h)]["endpoint_calibration"]["joint_score"] <= 1.25 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_episode"].values()
            for h in (8, 16)
        ),
        both_seeds_protected=all(
            g[str(h)]["endpoint_calibration"]["joint_score"] <= 1.05 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_seed"].values()
            for h in (8, 16)
        ),
    )
    return dict(passed=all(checks.values()), checks=checks, component_checks=components, episode_passes=episodes)


def check_validation(evidence, candidate, validation, config):
    score = evidence.json(validation["score_ref"])
    numerical = evidence.json(score["numerical_ref"])
    prepared = evidence.json(score["prepared_ref"])
    anchors = {a["name"]: a for a in evidence.json(prepared["current_inputs_ref"])["current_inputs"]}
    original = {r["name"]: r for r in score["state_rows"]}
    recorded = {r["name"]: r for r in validation["state_rows"]}
    archives, result, maximum = {}, [], 0.0
    for item in numerical["generated_refs"]:
        if item["variant"] != "raw":
            continue
        row, anchor = original[item["name"]], anchors[item["anchor_name"]]
        episode = row["episode"]
        if episode not in archives:
            with np.load(evidence.path(score["future_archive_refs"][episode]), allow_pickle=False) as z:
                archives[episode] = z["state128"].copy()
        source, frame = archives[episode], anchor["anchor_frame"]
        states = source[frame : frame + 41]
        action = np.asarray(evidence.json(item["response_ref"])["outputs"][0]["content"]["action"])
        position, attitude, aperture = integrate(states[0], action)
        raw = errors(states, position, attitude, aperture)
        adjusted = errors(states, *corrected(states[0], position, attitude, aperture, candidate, config))
        for key in raw:
            maximum = max(maximum, compare(raw[key], recorded[row["name"]]["errors"]["wam"][key]))
            maximum = max(
                maximum, compare(adjusted[key], recorded[row["name"] + "_endpoint_calibration"]["errors"]["wam"][key])
            )
        # The original 056 independent audit already established these two history-only baselines.
        result.append(
            dict(
                episode=episode,
                task_index=row["task_index"],
                sampling_seed=row["sampling_seed"],
                raw=raw,
                endpoint_calibration=adjusted,
                hold=row["errors"]["hold"],
                constant_velocity=row["errors"]["constant_velocity"],
            )
        )
    require(len(result) == 48 and len(archives) == 6, "validation denominator differs")
    groups = {"pooled": pool(result)}
    for label, field in (("by_episode", "episode"), ("by_task", "task_index"), ("by_seed", "sampling_seed")):
        groups[label] = {str(v): pool([r for r in result if r[field] == v]) for v in sorted({r[field] for r in result})}
    metric_difference = compare(groups, validation["state_metrics"])
    require(gate(groups) == validation["original_054_gate_on_saved_responses"], "independent fixed gate differs")
    return dict(requests=len(result), max_endpoint_error_difference=maximum, max_metric_difference=metric_difference)


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "explicit CPU required")
    evidence = Evidence()
    config = evidence.json(reference(CONFIG))
    root = Path(config["output"])
    candidate_ref, validation_ref = reference(root / "candidate.json"), reference(root / "validation.json")
    candidate, validation = evidence.json(candidate_ref), evidence.json(validation_ref)
    require(validation["candidate_ref"] == candidate_ref, "frozen candidate identity differs")
    for inputs in (candidate["input_sha256"], validation["input_sha256"]):
        evidence.mapping(inputs)
    fit = check_fit(candidate, training_rows(evidence, config), config)
    scored = check_validation(evidence, candidate, validation, config)
    evidence.path(reference(Path(__file__)))
    evidence.finish()
    output = dict(
        complete=True,
        passed=True,
        candidate_ref=candidate_ref,
        validation_ref=validation_ref,
        fit_audit=fit,
        validation_audit=scored,
        input_sha256=evidence.inputs,
        new_fits=0,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        evidence_boundary="Independent arithmetic and LP certificate, not new sampling or physical task qualification.",
    )
    print(json.dumps(publish(root / "independent.json", output)), flush=True)


if __name__ == "__main__":
    main()

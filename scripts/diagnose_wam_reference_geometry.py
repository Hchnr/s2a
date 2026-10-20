"""Read-only geometry diagnostics for six raw alpha1 WAM intent-hold trials.

Target/object data enter offline metrics and plots only. No simulator, model,
expert trajectory, learned correction or online controller is run here.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, sha256
from s2a.geometry import rot6d_to_matrix

LABELS = ("8800_g10", "15400_g10")
RECORDS = ("task0_init0", "task1_init0", "task2_init0")
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")
DESTINATION = REPO / "outputs/wam_stability_20261021/reference_geometry_diagnosis"


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = str(Path(path).resolve())
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if path in self.hashes and self.hashes[path] != digest:
            raise ValueError(f"source changed during read-only diagnosis: {path}")
        self.hashes[path] = digest
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"source SHA changed: {path}")


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        key: float(value)
        for key, value in zip(("min", "median", "p90", "max"), np.percentile(values, [0, 50, 90, 100]), strict=True)
    }


def decode(anchor, actions, alpha=1):
    current, result = anchor.copy(), []
    for action in actions:
        delta = np.eye(4)
        delta[:3, :3] = rot6d_to_matrix(action[3:9])
        if alpha != 1:
            delta[:3, :3] = Rotation.from_rotvec(alpha * Rotation.from_matrix(delta[:3, :3]).as_rotvec()).as_matrix()
        delta[:3, 3] = action[:3]
        current = current @ delta
        result.append(current.copy())
    return np.asarray(result)


def longest(values):
    current = maximum = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(current, maximum)
    return maximum


def event(row, tcp, target, *, step):
    difference = tcp[:3, 3] - target
    return dict(
        step=step,
        endpoint_time_s=(step + 1) / 20,
        actual_target_body_origin_distance_mm=float(1000 * np.linalg.norm(difference)),
        actual_target_body_origin_xy_distance_mm=float(1000 * np.linalg.norm(difference[:2])),
        actual_above_target_body_origin_mm=float(1000 * difference[2]),
        desired_target_body_origin_xy_distance_mm=float(
            1000 * np.linalg.norm(np.asarray(row["desired_tcp"])[:2, 3] - target[:2])
        ),
        opening_target=row["opening_target"],
        opening_after=row["opening_after"],
        native_command=row["command"][-1],
        native_target_contact=row["objects"][row["_target"]]["grasp_contact"],
    )


def chunk_consumption(rows, chunks, index, actions, *, new):
    chunk = chunks[index]
    step = chunk["step"]
    next_step = chunks[index + 1]["step"] if index + 1 < len(chunks) else 600
    consumed = rows[step:next_step]
    times = np.asarray(
        [r["model_time_s"] for r in consumed] if new else [(r["step"] - step + 1) / 20 for r in consumed]
    )
    endpoints = (np.arange(40) + 1) / 20
    opening = np.clip(actions[:, 9], 0, 1)
    low = opening <= 0.2
    strong = opening <= chunk.get("anchor_opening", rows[step]["opening_before"]) - 0.2
    suffix = endpoints > times.max() + 1e-10
    raw_opening = [r["raw_opening_at_model_time"] if new else r["opening_target"] for r in consumed]
    return dict(
        physical_steps=next_step - step,
        physical_duration_s=(next_step - step) / 20,
        max_model_time_s=float(times.max()),
        model_horizon_s=2.0,
        generated_low_opening_endpoints=int(low.sum()),
        unconsumed_low_opening_endpoints=int((low & suffix).sum()),
        generated_strong_close_endpoints=int(strong.sum()),
        unconsumed_strong_close_endpoints=int((strong & suffix).sum()),
        consumed_raw_min_opening=float(min(raw_opening)),
        consumed_admitted_min_opening=float(min(r["opening_target"] for r in consumed)),
        first_generated_low_opening_time_s=None if not low.any() else float(endpoints[low][0]),
        suffix_status="none_model_horizon_consumed"
        if not suffix.any()
        else "replaced_before_model_horizon"
        if next_step < 600
        else "unconsumed_at_physics_budget_end",
        bridge_physics_steps=sum(bool(r.get("bridging", False)) for r in consumed),
        zero_tracking_scale_physics_steps=sum(
            r["reference_progress"] is not None and r["reference_progress"]["tracking_scale"] == 0 for r in consumed
        ),
    )


def load_case(case, evidence, target, h, *, new, governed=False):
    result = evidence.json(case / "result.json")
    assert result["error"] is None and result["steps"] == result["expected_steps"] == 600
    manifest = evidence.json(case / "manifest.json")
    evidence.json(case / "controller.json")
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    chunks = evidence.json(case / "chunks.json")
    assert len(rows) == 600 and len(chunks) == result["chunks"]
    if not governed:
        assert len(chunks) == 15
    if new:
        assert result["raw_checkpoint_reference"] is True and result["model_rotation_alpha"] == 1
    for path, digest in manifest["execution_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"frozen execution source changed: {path}")
        evidence.read(path)
    actual = np.asarray([r["after_source"] for r in rows]) @ h
    measured = np.asarray([r["measured_source"] for r in rows]) @ h
    desired = np.asarray([r["desired_tcp"] for r in rows])
    objects = np.asarray([r["objects"][target]["position"] for r in rows])
    dt = np.asarray([r["actual_dt"] for r in rows])
    np.testing.assert_allclose(dt, 0.05, atol=1e-9, rtol=0)
    errors = desired[:, :3, 3] - actual[:, :3, 3]
    angles = Rotation.from_matrix(desired[:, :3, :3] @ np.swapaxes(actual[:, :3, :3], 1, 2)).magnitude()
    contact = np.asarray([r["objects"][target]["grasp_contact"] for r in rows])
    success = np.asarray([r["success"] for r in rows])
    opening = np.asarray([r["opening_target"] for r in rows])
    close_commands = np.asarray([r["command"][-1] > 0 for r in rows])
    close_onsets = np.flatnonzero(close_commands & ~np.r_[False, close_commands[:-1]])
    closed_reference = opening <= 0.2
    low_opening_onsets = np.flatnonzero(closed_reference & ~np.r_[False, closed_reference[:-1]])
    for row in rows:
        row["_target"] = target
    onsets = [event(rows[i], actual[i], objects[i], step=int(i)) for i in close_onsets]
    low_onsets = [event(rows[i], actual[i], objects[i], step=int(i)) for i in low_opening_onsets]
    chunk_metrics, reference_angles, reference_speeds, reference_tilts = [], [], [], []
    for i, chunk in enumerate(chunks):
        actions = np.asarray(chunk["raw128"] if new else chunk["actions"])
        anchor = np.asarray(chunk["anchor_tcp"])
        targets = decode(anchor, actions)
        saved_targets = chunk["raw_targets_tcp"] if new else chunk["targets_tcp"]
        np.testing.assert_allclose(targets, saved_targets, atol=1e-11, rtol=0)
        if not governed:
            np.testing.assert_allclose(targets, desired[chunk["step"] : chunk["step"] + 40], atol=1e-11, rtol=0)
        assert actions.shape == (40, 128) and np.all(actions[:, 10:] == 0)
        response = evidence.json(case / f"response_{i:04d}.json")
        np.testing.assert_array_equal(actions, response["outputs"][0]["content"]["action"])
        evidence.json(case / f"request_{i:04d}.json")
        evidence.read(case / f"input_{i:04d}.png")
        query_step = chunk["step"]
        # Post-step row s-1 is the actual object sample available at query s.
        # At query0 no initial object sample was saved: use row0 and disclose.
        obj = objects[max(query_step - 1, 0)]
        start_vector = obj - anchor[:3, 3]
        end_vector = targets[-1, :3, 3] - anchor[:3, 3]
        distance = np.linalg.norm(targets[:, :3, 3] - obj, axis=1)
        xy = np.linalg.norm(targets[:, :2, 3] - obj[:2], axis=1)
        angle = Rotation.from_matrix(targets[:, :3, :3] @ anchor[:3, :3].T).magnitude()
        relative = Rotation.from_matrix(rot6d_to_matrix(actions[:, 3:9])).magnitude()
        reference_angles.extend(angle)
        reference_speeds.extend(relative * 20)
        reference_tilts.extend(np.arccos(np.clip(-targets[:, 2, 2], -1, 1)))
        counterfactual = {}
        for alpha in (0.5, 0.0):
            alternative = decode(anchor, actions, alpha)
            counterfactual[str(alpha)] = dict(
                unexecuted=True,
                last_world_position_change_mm=float(1000 * np.linalg.norm(alternative[-1, :3, 3] - targets[-1, :3, 3])),
                max_world_position_change_mm=float(
                    1000 * np.max(np.linalg.norm(alternative[:, :3, 3] - targets[:, :3, 3], axis=1))
                ),
                closest_query_target_body_origin_xy_mm=float(
                    1000 * np.min(np.linalg.norm(alternative[:, :2, 3] - obj[:2], axis=1))
                ),
            )
        chunk_metrics.append(
            dict(
                chunk=i,
                query_step=query_step,
                query_target_body_origin_source_step=max(query_step - 1, 0),
                query0_object_sample_is_first_post_step=query_step == 0,
                anchor_tcp_world=anchor[:3, 3].tolist(),
                query_target_body_origin_world=obj.tolist(),
                last_tcp_world=targets[-1, :3, 3].tolist(),
                world_displacement_mm=(1000 * end_vector).tolist(),
                towards_query_target_projection_mm=float(
                    1000 * np.dot(end_vector, start_vector) / np.linalg.norm(start_vector)
                ),
                cosine_towards_query_target=float(
                    np.dot(end_vector, start_vector)
                    / max(np.linalg.norm(end_vector) * np.linalg.norm(start_vector), 1e-15)
                ),
                anchor_target_body_origin_distance_mm=float(1000 * np.linalg.norm(start_vector)),
                last_target_body_origin_distance_mm=float(1000 * distance[-1]),
                closest_query_target_body_origin_distance_mm=float(1000 * distance.min()),
                closest_query_target_body_origin_xy_mm=float(1000 * xy.min()),
                min_reference_height_above_query_body_origin_mm=float(1000 * np.min(targets[:, 2, 3] - obj[2])),
                max_angle_from_anchor_rad=float(angle.max()),
                angular_path_length_rad=float(relative.sum()),
                translation_path_length_mm=float(1000 * np.linalg.norm(actions[:, :3], axis=1).sum()),
                opening_min=float(np.clip(actions[:, 9], 0, 1).min()),
                opening_last=float(np.clip(actions[-1, 9], 0, 1)),
                rotation_attenuation_geometry_counterfactual=counterfactual,
                consumption=chunk_consumption(rows, chunks, i, actions, new=new),
            )
        )
    near = np.linalg.norm(actual[:, :2, 3] - objects[:, :2], axis=1)
    distance = np.linalg.norm(actual[:, :3, 3] - objects, axis=1)
    actual_angles = Rotation.from_matrix(actual[:, :3, :3] @ measured[0, :3, :3].T).magnitude()
    report = dict(
        case=str(case),
        target=target,
        result=result,
        tcp_rmse_mm=float(1000 * np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        so3_rmse_rad=float(np.sqrt(np.mean(angles**2))),
        actual_world_min=actual[:, :3, 3].min(axis=0).tolist(),
        actual_world_max=actual[:, :3, 3].max(axis=0).tolist(),
        actual_path_length_mm=float(
            1000 * np.linalg.norm(np.diff(np.concatenate([measured[:1], actual])[:, :3, 3], axis=0), axis=1).sum()
        ),
        actual_angle_from_initial_rad=statistics(actual_angles),
        predicted_angle_from_chunk_anchor_rad=statistics(reference_angles),
        predicted_angular_speed_rad_s=statistics(reference_speeds),
        predicted_down_axis_tilt_rad=statistics(reference_tilts),
        predicted_angle_over_0p5rad_endpoints=int(np.sum(np.asarray(reference_angles) > 0.5)),
        predicted_angle_over_1rad_endpoints=int(np.sum(np.asarray(reference_angles) > 1)),
        predicted_angular_speed_over_1rad_s_endpoints=int(np.sum(np.asarray(reference_speeds) > 1)),
        target_body_origin_closest_xy_mm=float(1000 * near.min()),
        target_body_origin_closest_3d_mm=float(1000 * distance.min()),
        target_native_contact_endpoints=int(contact.sum()),
        target_native_contact_longest_endpoints=longest(contact),
        target_max_height_gain_mm=float(1000 * max(r["objects"][target]["height_gain_m"] for r in rows)),
        task_ever=bool(success.any()),
        task_terminal=bool(success[-1]),
        task_last40_stable=bool(success[-40:].all()),
        normalized_close_command_endpoints=int(close_commands.sum()),
        native_close_command_onsets=onsets,
        low_reference_opening_threshold=0.2,
        low_reference_opening_onsets=low_onsets,
        close_onsets_body_xy_over_50mm=sum(e["actual_target_body_origin_xy_distance_mm"] > 50 for e in onsets),
        chunks=chunk_metrics,
        governed=governed,
        reference_query_count=len(chunks),
        bridge_physics_steps=sum(bool(r.get("bridging", False)) for r in rows),
        bridge_recovery_advanced_steps=sum(bool(r.get("bridge_recovery_advanced", False)) for r in rows),
        zero_tracking_scale_physics_steps=sum(
            r["reference_progress"] is not None and r["reference_progress"]["tracking_scale"] == 0 for r in rows
        ),
        low_closing_suffix_replaced_chunks=sum(
            c["consumption"]["suffix_status"] == "replaced_before_model_horizon"
            and c["consumption"]["unconsumed_low_opening_endpoints"] > 0
            for c in chunk_metrics
        ),
    )
    np.testing.assert_allclose(report["tcp_rmse_mm"] / 1000, result["tcp_position_rmse_m"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(report["so3_rmse_rad"], result["rotation_rmse_rad"], atol=1e-12, rtol=0)
    return dict(
        report=report, rows=rows, actual=actual, desired=desired, objects=objects, chunks=chunks, manifest=manifest
    )


def compare(old, new):
    result = {}
    for field in ("measured_source", "after_source", "command", "opening_target", "opening_after"):
        a, b = np.asarray([r[field] for r in old["rows"]]), np.asarray([r[field] for r in new["rows"]])
        differences = np.max(np.abs(a - b).reshape(len(a), -1), axis=1)
        changed = np.flatnonzero(differences > 1e-12)
        result[field] = dict(
            first_difference_step_atol_1e_minus12=None if not len(changed) else int(changed[0]),
            max_abs=float(differences.max()),
        )
    a = np.asarray(old["chunks"][0]["actions"])
    b = np.asarray(new["chunks"][0]["raw128"])
    result["first_H40x128_exact"] = bool(np.array_equal(a, b))
    result["first_H40x128_max_abs"] = float(np.max(np.abs(a - b)))
    result["arm_command_first_step_exact"] = old["rows"][0]["command"][:6] == new["rows"][0]["command"][:6]
    return result


def figures(label, record, old, new, evidence, destination):
    import matplotlib.pyplot as plt
    from PIL import Image

    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), constrained_layout=True)
    indices = (0, len(new["chunks"]) // 2, len(new["chunks"]) - 1)
    for axis, index in zip(axes, indices, strict=True):
        path = Path(new["report"]["case"]) / f"input_{index:04d}.png"
        evidence.read(path)
        axis.imshow(Image.open(path))
        axis.set_title(
            f"{label}/{record}: query{index}, pre-step {new['chunks'][index]['step']}\nauxiliary top / wrist bottom"
        )
        axis.axis("off")
    for suffix in ("png", "pdf"):
        path = destination / f"{label}_{record}_actual_query_images.{suffix}"
        fig.savefig(path, dpi=160)
        paths.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    times = (np.arange(600) + 1) / 20
    new_name = f"raw alpha1 {new['report']['result']['variant']}"
    for data, color, name in ((old, "#0072B2", "old baseline"), (new, "#D55E00", new_name)):
        actual, desired, objects = data["actual"], data["desired"], data["objects"]
        axes[0, 0].plot(actual[:, 0, 3], actual[:, 1, 3], color=color, label=f"{name}: actual TCP")
        axes[0, 0].scatter(
            objects[0, 0], objects[0, 1], marker="x", color=color, label=f"{name}: target body origin, first endpoint"
        )
        for chunk in data["chunks"]:
            target = np.asarray(chunk["raw_targets_tcp"] if name.startswith("raw") else chunk["targets_tcp"])
            axes[0, 0].plot(target[:, 0, 3], target[:, 1, 3], color=color, alpha=0.2)
        axes[0, 1].plot(times, actual[:, 2, 3], color=color, label=f"{name}: actual z")
        axes[0, 1].plot(times, desired[:, 2, 3], color=color, alpha=0.5, linestyle="--", label=f"{name}: desired z")
        axes[0, 1].plot(times, objects[:, 2], color=color, alpha=0.3, linestyle=":")
        axes[0, 2].plot(
            times, np.linalg.norm(actual[:, :2, 3] - objects[:, :2], axis=1) * 1000, color=color, label=name
        )
        axes[1, 0].plot(
            times,
            np.linalg.norm(np.asarray([r["tcp_error"][:3] for r in data["rows"]]), axis=1) * 1000,
            color=color,
            label=f"{name}: TCP mm",
        )
        angle_axis = axes[1, 1]
        angle_axis.plot(
            times,
            np.linalg.norm(np.asarray([r["tcp_error"][3:] for r in data["rows"]]), axis=1),
            color=color,
            label=name,
        )
        axes[1, 2].plot(times, [r["opening_after"] for r in data["rows"]], color=color, label=f"{name}: actual opening")
        axes[1, 2].plot(times, [r["opening_target"] for r in data["rows"]], color=color, linestyle="--", alpha=0.4)
    labels = (
        ("world x (m)", "world y (m)"),
        ("endpoint time (s)", "world z (m); dotted=target body origin"),
        ("endpoint time (s)", "TCP/target body XY distance (mm)"),
        ("endpoint time (s)", "canonical TCP error (mm)"),
        ("endpoint time (s)", "SO(3) tracking error (rad)"),
        ("endpoint time (s)", "opening actual / dashed desired [0,1]"),
    )
    for axis, (xlabel, ylabel) in zip(axes.flat, labels, strict=True):
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    fig.suptitle(
        f"{label}/{record}: same initial scene, different closed-loop trajectories; seed42\nFaint XY lines are individual generated chunks, not one reanchored trajectory. Body-origin distances do not certify grasp feasibility."
    )
    for suffix in ("png", "pdf"):
        path = destination / f"{label}_{record}_geometry.{suffix}"
        fig.savefig(path, dpi=160)
        paths.append(path)
    plt.close(fig)
    return paths


def consumption_figure(label, data, destination):
    import matplotlib.pyplot as plt

    chunks = data["chunks"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index >= len(chunks):
            axis.axis("off")
            continue
        chunk = chunks[index]
        next_step = chunks[index + 1]["step"] if index + 1 < len(chunks) else 600
        rows = data["rows"][chunk["step"] : next_step]
        tau = np.asarray([r["model_time_s"] for r in rows])
        opening = np.clip(np.asarray(chunk["raw128"])[:, 9], 0, 1)
        axis.plot((np.arange(40) + 1) / 20, opening, color="#0072B2", label="full generated opening")
        axis.plot(
            tau, [r["opening_target"] for r in rows], color="#D55E00", linestyle="--", label="admitted vs model clock"
        )
        axis.plot(tau, [r["opening_after"] for r in rows], color="black", alpha=0.6, label="actual vs model clock")
        axis.axvline(tau.max(), color="red", linestyle=":", label="furthest model time consumed")
        axis.axvspan(tau.max(), 2, color="red", alpha=0.07)
        axis.set(
            title=f"query{index} at step{chunk['step']}; {len(rows) / 20:.2f}s physics\nmax model time{tau.max():.3f}s",
            xlabel="model time after query (s)",
            ylabel="opening [0,1]",
            xlim=(0, 2),
            ylim=(-0.05, 1.05),
        )
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(fontsize=6)
    figure.suptitle(
        f"{label}/bowl: raw alpha1 recovering_intent_hold; unconsumed suffix shaded\nLines use model time, not wall/physics time. Full generated future is diagnostic only. No native task success occurred."
    )
    paths = []
    for suffix in ("png", "pdf"):
        path = destination / f"{label}_bowl_governor_model_consumption.{suffix}"
        figure.savefig(path, dpi=160)
        paths.append(path)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("diagnostics require explicit empty CUDA and LD paths")
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    evidence, data, cases, comparisons = Evidence(), {}, [], []
    evidence.read(__file__)
    evidence.read(REPO / "src/s2a/geometry.py")
    evidence.read(REPO / "src/s2a/wam_regularization.py")
    evidence.read(REPO / "src/s2a/checkpoint_simulation.py")
    template = evidence.json(REPO / "configs/libero_wam_effect_20261019.json")
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    for label in LABELS:
        config = evidence.json(REPO / "configs" / f"wam_recovery_pilot_{label}_20261021.json")
        evidence.read(config["fixture"])
        ck = config["checkpoints"][0]["name"]
        for record, target in zip(RECORDS, TARGETS, strict=True):
            new_case = REPO / config["output_dir"] / "simulation" / f"{ck}_intent_hold" / f"{record}_s42"
            old_case = (
                REPO
                / "outputs/wam_stability_20261021/development"
                / label
                / "simulation"
                / f"{ck}_baseline"
                / f"{record}_s42"
            )
            old = load_case(old_case, evidence, target, h, new=False)
            new = load_case(new_case, evidence, target, h, new=True)
            comparison = compare(old, new)
            comparison["first_input_png_bytes_exact"] = evidence.read(old_case / "input_0000.png") == evidence.read(
                new_case / "input_0000.png"
            )
            comparisons.append(dict(label=label, record=record, comparison=comparison))
            cases.extend(
                [
                    dict(label=label, record=record, executor="old_baseline", **old["report"]),
                    dict(label=label, record=record, executor="raw_alpha1_intent_hold", **new["report"]),
                ]
            )
            data[(label, record)] = old, new
            if record == "task0_init0":
                gov_case = REPO / config["output_dir"] / "simulation" / f"{ck}_recovering_intent_hold" / f"{record}_s42"
                gov = load_case(gov_case, evidence, target, h, new=True, governed=True)
                cases.append(
                    dict(label=label, record=record, executor="raw_alpha1_recovering_intent_hold", **gov["report"])
                )
                data[(label, record + "_recovering")] = old, gov
    evidence.recheck()
    import matplotlib

    matplotlib.use("Agg")
    DESTINATION.mkdir(exist_ok=False)
    outputs = []
    for (label, record), (old, new) in data.items():
        outputs.extend(figures(label, record, old, new, evidence, DESTINATION))
        if new["report"]["governed"]:
            outputs.extend(consumption_figure(label, new, DESTINATION))
    metrics = DESTINATION / "geometry_diagnosis.json"
    write_new(
        metrics,
        dict(
            seed=args.seed,
            complete_raw_alpha1_cases=6,
            comparison_baseline_cases=6,
            complete_raw_alpha1_governor_bowl_cases=2,
            observed_physics_steps=8400,
            created_utc=datetime.now(timezone.utc).isoformat(),
            scope="read-only diagnosis of six raw alpha1 intent_hold, two raw alpha1 recovering bowl and six old g1.0 baselines; no new executions",
            limitations=[
                "Object poses/contact/success enter offline metrics and figures only; no privileged data is a control input.",
                "TCP-to-object body origin is a geometric proxy, not a semantic grasp center or a collision/IK feasibility certificate.",
                "Query0 object position uses the first post-step object sample, because no initial object positions were persisted.",
                "The two closed loops see different future images after their executions diverge; later path differences do not isolate a single causal factor.",
                "Alpha.5/0 curves are unexecuted kinematic reintegrations of saved raw chunks; modifying local rotation can modify world translation.",
                "No expert future, heldout task evaluation, new model inference, training or simulator is run in this script.",
            ],
            cases=cases,
            paired_initial_comparisons=comparisons,
        ),
    )
    outputs.append(metrics)
    evidence.recheck()
    manifest = DESTINATION / "manifest.json"
    write_new(
        manifest,
        dict(
            seed=args.seed,
            input_sha256=evidence.hashes,
            output_sha256={str(p): sha256(p) for p in outputs},
            input_hashes_verified_before_after=True,
        ),
    )
    print(json.dumps(dict(complete=True, output_dir=str(DESTINATION), manifest_sha256=sha256(manifest))))


if __name__ == "__main__":
    main()

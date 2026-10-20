"""Read-only, incremental integrity and causal-reference audit of 24 WAM pilots.

No model or simulator is imported or started. Native task/contact booleans are
recounted as metrics only; they never enter the independently replayed control.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.geometry import inverse, rot6d_to_matrix
from s2a.gripper import OpeningIntent
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

LABELS = ("8800_g15", "8800_g10", "15400_g15", "15400_g10")
TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f"evidence changed during audit: {path}")
        self.hashes[str(path)] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def check(self, path, digest):
        self.read(path)
        if self.hashes[str(Path(path).resolve())] != digest:
            raise ValueError(f"SHA256 mismatch: {path}")

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.check(path, digest)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def pose_difference(desired, measured):
    return np.r_[
        desired[:3, 3] - measured[:3, 3], Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
    ]


def interpolate(start, end, fraction):
    result = start.copy()
    result[:3, 3] += fraction * (end[:3, 3] - start[:3, 3])
    angle = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    result[:3, :3] = Rotation.from_rotvec(fraction * angle).as_matrix() @ start[:3, :3]
    return result


class GovernorReplay:
    """Independent reconstruction from saved measured states and WAM endpoints."""

    def __init__(self, config):
        self.c, self.admitted, self.opening = config, None, None

    def chunk(self, anchor, targets, openings, anchor_opening, dt):
        self.poses = np.concatenate([anchor[None], targets])
        self.values = np.r_[anchor_opening, openings]
        self.nominal = np.arange(len(targets) + 1) * dt
        if self.admitted is None:
            self.admitted, self.opening = anchor.copy(), float(anchor_opening)
        else:
            self.poses = np.concatenate([self.admitted[None], self.poses])
            self.values = np.r_[self.opening, self.values]
            self.nominal = np.r_[0.0, self.nominal]
        durations = []
        for i in range(len(self.poses) - 1):
            error = pose_difference(self.poses[i + 1], self.poses[i])
            durations.append(
                max(
                    self.nominal[i + 1] - self.nominal[i],
                    np.linalg.norm(error[:3]) / self.c["max_translation_speed_m_s"],
                    np.linalg.norm(error[3:]) / self.c["max_rotation_speed_rad_s"],
                    abs(self.values[i + 1] - self.values[i]) / self.c["max_opening_speed_s"],
                )
            )
        self.times, self.clock = np.r_[0.0, np.cumsum(durations)], 0.0
        return dict(retimed_duration_s=float(self.times[-1]), model_duration_s=float(self.nominal[-1]))

    def step(self, measured, dt):
        error = pose_difference(self.admitted, measured)
        position, angle = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        scale = float(
            np.clip(
                min(
                    (self.c["position_stop_m"] - position) / (self.c["position_stop_m"] - self.c["position_soft_m"]),
                    (self.c["rotation_stop_rad"] - angle) / (self.c["rotation_stop_rad"] - self.c["rotation_soft_rad"]),
                ),
                0,
                1,
            )
        )
        self.clock = min(self.clock + dt * scale, self.times[-1])
        segment = min(int(np.searchsorted(self.times, self.clock, side="right") - 1), len(self.poses) - 2)
        duration = self.times[segment + 1] - self.times[segment]
        fraction = (self.clock - self.times[segment]) / duration if duration else 1.0
        self.admitted = interpolate(self.poses[segment], self.poses[segment + 1], fraction)
        self.opening = float((1 - fraction) * self.values[segment] + fraction * self.values[segment + 1])
        return dict(
            target_tcp=self.admitted,
            opening=self.opening,
            model_time_s=float((1 - fraction) * self.nominal[segment] + fraction * self.nominal[segment + 1]),
            retimed_clock_s=float(self.clock),
            tracking_scale=scale,
            preceding_position_error_m=float(position),
            preceding_rotation_error_rad=float(angle),
            bridging=bool(segment == 0 and self.nominal[1] == 0),
            complete=bool(self.clock >= self.times[-1]),
        )


def same_mapping(saved, expected):
    if set(saved) != set(expected):
        raise ValueError("saved and reconstructed fields differ")
    for key, value in expected.items():
        if isinstance(value, bool):
            assert saved[key] is value
        else:
            np.testing.assert_allclose(saved[key], value, atol=1e-10, rtol=0, err_msg=key)


def metrics(rows, h):
    errors = np.asarray(
        [pose_difference(np.asarray(r["desired_tcp"]), np.asarray(r["after_source"]) @ h) for r in rows]
    )
    return dict(
        tcp_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows])),
    )


def longest(values):
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def compare_history(case, rows, chunks, evidence):
    old = Path("outputs/wam_horizon_diagnosis_20261020/execute40/simulation/8800_masked") / case.name
    previous = evidence.json(old / "result.json")
    assert previous["steps"] == previous["expected_steps"] == 600 and previous["error"] is None
    previous_rows = [json.loads(line) for line in evidence.read(old / "steps.jsonl").splitlines()]
    old_chunks = evidence.json(old / "chunks.json")
    fields = {}
    for key in (
        "measured_source",
        "desired_source",
        "after_source",
        "command",
        "opening_target",
        "opening_after",
        "actual_dt",
        "saturated",
        "rate_limited",
        "success",
    ):
        a, b = np.asarray([r[key] for r in rows]), np.asarray([r[key] for r in previous_rows])
        diff = np.any(a != b, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else a != b
        indices = np.flatnonzero(diff)
        fields[key] = dict(
            exact=bool(np.array_equal(a, b)),
            first_difference_step=None if not len(indices) else int(indices[0]),
            max_abs_difference=float(np.max(np.abs(a.astype(float) - b.astype(float)))),
        )
    query_checks = []
    for i, (new_chunk, old_chunk) in enumerate(zip(chunks, old_chunks, strict=True)):
        new_image, old_image = case / f"input_{i:04d}.png", old / f"input_{i:04d}.png"
        byte_equal = evidence.read(new_image) == evidence.read(old_image)
        old_response = evidence.json(old / f"response_{i:04d}.json")
        old_request = evidence.json(old / f"request_{i:04d}.json")
        new_request = evidence.json(case / f"request_{i:04d}.json")
        a, b = np.asarray(new_chunk["actions"]), np.asarray(old_chunk["actions"])
        np.testing.assert_array_equal(old_response["outputs"][0]["content"]["action"], b)
        query_checks.append(
            dict(
                chunk=i,
                image_bytes_exact=byte_equal,
                raw_H40x128_actions_exact=bool(np.array_equal(a, b)),
                raw_action_max_abs_difference=float(np.max(np.abs(a - b))),
                request_semantics_except_image_path_exact={k: v for k, v in new_request.items() if k != "vision_path"}
                == {k: v for k, v in old_request.items() if k != "vision_path"},
            )
        )
    return dict(
        historical_case=str(old.resolve()),
        queries=query_checks,
        fields=fields,
        all_physical_logged_fields_exact=all(x["exact"] for x in fields.values()),
        all_images_and_raw_actions_exact=all(
            x["image_bytes_exact"] and x["raw_H40x128_actions_exact"] for x in query_checks
        ),
        comparison_is_diagnostic_not_a_repair=True,
    )


def verify_case(config, record, variant, case, evidence, label):
    result = evidence.json(case / "result.json")
    manifest = evidence.json(case / "manifest.json")
    controller = evidence.json(case / "controller.json")
    chunks = evidence.json(case / "chunks.json")
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    assert result["steps"] == result["expected_steps"] == len(rows) == config["online_max_steps"] == 600
    assert result["error"] is None and manifest["source"] == result["source"] == "wam"
    assert manifest["experiment"] == config and manifest["record"] == record and manifest["variant"] == variant
    assert manifest["inference_latency_excluded"] is True and manifest["oracle_is_wam_success_evidence"] is False
    for path, digest in manifest["execution_sha256"].items():
        evidence.check(path, digest)
    template = evidence.json(config["policy_template"])
    evidence.check(template["tool_contract"], manifest["tool_sha256"])
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    ck = config["checkpoints"][0]
    checkpoint = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    info = manifest["server_info"]
    assert Path(info["checkpoint_path"]).resolve() == checkpoint.resolve() and not info["ema_weights"]
    assert info["acceptance_intervention"]["active_slot_mask"] is True
    assert ck["gpu"] in (4, 5, 6, 7)
    assert result["checkpoint"] == ck["name"] and result["variant"] == variant["name"] and result["sampling_seed"] == 42
    assert len(chunks) == result["chunks"] and [r["step"] for r in rows] == list(range(600))
    dt = 1 / config["control_fps"]
    np.testing.assert_allclose([r["actual_dt"] for r in rows], dt, atol=1e-9, rtol=0)
    assert chunks[0]["physics_sha256"] == controller["actual_initial_physics_sha256"] == record["initial_state_sha256"]
    assert variant["rotation"] == "response" and variant["gripper"] == "intent"
    contract = controller["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    adapter = OSCStateToAction(
        interface,
        contract["kp"],
        contract["kd"],
        dt,
        compensate=True,
        inverse_weight=config["inverse_weight"],
        max_command_rate=config["max_command_rate"],
    )
    intent = OpeningIntent(rows[0]["opening_before"], band=config["opening_hysteresis"])
    governor = GovernorReplay(config["reference_governor"])
    active, query_checks, model_progress = -1, [], []
    targets, openings = None, None
    previous_target, previous_opening, previous_progress = None, None, None
    max_speed = np.zeros(3)
    for step, row in enumerate(rows):
        measured = np.asarray(row["measured_source"])
        if step:
            np.testing.assert_array_equal(measured, rows[step - 1]["after_source"])
            np.testing.assert_allclose(
                row["motor_opening_before"], rows[step - 1]["motor_opening_after"], atol=1e-12, rtol=0
            )
        anchor = measured @ h
        new_query = (
            active < 0
            or (previous_progress["complete"] if variant["governor"] else step - chunks[active]["step"] >= len(targets))
            or step - chunks[active]["step"] >= config["max_observation_interval_steps"]
        )
        if new_query:
            active += 1
            chunk = chunks[active]
            assert chunk["step"] == step and chunk["physics_unchanged"] is True
            assert len(chunk["physics_sha256"]) == 64 and all(c in "0123456789abcdef" for c in chunk["physics_sha256"])
            np.testing.assert_allclose(chunk["anchor_tcp"], anchor, atol=1e-12, rtol=0)
            request = evidence.json(case / f"request_{active:04d}.json")
            response = evidence.json(case / f"response_{active:04d}.json")
            image_path = case / f"input_{active:04d}.png"
            evidence.read(image_path)
            pixels = np.asarray(Image.open(image_path).convert("RGB"))
            assert (
                pixels.shape == (960, 640, 3) and hashlib.sha256(pixels.tobytes()).hexdigest() == chunk["image_sha256"]
            )
            assert Path(request["vision_path"]).resolve() == image_path.resolve()
            expected_request = dict(
                name=f"s2a_{case.name}_{active}",
                model_mode="wam",
                prompt=record["task"],
                domain_name="franka_panda__parallel_gripper",
                image_size=960,
                action_chunk_size=config["online_chunk_size"],
                fps=config["control_fps"],
                num_frames=config["online_chunk_size"] + 1,
                seed=42,
                num_steps=config["sampling_steps"],
                guidance=config["guidance"],
                shift=config["shift"],
                extra=dict(
                    action128=dict(
                        action_domain=template["action_domain"],
                        action_dim_mask=[True] * 10 + [False] * 118,
                        action_fps=config["control_fps"],
                        task_title=record["task"],
                        selected_roles=["auxiliary_0", "left_wrist"],
                        camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                        return_video=False,
                    )
                ),
            )
            assert {k: v for k, v in request.items() if k != "vision_path"} == expected_request
            assert response["status"] == "success"
            content = response["outputs"][0]["content"]
            metadata = content["action128"]
            assert Path(metadata["checkpoint_path"]).resolve() == checkpoint.resolve() and metadata["seed"] == 42
            assert (
                metadata["action_units"] == "raw_physical_or_source_units"
                and metadata["action_domain"] == template["action_domain"]
            )
            assert (
                metadata["action_dim_mask"] == [True] * 10 + [False] * 118
                and metadata["action_fps"] == config["control_fps"]
            )
            actions = np.asarray(content["action"], dtype=float)
            assert actions.shape == (40, 128) and np.isfinite(actions).all() and np.all(actions[:, 10:] == 0)
            np.testing.assert_array_equal(actions, chunk["actions"])
            endpoints, pose = [], anchor.copy()
            for action in actions[: config["online_execution_horizon"]]:
                delta = np.eye(4)
                delta[:3, :3], delta[:3, 3] = rot6d_to_matrix(action[3:9]), action[:3]
                pose = pose @ delta
                endpoints.append(pose.copy())
            targets, openings = np.asarray(endpoints), np.clip(actions[: config["online_execution_horizon"], 9], 0, 1)
            np.testing.assert_allclose(targets, chunk["targets_tcp"], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(openings, chunk["openings"])
            if variant["governor"]:
                same_mapping(chunk["retiming"], governor.chunk(anchor, targets, openings, row["opening_before"], dt))
            else:
                assert chunk["retiming"] is None
            query_checks.append(
                dict(
                    chunk=active,
                    step=step,
                    physics_sha256=chunk["physics_sha256"],
                    raw_action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                )
            )
        assert row["chunk_index"] == active
        if variant["governor"]:
            previous_progress = governor.step(anchor, dt)
            same_mapping(row["reference_progress"], previous_progress)
            target, opening = previous_progress["target_tcp"], previous_progress["opening"]
            model_progress.append(
                dict(
                    step=step,
                    chunk=active,
                    model_time_s=previous_progress["model_time_s"],
                    bridging=previous_progress["bridging"],
                    tracking_scale=previous_progress["tracking_scale"],
                )
            )
        else:
            assert row["reference_progress"] is None
            offset = step - chunks[active]["step"]
            target, opening = targets[offset], openings[offset]
        np.testing.assert_allclose(row["desired_tcp"], target, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["desired_source"], target @ inverse(h), atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["opening_target"], opening, atol=1e-10, rtol=0)
        converted = adapter.convert(measured, np.asarray(row["desired_source"]), row["measured_twist"])
        np.testing.assert_allclose(
            row["command"], np.r_[converted["command"], intent.update(row["opening_target"])], atol=1e-12, rtol=0
        )
        assert row["saturated"] == converted["saturated"] and row["rate_limited"] == converted["rate_limited"]
        assert row["rotation_inverse"] is None and row["gripper"] is None
        if variant["governor"]:
            prior = anchor if previous_target is None else previous_target
            prior_opening = row["opening_before"] if previous_opening is None else previous_opening
            change = pose_difference(target, prior)
            speeds = np.array(
                [np.linalg.norm(change[:3]) / dt, np.linalg.norm(change[3:]) / dt, abs(opening - prior_opening) / dt]
            )
            limit = np.array(
                [
                    config["reference_governor"][k]
                    for k in ("max_translation_speed_m_s", "max_rotation_speed_rad_s", "max_opening_speed_s")
                ]
            )
            assert np.all(speeds <= limit + 1e-8)
            max_speed = np.maximum(max_speed, speeds)
        previous_target, previous_opening = target.copy(), opening
        np.testing.assert_allclose(
            row["tcp_error"], pose_difference(target, np.asarray(row["after_source"]) @ h), atol=1e-10, rtol=0
        )
    assert active + 1 == len(chunks)
    outcome = task_outcome(
        np.asarray([r["success"] for r in rows]), expected_steps=600, stable_steps=config["stable_success_last_steps"]
    )
    assert result["ever_success"] == outcome["ever"] and result["stable_success"] == outcome["stable"]
    assert result["final_success_steps"] == sum(r["success"] for r in rows[-40:])
    calculated = metrics(rows, h)
    for k, v in calculated.items():
        np.testing.assert_allclose(result[k], v, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["physical_elapsed_s"], sum(r["actual_dt"] for r in rows), atol=1e-12, rtol=0)
    target_name = TARGETS[record["task_id"]]
    objects = [r["objects"][target_name] for r in rows]
    contact = [o["grasp_contact"] for o in objects]
    assert all(isinstance(v, bool) for v in contact)
    initial_z = np.asarray([o["position"][2] - o["height_gain_m"] for o in objects])
    np.testing.assert_allclose(initial_z, initial_z[0], atol=1e-12, rtol=0)
    report = dict(
        label=label,
        record=record["name"],
        task_id=record["task_id"],
        checkpoint=ck["name"],
        guidance=config["guidance"],
        variant=variant["name"],
        sampling_seed=42,
        steps=600,
        outcome=outcome,
        metrics=calculated,
        last40_metrics=metrics(rows[-40:], h),
        target=dict(
            name=target_name,
            native_bilateral_contact_steps=sum(contact),
            longest_contact_endpoints=longest(contact),
            max_height_gain_mm=1000 * max(o["height_gain_m"] for o in objects),
            displacement_from_first_post_step_endpoint_mm=1000
            * float(
                np.max(np.linalg.norm(np.asarray([o["position"] for o in objects]) - objects[0]["position"], axis=1))
            ),
        ),
        queries=query_checks,
        governor=dict(
            enabled=variant["governor"],
            reconstructed_causally=True,
            bridge_steps=sum(p["bridging"] for p in model_progress),
            paused_steps=sum(p["tracking_scale"] == 0 for p in model_progress),
            max_reference_speeds=max_speed.tolist(),
            model_progress=model_progress,
            acceleration_limit_configured=False,
        ),
        pause_evidence=dict(
            runtime_before_after_full_state_equality_assertion_succeeded=True,
            first_query_sha_matches_actual_initial=True,
            full_state_before_after_arrays_not_persisted=True,
            independent_full_state_hash_reconstruction=False,
        ),
        limitations=[
            "Native task/contact predicates are recorded by the fixed simulator and recounted here; no new simulator instance is used to recompute predicates.",
            "Opening and body motion are metrics only; neither contacts nor task labels enter the independent replay of control.",
            "Development pilot evidence cannot establish unseen-scene generalization or a unique checkpoint/control root cause.",
        ],
    )
    if label == "8800_g15" and variant["name"] == "baseline":
        report["historical_baseline_repetition"] = compare_history(case, rows, chunks, evidence)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU audit requires empty CUDA and LD paths")
    root = Path.cwd()
    directory = root / "outputs/wam_stability_20261021/pilot_audits"
    directory.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    if not output.is_relative_to(directory):
        raise ValueError("audit output must be under pilot_audits/")
    if output.exists():
        raise FileExistsError(output)
    checks = directory / "incremental_checks"
    checks.mkdir(exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    verified = []
    pending = []
    errors = []
    all_hashes = {}
    check_hashes = {}
    for label in LABELS:
        ep = Evidence()
        config_path = Path("configs") / f"wam_stability_pilot_{label}_20261021.json"
        config = ep.json(config_path)
        assert (
            config["seed"] == 42
            and config["sampling_seeds"] == [42]
            and config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
        )
        ep.check(config["fixture"], config["fixture_sha256"])
        fixture = ep.json(config["fixture"])
        records = {r["name"]: r for r in fixture["records"]}
        for variant in config["variants"]:
            for name in config["records"]:
                ck = config["checkpoints"][0]["name"]
                case = Path(config["output_dir"]).resolve() / "simulation" / f"{ck}_{variant['name']}" / f"{name}_s42"
                result_path = case / "result.json"
                identity = f"{label}_{variant['name']}_{name}_s42"
                if not result_path.exists():
                    pending.append(dict(identity=identity, status="not_completed", result=str(result_path)))
                    continue
                try:
                    result = json.loads(result_path.read_bytes())
                except json.JSONDecodeError:
                    pending.append(dict(identity=identity, status="result_currently_being_written"))
                    continue
                if result.get("error") is not None:
                    errors.append(
                        dict(
                            identity=identity, error=result["error"], steps=result.get("steps"), result=str(result_path)
                        )
                    )
                    continue
                if result.get("steps") != 600:
                    pending.append(
                        dict(identity=identity, status="partial_not_task_failure", steps=result.get("steps"))
                    )
                    continue
                check_path = checks / f"{identity}.json"
                if check_path.exists():
                    saved = ep.json(check_path)
                    for path, digest in saved["files_sha256"].items():
                        ep.check(path, digest)
                    value = saved["verification"]
                else:
                    local = Evidence()
                    local.check(config_path, sha256(config_path))
                    local.check(config["fixture"], config["fixture_sha256"])
                    value = verify_case(config, records[name], variant, case, local, label)
                    local.recheck()
                    write_new(
                        check_path,
                        dict(
                            created_utc=datetime.now(timezone.utc).isoformat(),
                            audit_script_sha256=sha256(__file__),
                            verification=value,
                            files_sha256=local.hashes,
                        ),
                    )
                    ep.hashes.update(local.hashes)
                verified.append(value)
                check_hashes[str(check_path.resolve())] = sha256(check_path)
        ep.recheck()
        all_hashes.update(ep.hashes)
    counts = {}
    for label in LABELS:
        for variant in ("baseline", "governor_only"):
            rows = [x for x in verified if x["label"] == label and x["variant"] == variant]
            counts[f"{label}/{variant}"] = dict(
                expected=3,
                verified=len(rows),
                pending_or_execution_error=3 - len(rows),
                ever=sum(x["outcome"]["ever"] for x in rows),
                terminal=sum(x["outcome"]["terminal"] for x in rows),
                stable=sum(x["outcome"]["stable"] for x in rows),
            )
    complete = len(verified) == 24 and not pending and not errors
    for path, digest in all_hashes.items():
        assert sha256(path) == digest
    report = dict(
        snapshot_utc=snapshot,
        seed=42,
        expected=24,
        verified=len(verified),
        complete=complete,
        full_pilot_matrix_verified=complete,
        independent_acceptance=False,
        selection_allowed=False,
        counts=counts,
        cases=verified,
        pending_not_counted_as_task_failure=pending,
        execution_errors_excluded_from_success_denominators=errors,
        input_sha256=all_hashes,
        incremental_check_sha256=check_hashes,
        scope="Frozen four model/guidance settings ×baseline/governor_only×three old init0 scenes, samplingseed42; read-only audit, no new inference or physics",
    )
    write_new(output, report)
    print(
        json.dumps(
            dict(
                path=str(output),
                sha256=sha256(output),
                verified=len(verified),
                expected=24,
                complete=complete,
                counts=counts,
                errors=errors,
            )
        )
    )


if __name__ == "__main__":
    main()

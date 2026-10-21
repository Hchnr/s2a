"""Independent position/hold audit, explicitly forked from cd1580b recovery checker.

The original checker and frozen004 verifier remain unchanged. This version
registers exactly nine privileged oracle qualifications and24 WAM development
cases, adds independent scalar v3 gripper replay and003 first-query regression.
No dynamic monkeypatch, model request or simulator import is used.
"""

import argparse
import hashlib
import json
import math
import os
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from verify_gripper_position_hold import GripperReplay

from s2a.artifacts import REPO
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig
from s2a.gripper_position_hold import GripperPositionHold, GripperPositionHoldConfig
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

CONFIGS = (
    ("configs/gripper_position_hold_qualification_20261021.json", "oracle"),
    ("configs/wam_position_pilot_8800_g10_20261021.json", "wam"),
    ("configs/wam_position_pilot_15400_g10_20261021.json", "wam"),
)
TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.yaml_cache = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f"evidence changed during read-only audit: {path}")
        self.hashes[str(path)] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def check(self, path, expected):
        self.read(path)
        assert self.hashes[str(Path(path).resolve())] == expected, str(path)

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.check(path, digest)

    def yaml(self, path):
        path = str(Path(path).resolve())
        if path not in self.yaml_cache:
            self.yaml_cache[path] = yaml.safe_load(self.read(path))
        return self.yaml_cache[path]


def close(saved, expected, *, tolerance=1e-10):
    np.testing.assert_allclose(saved, expected, rtol=0, atol=tolerance)


def mapping(saved, expected):
    assert set(saved) == set(expected), (set(saved), set(expected))
    for key, value in expected.items():
        if value is None or isinstance(value, (str, bool)):
            assert saved[key] == value, key
        else:
            close(saved[key], value)


def error(desired, measured):
    return np.r_[
        desired[:3, 3] - measured[:3, 3], Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
    ]


def interpolate(start, end, fraction):
    result = np.eye(4)
    result[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    relative = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    result[:3, :3] = Rotation.from_rotvec(fraction * relative).as_matrix() @ start[:3, :3]
    return result


def sample(anchor, poses, openings, initial_opening, clock, dt):
    cursor = np.clip(clock / dt, 0, len(poses))
    index = min(math.floor(cursor), len(poses) - 1)
    fraction = cursor - index
    previous = anchor if index == 0 else poses[index - 1]
    previous_opening = initial_opening if index == 0 else openings[index - 1]
    return interpolate(previous, poses[index], fraction), float(
        (1 - fraction) * previous_opening + fraction * openings[index]
    )


def decode(anchor, actions):
    """Independent scalar Gram-Schmidt/local-delta integration."""
    current = anchor.copy()
    result = []
    for row in actions:
        first = row[3:6] / np.linalg.norm(row[3:6])
        second = row[6:9] - first * np.dot(first, row[6:9])
        second /= np.linalg.norm(second)
        delta = np.eye(4)
        delta[:3, :3] = np.column_stack((first, second, np.cross(first, second)))
        delta[:3, 3] = row[:3]
        current = current @ delta
        result.append(current.copy())
    return np.asarray(result)


def rmse(vectors):
    values = np.asarray(vectors)
    return dict(
        position_rmse_m=float(np.sqrt(np.mean(np.sum(values[:, :3] ** 2, axis=1)))),
        so3_rmse_rad=float(np.sqrt(np.mean(np.sum(values[:, 3:] ** 2, axis=1)))),
    )


def native_delta_rotation(vector):
    """robosuite's documented float32 quaternion conversion, without import.

    Source: transform_utils.quat2mat/axisangle2quat. Its deliberate float32
    quaternion roundtrip differs from a full-float64 SO(3) exponential.
    """
    angle = np.linalg.norm(vector)
    quaternion = np.array([1.0, 0, 0, 0], dtype=np.float32)
    if angle != 0:
        quaternion = np.r_[np.cos(angle / 2), vector / angle * np.sin(angle / 2)].astype(np.float32)
    # Numba/native runtime promotes the scalar division to float64. Cast the
    # dot explicitly so the auditor's NumPy2 weak-scalar rules do not round it.
    quaternion *= math.sqrt(2.0 / float(np.dot(quaternion, quaternion)))
    p = np.outer(quaternion, quaternion)
    return np.array(
        [
            [1.0 - float(p[2, 2]) - float(p[3, 3]), p[1, 2] - p[3, 0], p[1, 3] + p[2, 0]],
            [p[1, 2] + p[3, 0], 1.0 - float(p[1, 1]) - float(p[3, 3]), p[2, 3] - p[1, 0]],
            [p[1, 3] - p[2, 0], p[2, 3] + p[1, 0], 1.0 - float(p[1, 1]) - float(p[2, 2])],
        ]
    )


class RecoveryReplay:
    """Independent scalar reconstruction; does not call either governor module.

    The clock plan is rebuilt from segment lengths and three separate speed
    ceilings. Each bridge exception is retested in metres and radians against
    the preceding admitted reference, using current measured TCP only.
    """

    def __init__(self, limits):
        self.limits, self.admitted, self.opening = limits, None, None

    def chunk(self, anchor, targets, openings, measured_opening, dt):
        previous = self.admitted is not None
        opening = self.opening if previous else measured_opening
        self.poses = np.concatenate([anchor[None], targets])
        self.values = np.r_[opening, openings]
        self.nominal = np.arange(len(targets) + 1) * dt
        if previous:
            self.poses = np.concatenate([self.admitted[None], self.poses])
            self.values = np.r_[self.opening, self.values]
            self.nominal = np.r_[0.0, self.nominal]
        else:
            self.admitted, self.opening = anchor.copy(), float(opening)
        durations = []
        for i in range(len(self.poses) - 1):
            delta = error(self.poses[i + 1], self.poses[i])
            durations.append(
                max(
                    self.nominal[i + 1] - self.nominal[i],
                    np.linalg.norm(delta[:3]) / self.limits["max_translation_speed_m_s"],
                    np.linalg.norm(delta[3:]) / self.limits["max_rotation_speed_rad_s"],
                    abs(self.values[i + 1] - self.values[i]) / self.limits["max_opening_speed_s"],
                )
            )
        self.times, self.clock = np.r_[0.0, np.cumsum(durations)], 0.0
        return dict(
            retimed_duration_s=float(self.times[-1]),
            model_duration_s=float(self.nominal[-1]),
            measured_anchor_opening=measured_opening,
            opening_reference_at_anchor=float(opening),
            bridge_opening_preserved=previous,
        )

    def at(self, clock):
        segment = min(int(np.searchsorted(self.times, clock, side="right") - 1), len(self.poses) - 2)
        interval = self.times[segment + 1] - self.times[segment]
        fraction = (clock - self.times[segment]) / interval if interval else 1.0
        return (
            interpolate(self.poses[segment], self.poses[segment + 1], fraction),
            float((1 - fraction) * self.values[segment] + fraction * self.values[segment + 1]),
            float((1 - fraction) * self.nominal[segment] + fraction * self.nominal[segment + 1]),
            bool(segment == 0 and self.nominal[1] == 0),
        )

    def step(self, measured, dt):
        previous_clock, previous_target = self.clock, self.admitted.copy()
        preceding = error(previous_target, measured)
        distances = [float(np.linalg.norm(preceding[:3])), float(np.linalg.norm(preceding[3:]))]
        c = self.limits
        scale = float(
            np.clip(
                min(
                    (c["position_stop_m"] - distances[0]) / (c["position_stop_m"] - c["position_soft_m"]),
                    (c["rotation_stop_rad"] - distances[1]) / (c["rotation_stop_rad"] - c["rotation_soft_rad"]),
                ),
                0,
                1,
            )
        )
        self.clock = min(previous_clock + dt * scale, self.times[-1])
        self.admitted, self.opening, model_clock, bridge = self.at(self.clock)
        result = dict(
            target_tcp=self.admitted.copy(),
            opening=self.opening,
            model_time_s=model_clock,
            retimed_clock_s=float(self.clock),
            tracking_scale=scale,
            preceding_position_error_m=distances[0],
            preceding_rotation_error_rad=distances[1],
            bridging=bridge,
            complete=bool(self.clock >= self.times[-1]),
            nominal_tracking_scale=scale,
            bridge_recovery_advanced=False,
        )
        gate = None
        if self.nominal[1] == 0 and previous_clock < self.times[1] and scale < 1:
            candidate_clock = min(previous_clock + dt, self.times[1])
            if candidate_clock > self.clock:
                fraction = candidate_clock / self.times[1]
                candidate = interpolate(self.poses[0], self.poses[1], fraction)
                next_error = error(candidate, measured)
                following = [float(np.linalg.norm(next_error[:3])), float(np.linalg.norm(next_error[3:]))]
                allowed = all(b <= a + 1e-12 for a, b in zip(distances, following, strict=True)) and any(
                    b < a - 1e-12 for a, b in zip(distances, following, strict=True)
                )
                gate = dict(preceding=distances, candidate=following, allowed=allowed)
                if allowed:
                    self.clock, self.admitted = float(candidate_clock), candidate
                    self.opening = float((1 - fraction) * self.values[0] + fraction * self.values[1])
                    result.update(
                        target_tcp=candidate.copy(),
                        opening=self.opening,
                        model_time_s=0.0,
                        retimed_clock_s=self.clock,
                        tracking_scale=float((self.clock - previous_clock) / dt),
                        bridging=bool(self.clock < self.times[1]),
                        complete=False,
                        bridge_recovery_advanced=True,
                    )
        return result, gate


def verify_case(config, record, variant, case, evidence, *, source, checkpoint, full):
    result = evidence.json(case / "result.json")
    manifest = evidence.json(case / "manifest.json")
    controller = evidence.json(case / "controller.json")
    chunks = evidence.json(case / "chunks.json")
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == result["steps"] == result["expected_steps"] == config["online_max_steps"] == 600
    assert result["error"] is None and result["complete"] and result["execution_inputs_unchanged"]
    assert manifest["source"] == result["source"] == source
    assert manifest["experiment"] == config and manifest["variant"] == variant and manifest["record"] == record
    assert result["sampling_seed"] == 42 and result["variant"] == variant["name"]
    assert result["checkpoint"] == checkpoint and result["model_rotation_alpha"] == variant["model_rotation_alpha"]
    raw_flag = source == "wam" and variant["model_rotation_alpha"] == 1
    assert manifest["raw_checkpoint_reference"] is raw_flag and result["raw_checkpoint_reference"] is raw_flag
    assert manifest["reference_intervention"] == (
        "raw" if variant["model_rotation_alpha"] == 1 else "model_rotation_attenuation"
    )
    assert manifest["oracle_is_wam_success_evidence"] is False and result["oracle_is_wam_success_evidence"] is False
    assert manifest["inference_latency_excluded"] is True and result["inference_latency_excluded"] is True
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == manifest["environment"]["LD_LIBRARY_PATH"] == ""
    for path, signature in manifest["execution_sha256"].items():
        evidence.check(path, signature)
    required_runtime = [
        "src/s2a/stability_recovery_simulation.py",
        "src/s2a/stability_position_simulation.py",
        "scripts/run_wam_position_matrix.py",
        "src/s2a/gripper_position_hold.py",
        "src/s2a/gripper_intent_hold.py",
        "src/s2a/reference_governor_v2.py",
        "src/s2a/wam_regularization.py",
        "src/s2a/osc.py",
        "src/s2a/action_reference.py",
    ]
    assert all(str((REPO / p).resolve()) in manifest["execution_sha256"] for p in required_runtime)
    assert set(manifest["simulator_source_sha256"]) <= set(manifest["execution_sha256"])
    evidence.check(case / "model.xml", manifest["initial_model_xml_sha256"])
    template = evidence.json(config["policy_template"])
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    q = np.asarray(controller["source_from_control_rotation"])
    close(q.T @ q, np.eye(3))
    close(np.linalg.det(q), 1)
    assert chunks and len(chunks) == result["chunks"]
    assert chunks[0]["step"] == 0
    assert controller["actual_initial_physics_sha256"] == record["initial_state_sha256"]
    assert (
        hashlib.sha256(np.asarray(record["initial_state"], dtype=float).tobytes()).hexdigest()
        == record["initial_state_sha256"]
    )
    assert chunks[0]["physics_before_sha256"] == controller["actual_initial_physics_sha256"]
    close(controller["aligned_initial_motor_target"], rows[0]["motor_target_before"])
    initial = rows[0]["opening_before"]
    target = (0.079 * initial + 0.001) / 0.04 - 1
    close(controller["aligned_initial_motor_target"], [target, -target])
    dt = 1 / config["control_fps"]
    contract = controller["contract"]
    assert contract["gripper_model"] == "PandaGripper" and contract["control_frame"] == "MuJoCo world"
    assert contract["control_freq"] == 20 and contract["model_timestep"] == 0.002
    substeps = round(dt / contract["model_timestep"])
    assert substeps == 25
    close(substeps * contract["model_timestep"], dt)
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    arm = OSCStateToAction(
        interface,
        contract["kp"],
        contract["kd"],
        dt,
        compensate=True,
        inverse_weight=config["inverse_weight"],
        max_command_rate=config["max_command_rate"],
    )
    position = variant["gripper"] == "position_hold"
    gripper_type, config_type, config_name = (
        (GripperPositionHold, GripperPositionHoldConfig, "gripper_position_hold")
        if position
        else (GripperIntentHold, GripperIntentHoldConfig, "gripper_intent_hold")
    )
    gripper = gripper_type(initial, config=config_type(**config[config_name]))
    scalar_gripper = GripperReplay(initial, config[config_name], position)
    assert controller["gripper_config"] == asdict(gripper.config)
    assert asdict(gripper.config) == asdict(config_type())
    previous = np.zeros(7)
    references, native_source_outcome = None, None
    if source == "oracle":
        assert checkpoint is None and variant["model_rotation_alpha"] == 1
        assert not manifest["raw_checkpoint_reference"]
        assert manifest["oracle_reference_mode"] == "fixed_world_chunks_selected_by_current_physical_step"
        paths = [p for p in manifest["execution_sha256"] if p.endswith("/reference.npz")]
        assert len(paths) == 1
        with np.load(paths[0]) as data:
            references, oracle_opening = data["poses"] @ h, data["openings"]
        native_folder = Path(paths[0]).parent
        native_rows = [json.loads(line) for line in evidence.read(native_folder / "steps.jsonl").splitlines()]
        native_result = evidence.json(native_folder / "result.json")
        assert native_result["error"] is None and native_result["steps"] == len(native_rows) == 600
        native_source_outcome = task_outcome(
            [r["success"] for r in native_rows], expected_steps=600, stable_steps=config["stable_success_last_steps"]
        )
        assert native_source_outcome["stable"]
        close(references, np.asarray([r["after_source"] for r in native_rows]) @ h)
        close(oracle_opening, [r["opening_after"] for r in native_rows])
    else:
        ck = next(c for c in config["checkpoints"] if c["name"] == checkpoint)
        info = manifest["server_info"]
        expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        assert Path(info["checkpoint_path"]).resolve() == expected.resolve() and not info["ema_weights"]
        assert Path(info["config_file"]).resolve() == (Path(ck["run"]) / "config.yaml").resolve()
        assert info["acceptance_intervention"]["active_slot_mask"] is True and ck["gpu"] in (4, 5, 6, 7)
        for relative in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
            evidence.check(REPO / relative, info["acceptance_intervention"]["code"]["files_sha256"][relative])
    tracking = {
        name: []
        for name in (
            "admitted",
            "raw_at_model_clock",
            "effective_at_model_clock",
            "raw_at_nominal_clock",
            "effective_at_nominal_clock",
        )
    }
    zero_motor_steps, max_goal_rotation_error = 0, 0.0
    governor = RecoveryReplay(config["reference_governor"]) if full and variant["governor"] else None
    bridge_gates = []
    max_reference_speed = np.zeros(3)
    chunks_by_step = {c["step"]: (i, c) for i, c in enumerate(chunks)}
    assert len(chunks_by_step) == len(chunks)
    chunk, count = None, 0
    for step, row in enumerate(rows):
        assert row["step"] == step
        due = step == 0
        if step:
            due = (
                rows[step - 1]["reference_progress"]["complete"]
                if variant["governor"]
                else step - chunk["step"] >= count
            ) or step - chunk["step"] >= config["max_observation_interval_steps"]
        assert bool(due) == (step in chunks_by_step), "requery timing differs from causal completion/budget"
        close(row["actual_dt"], dt, tolerance=1e-9)
        close(row["physical_time_after_s"] - row["physical_time_before_s"], dt, tolerance=1e-9)
        if step:
            close(row["measured_source"], rows[step - 1]["after_source"])
            close(row["gripper_qpos_before"], rows[step - 1]["gripper_qpos_after"])
            close(row["opening_before"], rows[step - 1]["opening_after"])
            close(row["motor_opening_before"], rows[step - 1]["motor_opening_after"])
            close(row["motor_target_before"], rows[step - 1]["motor_target_after"])
            close(row["physical_time_before_s"], rows[step - 1]["physical_time_after_s"])
        measured, after, desired = (np.asarray(row[k]) for k in ("measured_source", "after_source", "desired_source"))
        desired_tcp = np.asarray(row["desired_tcp"])
        close(desired @ h, desired_tcp)
        if step in chunks_by_step:
            chunk_index, chunk = chunks_by_step[step]
            anchor = np.asarray(chunk["anchor_tcp"])
            close(anchor, measured @ h)
            close(chunk["anchor_opening"], row["opening_before"])
            assert chunk["physics_unchanged"]
            assert chunk["physics_before_sha256"] == chunk["physics_after_sha256"]
            close(chunk["physics_before_time_s"], row["physical_time_before_s"])
            close(chunk["physics_after_time_s"], row["physical_time_before_s"])
            raw, effective = np.asarray(chunk["raw128"]), np.asarray(chunk["effective128"])
            assert raw.ndim == 2 and raw.shape[1] == 128 and np.isfinite(raw).all() and not raw[:, 10:].any()
            assert effective.shape == raw.shape and np.isfinite(effective).all() and not effective[:, 10:].any()
            close(chunk["raw_targets_tcp"], decode(anchor, raw))
            close(chunk["effective_targets_tcp"], decode(anchor, effective))
            close(effective[:, :3], raw[:, :3])
            close(effective[:, 9:], raw[:, 9:])
            alpha = variant["model_rotation_alpha"]
            if alpha == 1:
                np.testing.assert_array_equal(raw, effective)
            else:
                # Per-delta SO(3), not cumulative orientation, is attenuated.
                for a, b in zip(raw, effective, strict=True):
                    ar = decode(np.eye(4), a[None])[0][:3, :3]
                    br = decode(np.eye(4), b[None])[0][:3, :3]
                    close(br, Rotation.from_rotvec(alpha * Rotation.from_matrix(ar).as_rotvec()).as_matrix())
            openings = np.clip(raw[:, 9], 0, 1)
            close(chunk["openings"], openings)
            count = chunk["execution_endpoint_count"]
            assert count == min(config["online_execution_horizon"], len(raw))
            raw_poses = np.asarray(chunk["raw_targets_tcp"])[:count]
            effective_poses = np.asarray(chunk["effective_targets_tcp"])[:count]
            if source == "oracle":
                close(chunk["raw_targets_tcp"], references[step : step + len(raw)])
                close(openings, oracle_opening[step : step + len(raw)])
                assert chunk["response_identity"] is None and chunk["request_sha256"] is None
            else:
                verify_query(config, case, chunk_index, chunk, evidence, checkpoint)
            if governor is not None:
                mapping(
                    chunk["retiming"],
                    governor.chunk(anchor, effective_poses, openings[:count], row["opening_before"], dt),
                )
        assert row["chunk_index"] == chunk_index and row["chunk_physical_step"] == step - chunk["step"]
        assert step - chunk["step"] < config["max_observation_interval_steps"]
        if not variant["governor"]:
            assert step - chunk["step"] < count
            close(desired_tcp, effective_poses[step - chunk["step"]])
            close(row["opening_target"], openings[step - chunk["step"]])
            assert row["reference_progress"] is None and not row["bridging"] and not row["bridge_recovery_advanced"]
            close(row["model_time_s"], (step - chunk["step"] + 1) * dt)
            close(row["retimed_clock_s"], row["model_time_s"])
        elif governor is not None:
            expected, bridge_gate = governor.step(measured @ h, dt)
            mapping(row["reference_progress"], expected)
            close(row["desired_tcp"], expected["target_tcp"])
            close(row["opening_target"], expected["opening"])
            for key in ("model_time_s", "retimed_clock_s", "bridging", "bridge_recovery_advanced"):
                close(row[key], expected[key])
            if bridge_gate is not None:
                bridge_gates.append(dict(step=step, **bridge_gate))
            previous_pose = anchor if step == 0 else np.asarray(rows[step - 1]["desired_tcp"])
            previous_opening = initial if step == 0 else rows[step - 1]["opening_target"]
            motion = error(desired_tcp, previous_pose)
            speeds = np.array(
                [
                    np.linalg.norm(motion[:3]) / dt,
                    np.linalg.norm(motion[3:]) / dt,
                    abs(row["opening_target"] - previous_opening) / dt,
                ]
            )
            max_reference_speed = np.maximum(max_reference_speed, speeds)
            limits = config["reference_governor"]
            assert np.all(
                speeds
                <= np.array(
                    [
                        limits["max_translation_speed_m_s"],
                        limits["max_rotation_speed_rad_s"],
                        limits["max_opening_speed_s"],
                    ]
                )
                + 1e-9
            )
            previous_clock = 0 if step == chunk["step"] else rows[step - 1]["model_time_s"]
            assert previous_clock - 1e-10 <= row["model_time_s"] <= count * dt + 1e-10
        raw_pose, raw_opening = sample(
            anchor, raw_poses, openings[:count], chunk["anchor_opening"], row["model_time_s"], dt
        )
        effective_pose, effective_opening = sample(
            anchor, effective_poses, openings[:count], chunk["anchor_opening"], row["model_time_s"], dt
        )
        close(row["raw_desired_tcp"], raw_pose)
        close(row["effective_desired_tcp"], effective_pose)
        close(row["raw_opening_at_model_time"], raw_opening)
        close(row["effective_opening_at_model_time"], effective_opening)
        nominal_time = (step - chunk["step"] + 1) * dt
        nominal_raw, _ = sample(anchor, raw_poses, openings[:count], chunk["anchor_opening"], nominal_time, dt)
        nominal_effective, _ = sample(
            anchor, effective_poses, openings[:count], chunk["anchor_opening"], nominal_time, dt
        )
        for key, target_pose in [
            ("admitted", desired_tcp),
            ("raw_at_model_clock", raw_pose),
            ("effective_at_model_clock", effective_pose),
            ("raw_at_nominal_clock", nominal_raw),
            ("effective_at_nominal_clock", nominal_effective),
        ]:
            tracking[key].append(error(target_pose, after @ h))
        close(row["tcp_error"], tracking["admitted"][-1])
        close(row["source_error"], error(desired, after))
        close(row["raw_model_tcp_error"], tracking["raw_at_model_clock"][-1])
        close(row["effective_model_tcp_error"], tracking["effective_at_model_clock"][-1])
        reconstructed = arm.convert(measured, desired, row["measured_twist"])
        close(row["command"][:6], reconstructed["command"])
        assert row["saturated"] == reconstructed["saturated"] and row["rate_limited"] == reconstructed["rate_limited"]
        motor_before, motor_after = np.asarray(row["motor_target_before"]), np.asarray(row["motor_target_after"])
        motor_width = float(np.clip((0.04 * (motor_before[0] + 1) - 0.001) / 0.079, 0, 1))
        close(row["motor_opening_before"], motor_width)
        for label in ("before", "after"):
            qpos = row[f"gripper_qpos_{label}"]
            close(row[f"opening_{label}"], np.clip((qpos[0] - qpos[1] - 0.001) / 0.079, 0, 1))
        motor_after_width = float(np.clip((0.04 + 0.02 * (motor_after[0] - motor_after[1]) - 0.001) / 0.079, 0, 1))
        close(row["motor_opening_after"], motor_after_width)
        grip = gripper.update(
            row["opening_target"],
            row["opening_before"],
            motor_opening=motor_width,
            previous_command=previous[-1],
            dt=dt,
        )
        mapping(row["gripper"], asdict(grip))
        scalar_command = scalar_gripper.step(
            row["opening_target"],
            row["opening_before"],
            motor_width,
            previous[-1],
            dt,
        )
        assert row["command"][-1] == grip.command == scalar_command
        assert row["gripper"]["closing_episode"] == scalar_gripper.latched
        command = np.asarray(row["command"])
        assert command[-1] in (-1, 0, 1)
        assert np.all(command[:6] >= interface.input_min - 1e-12)
        assert np.all(command[:6] <= interface.input_max + 1e-12)
        assert np.max(np.abs(command[:6] - previous[:6])) <= config["max_command_rate"] * dt + 1e-12
        close(motor_after, np.clip(motor_before + np.array([-1, 1]) * 0.01 * substeps * np.sign(command[-1]), -1, 1))
        close(row["actual_gripper_actuator_control"], [0.02, -0.02] + 0.02 * motor_after)
        if command[-1] == 0:
            np.testing.assert_array_equal(motor_before, motor_after)
            zero_motor_steps += 1
        delta = interface.decode(command[:6])
        close(row["controller_goal_position_after"], measured[:3, 3] + delta[:3])
        expected_goal = native_delta_rotation(delta[3:]) @ measured[:3, :3] @ q
        if not np.any(delta[3:]):
            expected_goal = np.asarray(row["controller_goal_before"])
        max_goal_rotation_error = max(
            max_goal_rotation_error, float(np.max(np.abs(np.asarray(row["controller_goal_after"]) - expected_goal)))
        )
        # Native quat2mat deliberately uses float32; the actual controller
        # SITE matrix is not logged per endpoint. A uniform declared numeric
        # tolerance covers that vendor conversion, not a task-success gate.
        close(row["controller_goal_after"], expected_goal, tolerance=1e-6)
        close(row["controller_goal_source_rotation_before"], np.asarray(row["controller_goal_before"]) @ q.T)
        previous = command
    outcome = task_outcome(
        [r["success"] for r in rows], expected_steps=600, stable_steps=config["stable_success_last_steps"]
    )
    assert result["task_outcome"] == outcome
    assert result["stable_success"] == outcome["stable"] and result["ever_success"] == outcome["ever"]
    target = TARGETS[record["task_id"]]
    for name in rows[0]["objects"]:
        first = rows[0]["objects"][name]
        initial_height = first["position"][2] - first["height_gain_m"]
        for row in rows:
            obj = row["objects"][name]
            assert type(obj["grasp_contact"]) is bool
            close(obj["height_gain_m"], obj["position"][2] - initial_height)
    admitted = rmse(tracking["admitted"])
    close(result["tcp_position_rmse_m"], admitted["position_rmse_m"])
    close(result["rotation_rmse_rad"], admitted["so3_rmse_rad"])
    consumption = []
    for i, chunk in enumerate(chunks):
        final_step = chunks[i + 1]["step"] - 1 if i + 1 < len(chunks) else len(rows) - 1
        consumed = rows[final_step]["model_time_s"]
        available = chunk["execution_endpoint_count"] * dt
        generated = len(chunk["raw128"]) * dt
        consumption.append(
            dict(
                chunk=i,
                first_step=chunk["step"],
                final_step=final_step,
                physical_steps=final_step - chunk["step"] + 1,
                generated_model_s=generated,
                execution_prefix_model_s=available,
                consumed_model_s=consumed,
                prefix_fraction=min(consumed / available, 1.0),
                discarded_on_requery_s=max(0.0, generated - consumed) if i + 1 < len(chunks) else 0.0,
                remaining_at_budget_s=max(0.0, generated - consumed) if i + 1 == len(chunks) else 0.0,
            )
        )
    model_consumed = sum(c["consumed_model_s"] for c in consumption)
    return dict(
        case=str(case),
        source=source,
        checkpoint=checkpoint,
        variant=variant["name"],
        record=record["name"],
        status="verified_basic" if not full else "verified",
        steps=600,
        chunks=len(chunks),
        oracle_native_source_outcome=native_source_outcome,
        independent_scalar_gripper_replay="passed",
        first_query003_regression=verify_first_query003(config, record, variant, case, chunks[0], evidence, checkpoint)
        if source == "wam"
        else None,
        outcome=outcome,
        zero_motor_steps=zero_motor_steps,
        controller_goal_rotation_max_abs_reconstruction_error=max_goal_rotation_error,
        controller_goal_rotation_tolerance=1e-6,
        tracking={k: rmse(v) for k, v in tracking.items()},
        target_grasp_contact_steps=sum(r["objects"][target]["grasp_contact"] for r in rows),
        object_metrics={
            name: dict(
                contact_steps=sum(r["objects"][name]["grasp_contact"] for r in rows),
                max_height_gain_m=max(r["objects"][name]["height_gain_m"] for r in rows),
            )
            for name in rows[0]["objects"]
        },
        independent_governor_replay=("passed" if governor else "pending") if variant["governor"] else "not_applicable",
        governor_diagnostics=dict(
            chunk_consumption=consumption,
            consumed_model_s=model_consumed,
            generated_model_s=sum(c["generated_model_s"] for c in consumption),
            model_consumed_per_physical_second=model_consumed / (len(rows) * dt),
            generated_reference_fraction=model_consumed / sum(c["generated_model_s"] for c in consumption),
            discarded_on_requery_s=sum(c["discarded_on_requery_s"] for c in consumption),
            remaining_at_budget_s=sum(c["remaining_at_budget_s"] for c in consumption),
            bridging_endpoint_count=sum(r["bridging"] for r in rows),
            bridge_recovery_advanced_count=sum(r["bridge_recovery_advanced"] for r in rows),
            paused_clock_endpoints=sum(
                r["reference_progress"] is not None and r["reference_progress"]["tracking_scale"] == 0 for r in rows
            ),
            nominal_tracking_paused_endpoints=sum(
                r["reference_progress"] is not None and r["reference_progress"]["nominal_tracking_scale"] == 0
                for r in rows
            ),
            max_admitted_reference_speeds=max_reference_speed.tolist() if governor else None,
            independently_recomputed_bridge_gates=bridge_gates,
        ),
        component_ablation=variant["model_rotation_alpha"] == 0,
        raw_checkpoint_evidence=source == "wam" and variant["model_rotation_alpha"] == 1,
    )


def verify_query(config, case, index, chunk, evidence, checkpoint):
    """Independently bind saved pixels/request/response to each generated chunk."""
    ck = next(c for c in config["checkpoints"] if c["name"] == checkpoint)
    assert ck["gpu"] in (4, 5, 6, 7)
    expected_checkpoint = (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
    request_path, response_path = (case / f"{kind}_{index:04d}.json" for kind in ("request", "response"))
    evidence.check(request_path, chunk["request_sha256"])
    evidence.check(response_path, chunk["response_sha256"])
    request, response = evidence.json(request_path), evidence.json(response_path)
    assert response["status"] == "success"
    assert request["model_mode"] == "wam" and request["seed"] == 42
    assert request["action_chunk_size"] == config["online_chunk_size"]
    assert request["fps"] == config["control_fps"] and request["num_frames"] == config["online_chunk_size"] + 1
    assert request["num_steps"] == config["sampling_steps"] and request["guidance"] == config["guidance"]
    assert request["shift"] == config["shift"]
    assert request["prompt"] == evidence.json(case / "manifest.json")["record"]["task"]
    extra = request["extra"]["action128"]
    mask = [True] * 10 + [False] * 118
    assert extra["action_dim_mask"] == mask and all(type(v) is bool for v in extra["action_dim_mask"])
    template = evidence.json(config["policy_template"])
    assert extra["action_domain"] == template["action_domain"]
    assert extra["action_fps"] == config["control_fps"] and extra["return_video"] is False
    assert extra["selected_roles"] == ["auxiliary_0", "left_wrist"]
    assert extra["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    image = case / f"input_{index:04d}.png"
    assert Path(request["vision_path"]).resolve() == image.resolve()
    evidence.read(image)
    pixels = np.asarray(Image.open(image))
    assert pixels.shape == (960, 640, 3) and pixels.dtype == np.uint8
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == chunk["image_sha256"]
    args = response["args"]
    for key in ("seed", "fps", "num_frames", "action_chunk_size", "guidance", "num_steps", "shift", "prompt"):
        assert args[key] == request[key], key
    assert args["condition_frame_indexes_vision"] == [0] and args["action_path"] is None
    assert args["native_prompt_upsampling"] is False and args["model_mode"] == "wam"
    content = response["outputs"][0]["content"]
    identity = content["action128"]
    assert Path(identity["checkpoint_path"]).resolve() == expected_checkpoint
    assert type(identity["seed"]) is int and identity["seed"] == request["seed"]
    assert identity["action_domain"] == extra["action_domain"] and identity["action_dim_mask"] == mask
    assert all(type(v) is bool for v in identity["action_dim_mask"])
    assert identity["action_units"] == "raw_physical_or_source_units"
    assert identity["action_fps"] == config["control_fps"]
    close(identity["action_timestamps_seconds"], np.arange(config["online_chunk_size"]) / config["control_fps"])
    mapping(chunk["response_identity"], {k: v for k, v in identity.items() if k != "normalized_action"})
    np.testing.assert_array_equal(content["action"], chunk["raw128"])
    assert len(content["action"]) == config["online_chunk_size"]
    saved_config = evidence.yaml(Path(ck["run"]) / "config.yaml")
    spec = saved_config["model"]["config"]["action128_spec"]
    transform = spec["transforms"][spec["registry"]["domains"].index(extra["action_domain"])]
    normalized = np.asarray(identity["normalized_action"], dtype=np.float32)
    assert normalized.shape == (config["online_chunk_size"], 128) and np.isfinite(normalized).all()
    assert not normalized[:, 10:].any()
    expected = normalized.copy()
    normalization_mask = np.asarray(transform["normalization_mask"], dtype=bool)
    expected[:, normalization_mask] = (
        expected[:, normalization_mask] - np.asarray(transform["bias"], dtype=np.float32)[normalization_mask]
    ) / np.asarray(transform["scale"], dtype=np.float32)[normalization_mask]
    expected[:, 10:] = 0
    close(content["action"], expected, tolerance=1e-8)


def verify_registration(config, source):
    """Only the already registered004/005 budget is admitted by this version."""
    fixed = dict(
        seed=42,
        control_fps=20,
        online_max_steps=600,
        online_chunk_size=40,
        online_execution_horizon=40,
        max_observation_interval_steps=80,
        stable_success_last_steps=40,
        inverse_weight=0.5,
        max_command_rate=5.0,
        sampling_steps=35,
        guidance=1.0,
        shift=5.0,
    )
    assert all(config[key] == value for key, value in fixed.items())
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
    assert config["sampling_seeds"] == [42]
    assert asdict(GripperIntentHoldConfig(**config["gripper_intent_hold"])) == asdict(GripperIntentHoldConfig())
    assert asdict(GripperPositionHoldConfig(**config["gripper_position_hold"])) == asdict(GripperPositionHoldConfig())
    expected = (
        [
            ("baseline_v2", "intent_hold", False, 1),
            ("position_hold", "position_hold", False, 1),
            ("recovering_position_hold", "position_hold", "recovering", 1),
        ]
        if source == "oracle"
        else [
            ("position_hold", "position_hold", False, 1),
            ("recovering_position_hold", "position_hold", "recovering", 1),
            ("recovering_position_hold_alpha05", "position_hold", "recovering", 0.5),
            ("recovering_position_hold_alpha0", "position_hold", "recovering", 0),
        ]
    )
    assert [(v["name"], v["gripper"], v["governor"], v["model_rotation_alpha"]) for v in config["variants"]] == expected
    assert all(v["rotation"] == "response" for v in config["variants"])
    if source == "wam":
        assert "oracle_reference_root" not in config
        assert len(config["checkpoints"]) == 1
        checkpoint = config["checkpoints"][0]
        assert (checkpoint["name"], checkpoint["gpu"], checkpoint["port"]) in (
            ("8800_masked", 5, 9345),
            ("15400_masked", 7, 9347),
        )


def verify_004_gate(evidence):
    """Bind the immutable original verdict and subsequent qpos supplement."""
    root = REPO / "outputs/wam_stability_20261021/gripper_position_hold_qualification"
    verdict_path = root / "qualification_independent_verification.json"
    supplement_path = root / "qualification_qpos_opening_supplement.json"
    verdict, supplement = evidence.json(verdict_path), evidence.json(supplement_path)
    assert verdict["qualified"] and verdict["complete"] and verdict["seed"] == 42
    assert verdict["verified_cases"] == 9 and verdict["verified_physics_steps"] == 5400
    assert verdict["baseline_exact"]["all_physical_fields_exact"]
    assert supplement["qualified"] and supplement["verified_cases"] == 9
    assert supplement["verified_physics_steps"] == 5400 and supplement["verified_continuity_pairs"] == 5391
    assert supplement["all_saved_input_sha256_rechecked"] and not supplement["physics_or_model_started"]
    evidence.check(verdict_path, supplement["original_verdict_sha256"])
    for document in (verdict, supplement):
        for path, signature in document["input_sha256"].items():
            evidence.check(path, signature)
    gate_path = REPO / "outputs/wam_stability_20261021/position_pilot_coordinators/qualification_gate.json"
    gate = evidence.json(gate_path)
    assert gate["qualified_cases"] == 9 and gate["physics_steps"] == 5400 and gate["baseline_exact"]
    assert gate["all_current_input_sha256_match"]
    assert set(gate["qualification_artifacts"]) == {str(verdict_path), str(supplement_path)}
    for path, signature in gate["qualification_artifacts"].items():
        evidence.check(path, signature)
    launch = evidence.json(gate_path.parent / "launch.json")
    evidence.check(gate_path, launch["qualification_gate_sha256"])
    assert launch["wam_cases_budget"] == 24 and launch["physics_steps_budget"] == 14400
    assert launch["model_query_budget_upper_bound"] == 360
    return dict(
        qualified=True,
        oracle_cases=9,
        oracle_physics_steps=5400,
        baseline_physical_fields_exact=True,
        original_verdict_sha256=evidence.hashes[str(verdict_path)],
        qpos_supplement_sha256=evidence.hashes[str(supplement_path)],
        root_admission_gate_sha256=evidence.hashes[str(gate_path)],
        all_original_inputs_rehashed=True,
    )


def verify_first_query003(config, record, variant, case, chunk, evidence, checkpoint):
    """Pair each005 first observation/output with its actual003 counterpart."""
    names = {
        "position_hold": "intent_hold",
        "recovering_position_hold": "recovering_intent_hold",
        "recovering_position_hold_alpha05": "recovering_intent_hold_alpha05",
        "recovering_position_hold_alpha0": "recovering_intent_hold_alpha0",
    }
    family = {"8800_masked": "8800_g10", "15400_masked": "15400_g10"}[checkpoint]
    previous = (
        REPO
        / "outputs/wam_stability_20261021/recovery_development"
        / family
        / "simulation"
        / f"{checkpoint}_{names[variant['name']]}"
        / f"{record['name']}_s42"
    )
    snapshot_path = REPO / "outputs/wam_stability_20261021/recovery_full_complete_20261021.json"
    evidence.check(snapshot_path, "b5ad882e7e696ec7d57a05987585c7ca5eea2ffc4c957c48ac00459c99a30eef")
    snapshot = evidence.json(snapshot_path)
    assert snapshot["whole_registered_protocol_complete"]
    assert chunk["step"] == 0
    for filename in ("input_0000.png", "request_0000.json", "response_0000.json", "chunks.json"):
        path = previous / filename
        evidence.check(path, snapshot["source_before_after_hashes"][str(path)])
    old_request, new_request = evidence.json(previous / "request_0000.json"), evidence.json(case / "request_0000.json")
    for key in (
        "model_mode",
        "seed",
        "fps",
        "num_frames",
        "action_chunk_size",
        "guidance",
        "num_steps",
        "shift",
        "prompt",
        "extra",
    ):
        assert old_request[key] == new_request[key], key
    old_response, new_response = (
        evidence.json(previous / "response_0000.json"),
        evidence.json(case / "response_0000.json"),
    )
    old_content, new_content = (response["outputs"][0]["content"] for response in (old_response, new_response))
    old_chunks = evidence.json(previous / "chunks.json")
    np.testing.assert_array_equal(old_content["action"], new_content["action"])
    np.testing.assert_array_equal(old_content["action"], old_chunks[0]["raw128"])
    np.testing.assert_array_equal(new_content["action"], chunk["raw128"])
    np.testing.assert_array_equal(
        old_content["action128"]["normalized_action"], new_content["action128"]["normalized_action"]
    )
    close(old_chunks[0]["anchor_tcp"], chunk["anchor_tcp"], tolerance=0)
    assert old_chunks[0]["physics_before_sha256"] == chunk["physics_before_sha256"]
    assert evidence.read(previous / "input_0000.png") == evidence.read(case / "input_0000.png")
    return dict(
        source_case=str(previous),
        image_png_bytes_exact=True,
        full40x128_raw_exact=True,
        full40x128_normalized_exact=True,
        fixed_initial_anchor_exact=True,
        initial_physics_sha_exact=True,
        model_conditions_exact=True,
        scope="first query only; later causal observations/trajectories can differ",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basic-oracle", action="store_true")
    parser.add_argument(
        "--reuse-verified", help="prior full artifact with identical auditor bytes and unchanged inputs"
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("read-only audit requires empty CUDA and LD paths")
    output = Path(args.output).resolve()
    if not output.is_relative_to(REPO / "outputs") or output.exists():
        raise ValueError("new audit artifact must be below repository outputs; refusing overwrite")
    evidence, cases, expected_groups = Evidence(), [], []
    evidence.read(__file__)
    qualification = verify_004_gate(evidence)
    reusable = {}
    if args.reuse_verified:
        previous_path = Path(args.reuse_verified).resolve()
        if not previous_path.is_relative_to(REPO / "outputs") or args.basic_oracle:
            raise ValueError("reuse requires a prior full repository output artifact")
        previous = evidence.json(previous_path)
        assert previous["scope"] == "full" and previous["schema_version"] == "s2a.position-readonly-audit.v1"
        own_source = str(Path(__file__).resolve())
        assert previous["source_before_after_hashes"][own_source] == evidence.hashes[own_source]
        for path, signature in previous["source_before_after_hashes"].items():
            evidence.check(path, signature)
        reusable = {str(Path(c["case"]).resolve()): c for c in previous["cases"] if c["status"] == "verified"}
    for config_path, source in CONFIGS:
        if args.basic_oracle and source != "oracle":
            continue
        config = evidence.json(config_path)
        verify_registration(config, source)
        evidence.check(config["fixture"], config["fixture_sha256"])
        fixtures = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
        checkpoint = None if source == "oracle" else config["checkpoints"][0]["name"]
        for variant in config["variants"]:
            expected_groups.append(
                dict(
                    source=source,
                    checkpoint=checkpoint,
                    variant=variant["name"],
                    model_rotation_alpha=variant["model_rotation_alpha"],
                    expected_cases=len(config["records"]) * len(config["sampling_seeds"]),
                )
            )
            for name in config["records"]:
                for seed in config["sampling_seeds"]:
                    case = (
                        Path(config["output_dir"])
                        / "simulation"
                        / f"{checkpoint or source}_{variant['name']}"
                        / f"{name}_s{seed}"
                    )
                    identity = dict(
                        case=str(case),
                        source=source,
                        checkpoint=checkpoint,
                        variant=variant["name"],
                        record=name,
                        sampling_seed=seed,
                    )
                    if not (case / "result.json").exists():
                        cases.append(dict(identity, status="partial" if case.exists() else "missing"))
                        continue
                    cached = reusable.get(str(case.resolve()))
                    if cached is not None:
                        cases.append(dict(cached, reused_from=str(Path(args.reuse_verified).resolve())))
                        continue
                    try:
                        # Completion JSON can briefly exist before its single
                        # write finishes. Do not fingerprint a partial marker.
                        json.loads((case / "result.json").read_bytes())
                    except json.JSONDecodeError:
                        cases.append(dict(identity, status="partial", reason="result marker being written"))
                        continue
                    result = evidence.json(case / "result.json")
                    if result["error"]:
                        cases.append(dict(identity, status="execution_error", error=result["error"]))
                        continue
                    try:
                        cases.append(
                            verify_case(
                                config,
                                fixtures[name],
                                variant,
                                case,
                                evidence,
                                source=source,
                                checkpoint=checkpoint,
                                full=not args.basic_oracle,
                            )
                        )
                    except Exception as exc:
                        cases.append(
                            dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc())
                        )
    evidence.read(__file__)
    evidence.recheck()
    oracle = [c for c in cases if c["source"] == "oracle"]
    basic_gate = (
        qualification["qualified"]
        and len(oracle) == 9
        and all(
            c["status"] in ("verified", "verified_basic")
            and c["outcome"]["stable"]
            and c["tracking"]["admitted"]["position_rmse_m"] <= 0.01
            and c["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
            and c["independent_scalar_gripper_replay"] == "passed"
            for c in oracle
        )
    )
    full_oracle_gate = basic_gate and all(
        c["independent_governor_replay"] in ("passed", "not_applicable") for c in oracle
    )
    groups = []
    for group in expected_groups:
        items = [c for c in cases if all(c[k] == group[k] for k in ("source", "checkpoint", "variant"))]
        verified = [c for c in items if c["status"] in ("verified", "verified_basic")]
        complete = len(verified) == group["expected_cases"]
        groups.append(
            dict(
                group,
                complete=complete,
                verified_cases=len(verified),
                stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
                ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
                terminal_successes_among_verified=sum(c["outcome"]["terminal"] for c in verified),
                stable_rate=sum(c["outcome"]["stable"] for c in verified) / len(verified) if complete else None,
                raw_checkpoint_reference=group["source"] == "wam" and group["model_rotation_alpha"] == 1,
                model_reference_changed=group["model_rotation_alpha"] != 1,
                component_ablation_only=group["model_rotation_alpha"] == 0,
                tracking_pooled_equal_length_cases={
                    ref: {
                        metric: float(np.sqrt(np.mean([c["tracking"][ref][metric] ** 2 for c in verified])))
                        for metric in ("position_rmse_m", "so3_rmse_rad")
                    }
                    for ref in verified[0]["tracking"]
                }
                if complete
                else None,
                consumed_model_per_physical_second=[
                    c["governor_diagnostics"]["model_consumed_per_physical_second"] for c in verified
                ],
            )
        )
    value = dict(
        schema_version="s2a.position-readonly-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="basic_oracle" if args.basic_oracle else "full",
        reused_verified_cases=sum("reused_from" in c for c in cases),
        cases=cases,
        groups=groups,
        oracle_basic_gate_passed=basic_gate,
        oracle_full_independent_gate_passed=full_oracle_gate,
        independent_governor_replay_complete=not args.basic_oracle
        and len(cases) == 33
        and all(c["status"] == "verified" for c in cases),
        completed_cases_independent_replay_passed=not args.basic_oracle
        and all(c["status"] == "verified" for c in cases if c["status"] not in ("missing", "partial")),
        qualification004=qualification,
        registered_oracle_cases=9,
        registered_wam_cases=24,
        oracle_in_wam_denominator=False,
        whole_registered_protocol_complete=not args.basic_oracle
        and len(cases) == 33
        and all(c["status"] == "verified" for c in cases),
        source_before_after_hashes=evidence.hashes,
        warnings=[
            "Task/contact are recorded metrics only, never replay inputs.",
            "Target fingerpad contact0 does not imply native task failure; these endpoint boolean metrics do not certify grasp force or continuous contact.",
            "Oracle is privileged qualification, not WAM success.",
            "Smaller admitted tracking error does not prove equal nominal reference following.",
            "Only old development scenes are covered, not fresh unseen-scene acceptance or real hardware.",
            "Alpha0 holds the measured anchor orientation per query, not the whole episode; it is only a component ablation, not full/raw checkpoint correctness.",
            "Alpha.5 changes the generated model reference. Local translation slots unchanged does not preserve the world path.",
            "Per-query full-state hash/time equality is authenticated from frozen logger; intermediate full physical arrays were not retained for independent rehashing.",
            "Controller goal matrix comparison uses uniform1e-6 vendor float32 numeric tolerance; all state/error/control checks retain their stricter tolerances. BODY quaternion observation merely reorders body_xquat; no claimed float32 sensor cast.",
            "Package versions/source files are authenticated; no complete OS/shared-library binary closure is claimed.",
            "Separate synthetic mechanical-blockage qualification remains unproven; this audit does not promote broad recovery safety.",
            "Any missing/partial/error case prevents whole-protocol completion; partial group counts do not establish a winning setting.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(payload)
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                oracle_basic_gate_passed=basic_gate,
                statuses={s: sum(c["status"] == s for c in cases) for s in {c["status"] for c in cases}},
            )
        )
    )


if __name__ == "__main__":
    main()

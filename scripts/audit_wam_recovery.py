"""Read-only incremental recovery audit; no model or simulator is imported.

The basic oracle gate authenticates the complete physical/control evidence.
Full audit additionally reconstructs reference recovery independently. Native
task/contact fields are counted as metrics and never replay inputs.
"""

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.gripper import OpeningIntent
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

CONFIGS = (
    ("configs/wam_recovery_executor_qualification_20261021.json", "oracle"),
    ("configs/wam_recovery_pilot_8800_g10_20261021.json", "wam"),
    ("configs/wam_recovery_pilot_15400_g10_20261021.json", "wam"),
)
TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


class Evidence:
    def __init__(self):
        self.hashes = {}

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
    assert manifest["oracle_is_wam_success_evidence"] is False and result["oracle_is_wam_success_evidence"] is False
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == manifest["environment"]["LD_LIBRARY_PATH"] == ""
    for path, signature in manifest["execution_sha256"].items():
        evidence.check(path, signature)
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
    assert chunks[0]["physics_before_sha256"] == controller["actual_initial_physics_sha256"]
    close(controller["aligned_initial_motor_target"], rows[0]["motor_target_before"])
    initial = rows[0]["opening_before"]
    target = (0.079 * initial + 0.001) / 0.04 - 1
    close(controller["aligned_initial_motor_target"], [target, -target])
    dt = 1 / config["control_fps"]
    contract = controller["contract"]
    assert contract["gripper_model"] == "PandaGripper" and contract["control_frame"] == "MuJoCo world"
    substeps = round(dt / contract["model_timestep"])
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
    gripper = GripperIntentHold(initial, config=GripperIntentHoldConfig(**config["gripper_intent_hold"]))
    intent = OpeningIntent(initial, band=config["opening_hysteresis"])
    assert controller["gripper_config"] == asdict(gripper.config)
    previous = np.zeros(7)
    references = None
    if source == "oracle":
        assert checkpoint is None and variant["model_rotation_alpha"] == 1
        assert not manifest["raw_checkpoint_reference"]
        assert manifest["oracle_reference_mode"] == "fixed_world_chunks_selected_by_current_physical_step"
        paths = [p for p in manifest["execution_sha256"] if p.endswith("/reference.npz")]
        assert len(paths) == 1
        with np.load(paths[0]) as data:
            references, oracle_opening = data["poses"] @ h, data["openings"]
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
    chunks_by_step = {c["step"]: (i, c) for i, c in enumerate(chunks)}
    assert len(chunks_by_step) == len(chunks)
    for step, row in enumerate(rows):
        assert row["step"] == step
        close(row["actual_dt"], dt, tolerance=1e-9)
        close(row["physical_time_after_s"] - row["physical_time_before_s"], dt, tolerance=1e-9)
        if step:
            close(row["measured_source"], rows[step - 1]["after_source"])
            close(row["gripper_qpos_before"], rows[step - 1]["gripper_qpos_after"])
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
                raw_rotations = decode(np.eye(4), np.column_stack((np.zeros((len(raw), 3)), raw[:, 3:])))
                # Per-delta SO(3), not cumulative orientation, is attenuated.
                for a, b in zip(raw, effective, strict=True):
                    ar = decode(np.eye(4), a[None])[0][:3, :3]
                    br = decode(np.eye(4), b[None])[0][:3, :3]
                    close(br, Rotation.from_rotvec(alpha * Rotation.from_matrix(ar).as_rotvec()).as_matrix())
                assert len(raw_rotations) == len(raw)
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
        assert row["chunk_index"] == chunk_index and row["chunk_physical_step"] == step - chunk["step"]
        assert step - chunk["step"] < config["max_observation_interval_steps"]
        if not variant["governor"]:
            assert step - chunk["step"] < count
            close(desired_tcp, effective_poses[step - chunk["step"]])
            close(row["opening_target"], openings[step - chunk["step"]])
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
        if variant["gripper"] == "intent_hold":
            grip = gripper.update(
                row["opening_target"],
                row["opening_before"],
                motor_opening=motor_width,
                previous_command=previous[-1],
                dt=dt,
            )
            mapping(row["gripper"], asdict(grip))
            close(row["command"][-1], grip.command)
        else:
            close(row["command"][-1], intent.update(row["opening_target"]))
        command = np.asarray(row["command"])
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
    admitted = rmse(tracking["admitted"])
    close(result["tcp_position_rmse_m"], admitted["position_rmse_m"])
    close(result["rotation_rmse_rad"], admitted["so3_rmse_rad"])
    return dict(
        case=str(case),
        source=source,
        checkpoint=checkpoint,
        variant=variant["name"],
        record=record["name"],
        status="verified_basic" if not full else "verified",
        steps=600,
        chunks=len(chunks),
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
        independent_governor_replay="pending" if variant["governor"] else "not_applicable",
        component_ablation=variant["model_rotation_alpha"] == 0,
        raw_checkpoint_evidence=source == "wam" and variant["model_rotation_alpha"] == 1,
    )


def verify_query(config, case, index, chunk, evidence, checkpoint):
    """Filled by the full matrix auditor before WAM results are accepted."""
    raise NotImplementedError("WAM response audit not ready; no WAM case is certified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basic-oracle", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("read-only audit requires empty CUDA and LD paths")
    output = Path(args.output).resolve()
    if not output.is_relative_to(REPO / "outputs") or output.exists():
        raise ValueError("new audit artifact must be below repository outputs; refusing overwrite")
    evidence, cases = Evidence(), []
    for config_path, source in CONFIGS:
        if args.basic_oracle and source != "oracle":
            continue
        config = evidence.json(config_path)
        evidence.check(config["fixture"], config["fixture_sha256"])
        fixtures = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
        checkpoint = None if source == "oracle" else config["checkpoints"][0]["name"]
        for variant in config["variants"]:
            for name in config["records"]:
                for seed in config["sampling_seeds"]:
                    case = (
                        Path(config["output_dir"])
                        / "simulation"
                        / f"{checkpoint or source}_{variant['name']}"
                        / f"{name}_s{seed}"
                    )
                    if not (case / "result.json").exists():
                        cases.append(
                            dict(case=str(case), source=source, status="partial" if case.exists() else "missing")
                        )
                        continue
                    result = evidence.json(case / "result.json")
                    if result["error"]:
                        cases.append(
                            dict(case=str(case), source=source, status="execution_error", error=result["error"])
                        )
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
                        cases.append(dict(case=str(case), source=source, status="audit_error", error=repr(exc)))
    evidence.read(__file__)
    evidence.recheck()
    oracle = [c for c in cases if c["source"] == "oracle"]
    basic_gate = len(oracle) == 6 and all(
        c["status"] in ("verified", "verified_basic") and c["outcome"]["stable"] for c in oracle
    )
    value = dict(
        schema_version="s2a.recovery-readonly-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="basic_oracle" if args.basic_oracle else "full",
        cases=cases,
        oracle_basic_gate_passed=basic_gate,
        independent_governor_replay_complete=False,
        source_before_after_hashes=evidence.hashes,
        warnings=[
            "Task/contact are recorded metrics only, never replay inputs.",
            "Oracle is privileged qualification, not WAM success.",
            "Smaller admitted tracking error does not prove equal nominal reference following.",
            "Full governor replay and future WAM identity audit remain pending in this basic gate.",
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

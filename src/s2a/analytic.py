"""P1 geometry and historical interface diagnostics, separate evidence."""

import numpy as np

from s2a.artifacts import write_json
from s2a.geometry import (
    error_summary,
    euler_pose_to_transform,
    integrate_local,
    inverse,
    pose9_to_transform,
    pose_error,
    transform_to_euler_pose,
)
from s2a.interface import DroidReferenceInterface


def analytic(config, arrays, out):
    frames = arrays["frame_index"]
    start, end = config["selection"]["frame_start_inclusive"], config["selection"]["frame_end_exclusive"]
    idx = np.flatnonzero((frames >= start) & (frames < end))
    tcp = pose9_to_transform(arrays["observation_state"][:, :9])
    delta = pose9_to_transform(arrays["action"][idx, :9])
    target_tcp = tcp[idx] @ delta
    integrated = integrate_local(tcp[idx[0]], arrays["action"][idx, :9])
    flange = tcp @ inverse(arrays["H_FC"])
    target_flange = target_tcp @ inverse(arrays["H_FC"])
    original_flange = euler_pose_to_transform(arrays["r1_observation_state_cartesian_position"])
    iface = DroidReferenceInterface(**config["reference_interface"]["parameters"])
    raw_command = iface.encode_target(target_flange, flange[idx], limit=False)
    command = iface.limit(raw_command)
    r1_target = euler_pose_to_transform(arrays["r1_action_cartesian_position"][idx])
    r1_velocity = arrays["r1_action_cartesian_velocity"][idx]
    # Two diagnostics: an exported observation may differ from create_action_dict.robot_state.
    reencoded = iface.encode_target(r1_target, original_flange[idx], limit=False)
    decoded = iface.decode_target(r1_velocity, original_flange[idx])
    metrics = {
        "evidence_level": "geometry_and_reference_interface_algebra_only",
        "single_step": error_summary(target_tcp, tcp[idx + 1]),
        "integrated": error_summary(integrated, tcp[np.r_[idx, idx[-1] + 1]]),
        "tcp_to_r1_flange": error_summary(flange, original_flange),
        "tcp_flange_roundtrip": error_summary(flange @ arrays["H_FC"], tcp),
        "target_vs_historical_commanded_pose": error_summary(target_flange, r1_target),
        "r1_command_decode_vs_recorded_target": error_summary(decoded, r1_target),
        "r1_command_reencode_rmse_per_axis": np.sqrt(np.mean((reencoded - r1_velocity) ** 2, axis=0)),
        "analytic_command_saturation_fraction": float(np.mean(np.any(np.abs(command - raw_command) > 1e-10, axis=1))),
        "gripper_command_max_error": float(
            np.max(np.abs(iface.gripper_position(arrays["action"][idx, 9]) - arrays["r1_action_gripper_position"][idx]))
        ),
    }
    for label, values in [("state", arrays["observation_state"][:, 3:9]), ("delta", arrays["action"][idx, 3:9])]:
        x, y = values[:, :3], values[:, 3:]
        metrics[label + "_rot6d_raw_max_defect"] = float(
            max(
                np.max(np.abs(np.linalg.norm(x, axis=1) - 1)),
                np.max(np.abs(np.linalg.norm(y, axis=1) - 1)),
                np.max(np.abs(np.sum(x * y, axis=1))),
            )
        )
    metrics["geometry_passed"] = all(
        v["position_max_m"] < 1e-5 and v["rotation_max_rad"] < 1e-4
        for k, v in metrics.items()
        if k in {"single_step", "integrated", "tcp_to_r1_flange", "tcp_flange_roundtrip"}
    )
    np.savez_compressed(
        out / "analytic_commands.npz",
        frame_index=frames[idx],
        target_tcp=target_tcp,
        target_flange=target_flange,
        cartesian_position=transform_to_euler_pose(target_flange),
        cartesian_velocity=command,
        unlimited_cartesian_velocity=raw_command,
        gripper_position=iface.gripper_position(arrays["action"][idx, 9]),
        historical_pose_error=pose_error(target_flange, r1_target),
    )
    write_json(out / "geometry_metrics.json", metrics)
    if not metrics["geometry_passed"]:
        raise ValueError("geometry gate failed; do not fit away a convention error")
    return metrics

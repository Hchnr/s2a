"""Bind a reset endpoint to the first already-generated reference."""

import numpy as np

from s2a.research.native_runtime.development_prefix_replay import ATOL, SAMPLE_FIELDS
from s2a.research.wam.training.native.native_training import gripper_cache_receipt, state_sha256


def saved_reset_endpoint(archive):
    """Expose only endpoint zero; later states and task labels are not returned."""
    if int(archive["endpoint_step"][0]) != 0:
        raise ValueError("The first saved endpoint must be step zero")
    sample = {name: np.asarray(archive[name][0]).copy() for name in SAMPLE_FIELDS}
    offsets = np.asarray(archive["gripper_current_action_offsets"][:2])
    if offsets.shape != (2,) or offsets[0] != 0 or offsets[1] not in (1, 2):
        raise ValueError("The reset must contain the original one- or two-value motor cache")
    cache = np.asarray(archive["gripper_current_action_flat"][: int(offsets[1])]).copy()
    if cache.shape != (int(offsets[1]),) or cache.dtype != np.float64 or not np.isfinite(cache).all():
        raise ValueError("The original finite float64 motor cache is required")
    sample["gripper_current_action"] = cache
    return sample


def verify_reset_endpoint(expected, expected_receipt, actual, actual_receipt, chunk, controller, tool):
    """Compare reset and reference metadata without stepping or selecting an action."""
    expected_sha = state_sha256(expected["sim_states"])
    actual_sha = state_sha256(actual["sim_states"])
    for sample, receipt, digest in ((expected, expected_receipt, expected_sha), (actual, actual_receipt, actual_sha)):
        state = np.asarray(sample["sim_states"])
        motor_receipt = gripper_cache_receipt(sample["gripper_current_action"])
        if (
            receipt["endpoint_step"] != 0
            or receipt["sim_state_sha256"] != digest
            or receipt["sim_state_dtype"] != "float64"
            or receipt["sim_state_shape"] != list(state.shape)
            or receipt["sim_time"] != float(sample["sim_time"])
            or any(receipt[key] != value for key, value in motor_receipt.items())
        ):
            raise ValueError("Reset receipt must describe its own exact state and time")
    if (
        chunk["step"] != 0
        or chunk["physics_before_sha256"] != expected_sha
        or chunk["physics_after_sha256"] != expected_sha
        or chunk["physics_unchanged"] is not True
        or chunk["physics_before_time_s"] != float(expected["sim_time"])
        or chunk["physics_after_time_s"] != float(expected["sim_time"])
        or controller["actual_initial_physics_sha256"] != expected_sha
    ):
        raise ValueError("The first generated reference must bind the recorded reset state and time")
    maximum = 0.0
    for name in (*SAMPLE_FIELDS, "gripper_current_action"):
        a, b = np.asarray(expected[name]), np.asarray(actual[name])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Finite matching reset field required: {name}")
        difference = float(np.max(np.abs(a - b)))
        if difference > ATOL:
            raise ValueError(f"Reset field differs from the recorded initial state: {name}")
        maximum = max(maximum, difference)
    if not np.array_equal(expected["gripper_current_action"], controller["aligned_initial_motor_target"]):
        raise ValueError("Saved initial motor target must match the controller reset")
    if float(expected["opening"]) != chunk["anchor_opening"]:
        raise ValueError("Reference opening anchor must match the recorded measurement")
    np.testing.assert_allclose(np.asarray(expected["source_pose"]) @ tool, chunk["anchor_tcp"], atol=ATOL, rtol=0)
    targets, openings = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["openings"])
    if (
        targets.shape != (40, 4, 4)
        or openings.shape != (40,)
        or not np.isfinite(targets).all()
        or not np.isfinite(openings).all()
    ):
        raise ValueError("A complete finite first generated reference is required")
    return {
        "recorded_state_sha256": expected_sha,
        "actual_state_sha256": actual_sha,
        "state_bytes_equal": expected_sha == actual_sha,
        "max_field_difference": maximum,
        "initial_time": float(actual["sim_time"]),
        "measured_opening": float(actual["opening"]),
        "motor_target": np.asarray(actual["gripper_current_action"]).copy(),
        "image_sha256": chunk["image_sha256"],
        "request_sha256": chunk["request_sha256"],
        "response_sha256": chunk["response_sha256"],
    }

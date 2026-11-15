"""Measured recovery-window compatibility with the existing WAM data domain."""

import hashlib

import numpy as np

from s2a.research.wam.training.clean.clean_training_data import normalize_active10, post_capture_labels


def source_identity(case, prepared, expected):
    """Bind all variants to one known training source, excluding recorded holds."""
    task, demo = expected
    if (
        case["task_index"] != task
        or case["source_name"] != f"task{task}_native_{demo}"
        or (task, demo) in {tuple(key) for key in prepared["exclusion"]["forbidden_demo_keys"]}
        or case["initial_sha256"] in set(prepared["exclusion"]["forbidden_initial_state_sha256"])
    ):
        raise ValueError("The recovery source must be an admitted non-forbidden training demo")
    matched = [
        row["source_identity"]
        for row in prepared["training_anchors"]
        if row["source_identity"]["task_index"] == task and row["source_identity"]["demo"] == demo
    ]
    if (
        not matched
        or any(row != matched[0] for row in matched)
        or matched[0]["initial_state_sha256"] != case["initial_sha256"]
    ):
        raise ValueError("The full task/demo/initial-state identity must match the original training source")
    return matched[0]


def window_indices(starts, horizon, periods):
    """Return outgoing indices while preserving overlaps and rejecting padding."""
    if (
        type(horizon) is not int
        or horizon <= 0
        or type(periods) is not int
        or not starts
        or any(type(start) is not int or start < 0 or start + horizon > periods for start in starts)
        or len(set(starts)) != len(starts)
    ):
        raise ValueError("Distinct complete actual H40 windows are required")
    rows = [list(range(start, start + horizon)) for start in starts]
    return dict(windows=rows, total=sum(map(len, rows)), unique=len({index for row in rows for index in row}))


def audit_window(data, start, tool, normalization):
    """Rebuild one H40 target from measured states and preserve command identity."""
    count = len(data["sim_states"])
    window_indices([start], 40, count - 1)
    endpoints = slice(start, start + 41)
    actions = slice(start, start + 40)
    if (
        data["sent_commands"].shape != (count - 1, 7)
        or not np.isfinite(data["sent_commands"]).all()
        or not np.array_equal(data["endpoint_step"][endpoints], np.arange(start, start + 41))
    ):
        raise ValueError("Actual 20 Hz endpoints and separate seven-dimensional commands are required")
    rebuilt = post_capture_labels(
        data["source_pose"][endpoints],
        data["opening"][endpoints],
        data["sim_time"][endpoints],
        source_from_tcp=tool,
        actual_source_site_poses=data["actual_source_site_pose"][endpoints],
        actual_eef_body_poses=data["actual_eef_body_pose"][endpoints],
        state_mask=data["state_mask"][endpoints],
        action_valid=data["action_valid"][actions],
        action_mask=data["action_mask"][actions],
    )
    expected = rebuilt["physical_action128"]
    if not np.array_equal(expected, data["action128"][actions]):
        raise ValueError("Stored Action128 differs from measured future TCP and opening")
    tcp = rebuilt["tcp_pose"]
    states = np.zeros((41, 128), dtype=np.float64)
    states[:, :3], states[:, 3:6], states[:, 6:9], states[:, 9] = (
        tcp[:, :3, 3],
        tcp[:, :3, 0],
        tcp[:, :3, 1],
        data["opening"][endpoints],
    )
    if not np.array_equal(states, data["state128"][endpoints]):
        raise ValueError("Stored state128 differs from measured TCP and opening")
    normalized = normalize_active10(expected, **normalization)
    normalized_bytes = np.ascontiguousarray(normalized["normalized_active10"]).tobytes()
    return dict(
        normalized_sha256=hashlib.sha256(normalized_bytes).hexdigest(),
        normalized_dtype=str(normalized["normalized_active10"].dtype),
        normalized_shape=list(normalized["normalized_active10"].shape),
        transform_sha256=normalized["transform_sha256"],
        clipped_count=normalized["clipped_count"],
        clipped_per_active_slot=normalized["clipped_per_active_slot"].tolist(),
        first_sim_time=float(data["sim_time"][start]),
        last_sim_time=float(data["sim_time"][start + 40]),
        observed_openings=expected[:, 9].copy(),
        actual_gripper_commands=data["sent_commands"][actions, 6].copy(),
    )

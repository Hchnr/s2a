"""Measured endpoint labels for native LIBERO commands, without simulation I/O."""

import hashlib

import numpy as np

from s2a.geometry import finite, inverse, matrix_to_rot6d, validate_transform


def native_anchor_indices(native_steps):
    """Four fixed native anchors plus the beginning of the terminal hold."""
    if isinstance(native_steps, (bool, np.bool_)) or int(native_steps) != native_steps:
        raise ValueError("native_steps must be an integer")
    n = int(native_steps)
    if not 43 <= n <= 600:
        raise ValueError("native trajectory must have 43..600 commands")
    anchors = [(j * (n - 40)) // 3 for j in range(4)] + [n]
    if len(set(anchors)) != 5 or any(a + 40 > n + 40 for a in anchors):
        raise ValueError("five distinct complete anchor windows required")
    return anchors


def native_command_schedule(native_commands):
    """Preserve all native commands, then hold the final gripper for 40 steps."""
    native = finite(native_commands)
    if native.ndim != 2 or native.shape[1] != 7:
        raise ValueError("expected native commands [N,7]")
    native_anchor_indices(len(native))
    hold = np.zeros((40, 7), dtype=np.float64)
    hold[:, -1] = native[-1, -1]
    return np.concatenate((native.copy(), hold), axis=0)


def state_sha256(raw_sim_state):
    """Hash the actual contiguous state bytes, keeping its dtype unmodified."""
    state = np.asarray(raw_sim_state)
    if state.ndim != 1 or state.dtype != np.float64 or not np.isfinite(state).all():
        raise ValueError("expected finite raw float64 flattened simulation state")
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def synchronous_labels(source_poses, openings, source_from_tcp, sim_times, *, dt=0.05):
    """Encode current TCP states and actual next-endpoint local motion labels.

    States, commands, local deltas and opening targets remain separate. The last
    endpoint has a valid measured state and no valid outgoing action.
    """
    source = validate_transform(source_poses)
    h = validate_transform(source_from_tcp)
    if source.ndim != 3 or len(source) < 2 or h.shape != (4, 4):
        raise ValueError("expected endpoint poses [T,4,4] and one tool transform")
    opening = finite(openings, (len(source),))
    times = finite(sim_times, (len(source),))
    if np.any((opening < 0) | (opening > 1)):
        raise ValueError("opening must be in [0,1]")
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0):
        raise ValueError("endpoint times must advance exactly one control period")
    tcp = validate_transform(source @ h)
    delta = inverse(tcp[:-1]) @ tcp[1:]
    state = np.zeros((len(source), 128), dtype=np.float64)
    state[:, :3] = tcp[:, :3, 3]
    state[:, 3:9] = matrix_to_rot6d(tcp[:, :3, :3])
    state[:, 9] = opening
    action = np.zeros_like(state)
    action[:-1, :3] = delta[:, :3, 3]
    action[:-1, 3:9] = matrix_to_rot6d(delta[:, :3, :3])
    action[:-1, 9] = opening[1:]
    state_mask = np.zeros_like(state, dtype=bool)
    state_mask[:, :10] = True
    action_mask = state_mask.copy()
    action_mask[-1] = False
    valid = np.ones(len(source), dtype=bool)
    valid[-1] = False
    return dict(
        tcp_pose=tcp,
        delta_pose=delta,
        state128=state,
        action128=action,
        state_mask=state_mask,
        action_mask=action_mask,
        action_valid=valid,
    )


def validate_physical_source(observed_source, actual_source_site_pose, actual_eef_body_pose):
    """LIBERO EEF observes grip-site position and EEF BODY orientation.

    This deliberately does not equate the controller site's orientation with
    the source observation's orientation. The actual single_arm sensor reads
    MuJoCo xquat directly; convert_quat only reorders its F64 components.
    """
    observed = validate_transform(observed_source)
    site = validate_transform(actual_source_site_pose)
    body = validate_transform(actual_eef_body_pose)
    if any(t.shape != (4, 4) for t in (observed, site, body)):
        raise ValueError("expected individual observed/site/body SE3 poses")
    physical = body.copy()
    physical[:3, 3] = site[:3, 3]
    if not np.allclose(observed, physical, atol=1e-10, rtol=0):
        raise ValueError("source observation differs from actual grip-site/BODY pose")
    return physical

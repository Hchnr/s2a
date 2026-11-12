"""Reconstruct geometry while auditing machine-precision quaternion normalization."""

import numpy as np

from s2a.geometry import validate_transform


def quaternion_spans(model):
    """Return qpos slices for MuJoCo free and ball joint quaternions."""
    spans = []
    if len(model.jnt_type) != model.njnt or len(model.jnt_qposadr) != model.njnt:
        raise ValueError("Joint types and qpos addresses must match the model")
    for joint in range(model.njnt):
        kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        start = address + 3 if kind == 0 else address if kind == 1 else None
        if start is not None:
            if start < 0 or start + 4 > model.nq or any(start < old + 4 and old < start + 4 for _, old in spans):
                raise ValueError("Quaternion qpos spans must be valid and disjoint")
            spans.append((joint, start))
    return spans


def reconstruct_endpoint(model, data, state, *, site_id, body_id, site_from_tcp, forward):
    """Allow only recorded, machine-precision normalization of valid quaternions."""
    state = np.asarray(state, dtype=float)
    site_from_tcp = validate_transform(np.asarray(site_from_tcp, dtype=float))
    if (
        type(site_id) is not int
        or type(body_id) is not int
        or not 0 <= site_id < model.nsite
        or not 0 <= body_id < model.nbody
        or model.na != 0
        or state.shape != (1 + model.nq + model.nv,)
        or not np.isfinite(state).all()
        or site_from_tcp.shape != (4, 4)
        or not callable(forward)
    ):
        raise ValueError("One finite saved state, valid body/site and fixed TCP transform are required")
    spans = quaternion_spans(model)
    data.time = float(state[0])
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    forward(model, data)
    restored = np.concatenate(([data.time], np.asarray(data.qpos), np.asarray(data.qvel)))
    if not np.isfinite(restored).all():
        raise ValueError("Forward geometry produced a nonfinite state")
    mask = np.zeros(len(state), dtype=bool)
    quaternions = []
    for joint, start in spans:
        indices = np.arange(start + 1, start + 5)
        mask[indices] = True
        before, after = state[indices], restored[indices]
        norm_before, norm_after = float(np.linalg.norm(before)), float(np.linalg.norm(after))
        difference = float(np.max(np.abs(after - before)))
        if difference and (
            abs(norm_before - 1) > 1e-12
            or difference > 1e-12
            or not np.allclose(after, before / norm_before, atol=2e-15, rtol=0)
        ):
            raise ValueError("Only machine-precision normalization of a unit quaternion is allowed")
        quaternions.append(
            dict(
                joint_index=joint,
                state_indices=indices.tolist(),
                norm_before=norm_before,
                norm_after=norm_after,
                max_absolute_change=difference,
            )
        )
    if not np.array_equal(state[~mask], restored[~mask]):
        raise ValueError("Time, velocity and non-quaternion positions must remain exactly unchanged")
    difference = np.abs(restored - state)
    changed = np.flatnonzero(difference)
    world_site = np.eye(4)
    world_site[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    world_site[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=float)
    world_site = validate_transform(world_site)
    tcp = (world_site @ site_from_tcp)[:3, 3]
    body = np.asarray(data.xpos[body_id], dtype=float).copy()
    if body.shape != (3,) or not np.isfinite(body).all() or not np.isfinite(tcp).all():
        raise ValueError("Finite world-frame TCP and body-origin positions are required")
    delta = tcp - body
    return dict(
        tcp_world_m=tcp.copy(),
        body_origin_world_m=body,
        tcp_minus_body_world_m=delta,
        body_origin_distance_m=float(np.linalg.norm(delta)),
        body_origin_xy_distance_m=float(np.linalg.norm(delta[:2])),
        tcp_above_body_origin_m=float(delta[2]),
        state_change=dict(
            changed_indices=changed.tolist(),
            max_absolute_change=float(difference.max(initial=0)),
            quaternion_norms=quaternions,
        ),
    )

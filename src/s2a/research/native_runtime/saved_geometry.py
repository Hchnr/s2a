"""Reconstruct world-frame TCP and body-origin positions from a saved state."""

import numpy as np

from s2a.geometry import validate_transform


def reconstruct_endpoint(model, data, state, *, site_id, body_id, site_from_tcp, forward):
    """Run one non-stepping kinematic query and verify the restored state is unchanged."""
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
    data.time = float(state[0])
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    forward(model, data)
    restored = np.concatenate(([data.time], np.asarray(data.qpos), np.asarray(data.qvel)))
    if not np.array_equal(restored, state):
        raise ValueError("A non-stepping geometry query must not change the saved state")
    world_site = np.eye(4)
    world_site[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    world_site[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=float)
    world_site = validate_transform(world_site)
    tcp = (world_site @ site_from_tcp)[:3, 3]
    body = np.asarray(data.xpos[body_id], dtype=float).copy()
    if body.shape != (3,) or not np.isfinite(body).all() or not np.isfinite(tcp).all():
        raise ValueError("Finite world-frame TCP and body-origin positions are required")
    difference = tcp - body
    return dict(
        tcp_world_m=tcp.copy(),
        body_origin_world_m=body,
        tcp_minus_body_world_m=difference,
        body_origin_distance_m=float(np.linalg.norm(difference)),
        body_origin_xy_distance_m=float(np.linalg.norm(difference[:2])),
        tcp_above_body_origin_m=float(difference[2]),
    )

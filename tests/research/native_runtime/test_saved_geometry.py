"""Saved-state geometry must preserve state and apply the registered TCP transform."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.saved_geometry import reconstruct_endpoint


def fixture():
    model = SimpleNamespace(nq=2, nv=2, na=0, nsite=1, nbody=1)
    data = SimpleNamespace(
        time=0.0,
        qpos=np.zeros(2),
        qvel=np.zeros(2),
        site_xmat=np.array([[0, -1, 0, 1, 0, 0, 0, 0, 1]], dtype=float),
        site_xpos=np.array([[1.0, 2.0, 3.0]]),
        xpos=np.array([[1.0, 3.0, 2.0]]),
    )
    site_from_tcp = np.eye(4)
    site_from_tcp[0, 3] = 0.5
    return model, data, np.array([0.4, 0.1, 0.2, 0.3, 0.4]), site_from_tcp


def test_world_frame_tcp_and_body_origin_use_the_registered_transform():
    model, data, state, transform = fixture()
    result = reconstruct_endpoint(
        model, data, state, site_id=0, body_id=0, site_from_tcp=transform, forward=lambda *_: None
    )
    np.testing.assert_allclose(result["tcp_world_m"], [1, 2.5, 3])
    np.testing.assert_allclose(result["tcp_minus_body_world_m"], [0, -0.5, 1])
    assert result["body_origin_distance_m"] == pytest.approx(np.sqrt(1.25))
    assert result["body_origin_xy_distance_m"] == pytest.approx(0.5)
    assert result["tcp_above_body_origin_m"] == pytest.approx(1.0)


def test_forward_cannot_change_or_advance_the_saved_state():
    model, data, state, transform = fixture()

    def changes_state(_model, actual):
        actual.qpos[0] += 0.01

    with pytest.raises(ValueError, match="must not change"):
        reconstruct_endpoint(model, data, state, site_id=0, body_id=0, site_from_tcp=transform, forward=changes_state)
    with pytest.raises(ValueError, match="finite saved state"):
        reconstruct_endpoint(
            model, data, state[:-1], site_id=0, body_id=0, site_from_tcp=transform, forward=lambda *_: None
        )

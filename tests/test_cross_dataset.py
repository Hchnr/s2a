import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2a.cross_dataset import align_native, geometry_checks, native_to_tcp, quaternion_pose


def test_native_wxyz_quaternion_order_is_not_scipy_xyzw_order():
    t = quaternion_pose([[1.0, 2.0, 3.0, 2**-0.5, 0.0, 0.0, 2**-0.5]])[0]
    assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    assert_allclose(t[:3, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="degenerate"):
        quaternion_pose([[0] * 7])


def test_m1_native_frame_offset_rotates_tcp_lever_and_uses_link8_contract():
    link8 = np.eye(4)
    link8[0, 3] = 0.1
    old_source = np.eye(4)
    old_source[0, 3] = 10
    h = native_to_tcp("interndata_m1", {"link8_from_canonical_tcp": link8, "source_from_tcp": old_source})
    assert_allclose(h[:3, 3], [0.1 / np.sqrt(2), 0.1 / np.sqrt(2), 0], atol=1e-15)
    assert_allclose(h[:3, :3] @ [1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], atol=1e-15)


def test_source_row_join_handles_subinterval_and_shuffled_native_rows():
    native = {"frame_index": np.array([7, 5, 6, 9]), "observation": np.array([0.7, 0.5, 0.6, 0.9])}
    aligned = align_native(native, {"row_range": [5, 8]}, np.arange(3))
    assert_array_equal(aligned["frame_index"], [5, 6, 7])
    assert_allclose(aligned["observation"], [0.5, 0.6, 0.7])
    native["frame_index"][0] = 6
    with pytest.raises(ValueError, match="duplicate"):
        align_native(native, {"row_range": [5, 8]}, np.arange(3))


def test_cumulative_geometry_resets_across_invalid_observation_and_excludes_terminal():
    frames = np.arange(5)
    observed = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
    observed[:, 0, 3] = [0, 0.1, 100, 0.2, 0.3]
    state = np.zeros((5, 128))
    state[:, 0] = observed[:, 0, 3]
    state[:, [3, 7]] = 1
    state_mask = np.ones((5, 128), dtype=bool)
    state_mask[2, :9] = False
    state[2, 3:9] = 0  # masked pose must not enter rotation decoding
    action = np.zeros((5, 128))
    action[:, 0] = 0.1
    action[:, [3, 7]] = 1
    rows = {
        "frame_index": frames,
        "observation.state": state,
        "observation.state_mask": state_mask,
        "action": action,
        "action_dim_mask": np.ones((5, 128), dtype=bool),
        "action_valid": np.ones(5, dtype=bool),
    }
    metrics, arrays = geometry_checks(rows, observed, np.eye(4))
    assert_array_equal(arrays["transition_index"], [0, 3])
    assert metrics["continuous_segments"] == 2
    assert metrics["valid_transitions"] == 2
    assert metrics["integrated"]["position_max_m"] < 1e-15
    assert metrics["native_observation"]["position_max_m"] < 1e-15
    assert_allclose(arrays["candidate_native_pose"][:, 0, 3], [0.1, 0.3])

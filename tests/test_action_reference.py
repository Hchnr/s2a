import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.geometry import inverse, transform


def test_oracle_and_wam_share_full_tcp_delta_decoder_in_original_world_frame():
    tool = transform([0.02, -0.01, 0.1], Rotation.from_euler("z", 45, degrees=True).as_matrix())
    measured = transform([0.3, -0.2, 0.5], Rotation.from_euler("z", 90, degrees=True).as_matrix())
    targets = transform(
        [[0.4, 0.2, 0.3], [0.41, 0.22, 0.31]],
        Rotation.from_euler("xyz", [[0, 10, 0], [5, 10, 4]], degrees=True).as_matrix(),
    )
    actions = encode_oracle_targets(measured @ tool, targets @ tool, [1, 0.2])
    decoded, openings = decode_action_targets(measured @ tool, actions, horizon=2)
    np.testing.assert_allclose(decoded @ inverse(tool), targets, atol=1e-14)
    np.testing.assert_array_equal(openings, [1, 0.2])
    np.testing.assert_array_equal(actions[:, 10:], 0)


def test_raw_output_is_preserved_while_opening_is_explicitly_clipped():
    actions = encode_oracle_targets(np.eye(4), np.eye(4)[None], [1])
    actions[0, 9] = 1.5
    _, opening = decode_action_targets(np.eye(4), actions, horizon=1)
    assert actions[0, 9] == 1.5 and opening[0] == 1
    actions[0, 10] = 1
    with pytest.raises(ValueError, match="active slots"):
        decode_action_targets(np.eye(4), actions, horizon=1)

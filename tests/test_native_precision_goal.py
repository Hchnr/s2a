import numpy as np
from scipy.spatial.transform import Rotation

from s2a.current_model_state_native_precision import rotation_goal_delta


def test_numeric_decoder_and_zero_command_goal_latch_are_distinct():
    current = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    previous = Rotation.from_rotvec([-0.3, 0.1, 0.2]).as_matrix()
    delta = np.array([0.12, -0.24, 0.18])
    numeric = Rotation.from_rotvec(delta).as_matrix().astype(np.float32).astype(float)
    decoded = rotation_goal_delta(delta, current, previous, matrix_decoder=lambda _: numeric)
    expected = Rotation.from_matrix((numeric @ current) @ current.T).as_rotvec()
    np.testing.assert_array_equal(decoded, expected)
    assert np.max(np.abs(decoded - delta)) > 1e-10
    sticky = rotation_goal_delta(
        np.zeros(3),
        current,
        previous,
        matrix_decoder=lambda _: (_ for _ in ()).throw(AssertionError("zero must latch")),
    )
    np.testing.assert_allclose(Rotation.from_rotvec(sticky).as_matrix(), previous @ current.T, atol=1e-14)

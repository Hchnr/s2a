import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.async_reference import AsyncReferenceBuffer, TimedReference
from s2a.geometry import matrix_to_rot6d, transform


def actions(count=8):
    result = np.zeros((count, 10))
    result[:, :3] = [0.001, 0, 0]
    result[:, 3:9] = [1, 0, 0, 0, 1, 0]
    result[:, 9] = 0.7
    return result


def test_delayed_reference_skips_elapsed_endpoints_and_keeps_original_world_anchor():
    anchor = transform([0.2, 0.3, 0.4], Rotation.from_euler("z", 90, degrees=True).as_matrix())
    a = actions()
    a[0, 3:9] = matrix_to_rot6d(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    reference = TimedReference.from_actions(0, 0, 0.05, anchor, a, horizon=8)
    buffer = AsyncReferenceBuffer(np.eye(4), 1, max_age=0.4)
    assert buffer.admit(reference, now=0.1) == "accepted"
    sampled = buffer.sample(now=0.1)
    assert sampled["reference_index"] == 2
    np.testing.assert_allclose(sampled["pose"][:3, 3], [0.198, 0.301, 0.4])
    a[:] = 100
    np.testing.assert_array_equal(sampled["pose"], buffer.sample(now=0.1)["pose"])
    sampled["pose"][:] = 0
    assert buffer.sample(now=0.1)["pose"][3, 3] == 1


def test_timeout_holds_last_fixed_goal_without_reintegrating_delta():
    buffer = AsyncReferenceBuffer(np.eye(4), 1, max_age=0.4)
    reference = TimedReference.from_actions(0, 0, 0.05, np.eye(4), actions(), horizon=8)
    assert buffer.admit(reference, now=0) == "accepted"
    last = buffer.sample(now=0.35)
    for t in (0.4, 1.0, 11.2):
        sample = buffer.sample(now=t)
        assert not sample["tracking"] and sample["reason"] == "expired"
        np.testing.assert_array_equal(sample["pose"], last["pose"])
        assert sample["opening"] == last["opening"]
    late = TimedReference.from_actions(1, 0.4, 0.05, np.eye(4), actions(), horizon=8)
    assert buffer.admit(late, now=11.2) == "expired"
    np.testing.assert_array_equal(buffer.sample(now=11.2)["pose"], last["pose"])


def test_rejected_responses_do_not_replace_active_reference_and_clocks_cannot_reverse():
    buffer = AsyncReferenceBuffer(np.eye(4), 1, max_age=0.4)
    reference = TimedReference.from_actions(2, 0.1, 0.05, np.eye(4), actions(), horizon=8)
    assert buffer.admit(reference, now=0.1) == "accepted"
    assert buffer.admit(reference, now=0.15) == "out_of_order"
    future = TimedReference.from_actions(3, 1, 0.05, np.eye(4), actions(), horizon=8)
    assert buffer.admit(future, now=0.15) == "future_observation"
    older = TimedReference.from_actions(3, 0, 0.05, np.eye(4), actions(), horizon=8)
    assert buffer.admit(older, now=0.15) == "older_observation"
    assert buffer.sample(now=0.15)["sequence"] == 2
    with pytest.raises(ValueError, match="monotonic"):
        buffer.sample(now=0.1)


def test_short_reference_exhaustion_and_episode_isolation():
    buffer = AsyncReferenceBuffer(np.eye(4), 1, max_age=0.4)
    other = AsyncReferenceBuffer(np.eye(4), 0.2, max_age=0.4)
    short = TimedReference.from_actions(0, 0, 0.05, np.eye(4), actions(), horizon=1)
    assert buffer.admit(short, now=0.05) == "exhausted"
    np.testing.assert_array_equal(buffer.sample(now=0.05)["pose"], np.eye(4))
    assert other.sample(now=0)["opening"] == 0.2
    with pytest.raises(ValueError, match="execution horizon"):
        TimedReference.from_actions(0, 0, 0.05, np.eye(4), actions(), horizon=9)
    with pytest.raises(ValueError, match="degenerate"):
        TimedReference.from_actions(0, 0, 0.05, np.eye(4), np.zeros((8, 10)), horizon=8)

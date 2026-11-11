"""Release revisions must remain causal, bounded and transactionally committed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from s2a.research.wam.tracking.preview.bounded_release_revision import (
    BoundedReleaseRevision,
    ReferenceChunk,
    ReleaseRevisionConfig,
)


def config():
    return ReleaseRevisionConfig(
        stride=8,
        bridge_steps=8,
        release_clearance=0.04,
        opening_revision_gap=0.04,
        current_position_gate_m=0.03,
        current_rotation_gate_rad=0.15,
        joint_buffer_rad=0.05,
    )


def chunk(step, opening, *, count=40, x_offset=0.0, angle=0.0):
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = (np.arange(count) + step + 1) * 0.001 + x_offset
    poses[:, :3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
    return ReferenceChunk(step, poses, np.full(count, opening))


def prepare(selector, old, latest, step, *, opening=0.25, closing=True, margin=0.1, tcp=None):
    if tcp is None:
        tcp = old.poses[step - old.step] if old is not None and step - old.step < len(old.poses) else np.eye(4)
    return selector.prepare(
        latest,
        committed=old,
        step=step,
        measured_tcp=tcp,
        measured_opening=opening,
        closing_episode=closing,
        previous_inner_margin_rad=margin,
    )


def test_conflict_keeps_aligned_pair_then_bridges_and_cannot_hold_twice():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(10, 0.2), chunk(20, 0.9, x_offset=0.01, angle=0.1)
    first = prepare(selector, old, latest, 24)
    assert first["source"] == "committed" and first["reason"] == "bounded_hold"
    assert first["absolute_first_endpoint"] == 25 and first["conflict_without_safety"]
    np.testing.assert_array_equal(first["poses"][:8], old.poses[14:22])
    np.testing.assert_array_equal(first["openings"][:8], old.openings[14:22])

    for k in range(8):
        alpha = (k + 1) / 8
        prior, new = old.poses[22 + k], latest.poses[12 + k]
        np.testing.assert_allclose(first["poses"][8 + k, :3, 3], (1 - alpha) * prior[:3, 3] + alpha * new[:3, 3])
        rotations = Rotation.from_matrix([prior[:3, :3], new[:3, :3]])
        expected = Slerp([0, 1], rotations)([alpha]).as_matrix()[0]
        np.testing.assert_allclose(first["poses"][8 + k, :3, :3], expected, atol=1e-12, rtol=0)
        assert first["openings"][8 + k] == pytest.approx((1 - alpha) * 0.2 + alpha * 0.9)

    assert selector.state()["mode"] == "idle"
    selector.commit(first, completed_steps=8)
    second = prepare(selector, old, latest, 32)
    assert second["source"] == "bridge" and second["reason"] == "bounded_bridge"
    np.testing.assert_allclose(second["poses"][:8], first["poses"][8:16], atol=1e-12, rtol=0)
    np.testing.assert_allclose(second["openings"][:8], first["openings"][8:16], atol=1e-12, rtol=0)
    selector.commit(second, completed_steps=8)
    third = prepare(selector, old, latest, 40)
    assert third["source"] == "latest" and third["reason"] == "episode_hold_already_used"
    np.testing.assert_array_equal(third["openings"], latest.openings[20:])
    selector.commit(third, completed_steps=8)
    reset_phase = prepare(selector, old, latest, 48, closing=False)
    selector.commit(reset_phase, completed_steps=8)
    assert selector.state()["mode"] == "idle"


def test_consistent_small_opening_release_is_not_blocked():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.117), chunk(8, 0.117)
    result = prepare(selector, old, latest, 8, opening=0.06)
    assert result["source"] == "latest" and not result["conflict_without_safety"]
    np.testing.assert_array_equal(result["openings"], latest.openings)


def test_unknown_inner_history_never_activates_hold_and_known_loss_stops():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    result = prepare(selector, old, latest, 8, margin=None)
    assert result["reason"] == "unknown_inner_margin" and result["conflict_without_safety"]
    assert result["source"] == "latest" and result["previous_inner_margin_rad"] is None
    selector.abort(result)
    with pytest.raises(ValueError, match="buffer loss"):
        prepare(selector, old, latest, 8, margin=0.0499)


@pytest.mark.parametrize("kind", ["position", "rotation"])
def test_measured_tcp_mismatch_refuses_old_reference(kind):
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    measured = old.poses[8].copy()
    if kind == "position":
        measured[0, 3] += 0.031
    else:
        measured[:3, :3] = Rotation.from_rotvec([0, 0, 0.151]).as_matrix()
    result = prepare(selector, old, latest, 8, tcp=measured)
    assert result["reason"] == "current_pose_mismatch" and result["source"] == "latest"


def test_insufficient_overlap_cannot_create_a_partial_hold():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2, count=20), chunk(8, 0.9)
    result = prepare(selector, old, latest, 8)
    assert result["reason"] == "insufficient_overlap" and result["source"] == "latest"
    np.testing.assert_array_equal(result["poses"], latest.poses)


def test_bridge_uses_a_freshly_generated_reference_after_the_completed_hold():
    selector = BoundedReleaseRevision(config())
    old, first_latest = chunk(0, 0.2), chunk(8, 0.9, x_offset=0.01)
    first = prepare(selector, old, first_latest, 8)
    selector.commit(first, completed_steps=8)
    refreshed = chunk(16, 0.7, x_offset=0.02, angle=0.08)
    bridge = prepare(selector, old, refreshed, 16)
    assert bridge["source"] == "bridge" and bridge["latest_source_step"] == 16
    assert bridge["openings"][0] == pytest.approx(0.2 * 7 / 8 + 0.7 / 8)
    np.testing.assert_array_equal(bridge["poses"][7], refreshed.poses[7])
    np.testing.assert_array_equal(bridge["openings"][7:], refreshed.openings[7:])


def test_bridge_falls_back_to_latest_if_actual_inner_history_becomes_unknown():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    first = prepare(selector, old, latest, 8)
    selector.commit(first, completed_steps=8)
    second = prepare(selector, old, latest, 16, margin=None)
    assert second["reason"] == "unknown_inner_margin" and second["source"] == "latest"
    selector.commit(second, completed_steps=8)
    third = prepare(selector, old, latest, 24)
    assert third["reason"] == "episode_hold_already_used" and third["source"] == "latest"


def test_closing_end_releases_to_latest_without_waiting_for_a_bridge():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    first = prepare(selector, old, latest, 8)
    selector.commit(first, completed_steps=8)
    result = prepare(selector, old, latest, 16, closing=False)
    assert result["source"] == "latest" and result["reason"] == "not_closing"
    selector.commit(result, completed_steps=8)
    assert selector.state()["mode"] == "idle"


def test_abort_and_commit_do_not_publish_unexecuted_or_mutated_state():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    result = prepare(selector, old, latest, 8)
    with pytest.raises(ValueError, match="full stride"):
        selector.commit(result, completed_steps=7)
    result["openings"][0] = 0.7
    with pytest.raises(ValueError, match="was changed"):
        selector.commit(result, completed_steps=8)
    selector.abort(result, completed_steps=0)
    assert selector.state()["mode"] == "idle" and not selector.state()["faulted"]
    result = prepare(selector, old, latest, 8)
    selector.abort(result, completed_steps=3)
    assert selector.state()["faulted"]
    with pytest.raises(ValueError, match="partially executed"):
        prepare(selector, old, latest, 8)
    selector.reset()
    assert prepare(selector, old, latest, 8)["reason"] == "bounded_hold"


def test_reference_times_and_same_generation_identity_cannot_be_forged():
    selector = BoundedReleaseRevision(config())
    old, latest = chunk(0, 0.2), chunk(8, 0.9)
    with pytest.raises(ValueError, match="before it was generated"):
        prepare(selector, old, latest, 7)
    first = prepare(selector, old, latest, 8, closing=False)
    selector.commit(first, completed_steps=8)
    with pytest.raises(ValueError, match="cannot change"):
        prepare(selector, latest, chunk(8, 0.8), 16)
    with pytest.raises(ValueError, match="Consecutive"):
        prepare(selector, old, latest, 17)
    with pytest.raises(ValueError, match="cannot go backwards"):
        prepare(selector, None, chunk(4, 0.9), 16)
    with pytest.raises(ValueError, match="previously selected"):
        prepare(selector, old, chunk(16, 0.9), 16)


def test_reference_chunk_copies_inputs_and_requires_valid_openings():
    poses = np.tile(np.eye(4), (4, 1, 1))
    openings = np.full(4, 0.5)
    reference = ReferenceChunk(0, poses, openings)
    poses[0, 0, 3], openings[0] = 4, 0
    assert reference.poses[0, 0, 3] == 0 and reference.openings[0] == 0.5
    with pytest.raises(ValueError, match="paired"):
        ReferenceChunk(0, poses, [0.1, 0.2, 0.3, 1.1])

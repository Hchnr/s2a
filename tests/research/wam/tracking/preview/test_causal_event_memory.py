"""A remembered phase must remain causal, aligned and episode-local."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.wam.tracking.preview.causal_event_memory import CausalEventMemory, EventMemoryConfig


def config():
    return EventMemoryConfig(
        horizon=40,
        stride=8,
        closing_below=0.35,
        opening_above=0.8,
        reference_position_m=0.02,
        reference_rotation_rad=0.1,
        current_position_m=0.03,
        current_rotation_rad=0.15,
    )


def references(opening=1.0):
    return np.tile(np.eye(4), (40, 1, 1)), np.full(40, opening)


def select(memory, poses, openings, step, measured_opening=0.9, measured_tcp=None):
    return memory.prepare(
        poses,
        openings,
        step=step,
        measured_tcp=np.eye(4) if measured_tcp is None else measured_tcp,
        measured_opening=measured_opening,
    )


def arm_closure(memory, index=20):
    poses, openings = references()
    openings[index:] = 0.1
    decision = select(memory, poses, openings, 0)
    assert decision["source"] == "latest" and decision["armed_new_event"]
    assert memory.state()["pending_event"] is None
    memory.commit(decision, completed_steps=8)
    return poses, openings


def test_event_survives_two_replans_and_uses_aligned_old_pose_opening_pair():
    memory = CausalEventMemory(config())
    old_poses, old_openings = arm_closure(memory)
    latest_poses, latest_openings = references()
    one = select(memory, latest_poses, latest_openings, 8)
    assert one["reason"] == "event_retained" and one["source"] == "latest"
    memory.commit(one, completed_steps=8)
    two = select(memory, latest_poses, latest_openings, 16)
    assert two["reason"] == "event_due" and two["source"] == "remembered"
    assert two["event"]["absolute_endpoint"] == 21 and two["source_indices"] == list(range(16, 24))
    np.testing.assert_array_equal(two["poses"], old_poses[16:24])
    np.testing.assert_array_equal(two["openings"], old_openings[16:24])
    memory.commit(two, completed_steps=8)
    assert memory.state()["pending_event"] is None


def test_opening_event_uses_the_same_causal_rules():
    memory = CausalEventMemory(config())
    poses, openings = references(0.1)
    openings[12:] = 0.9
    first = select(memory, poses, openings, 0, measured_opening=0.2)
    memory.commit(first, completed_steps=8)
    new_poses, new_openings = references(0.1)
    second = select(memory, new_poses, new_openings, 8, measured_opening=0.2)
    assert second["source"] == "remembered" and second["event"]["direction"] == "opening"
    np.testing.assert_array_equal(second["openings"], openings[8:16])


def test_latest_covers_original_event_and_prevents_reuse_of_older_reference():
    memory = CausalEventMemory(config())
    arm_closure(memory)
    poses, openings = references()
    openings[12:] = 0.1
    decision = select(memory, poses, openings, 8)
    assert decision["reason"] == "latest_covers_event" and decision["source"] == "latest"
    assert decision["armed_new_event"]
    memory.commit(decision, completed_steps=8)
    assert memory.state()["pending_event"]["source_step"] == 8


@pytest.mark.parametrize("mismatch", ["event_position", "prefix_rotation", "current_position", "current_rotation"])
def test_pose_disagreement_drops_the_remembered_reference(mismatch):
    memory = CausalEventMemory(config())
    arm_closure(memory)
    poses, openings = references()
    current = np.eye(4)
    if mismatch == "event_position":
        poses[12, 0, 3] = 0.021
    elif mismatch == "prefix_rotation":
        poses[3, :3, :3] = Rotation.from_rotvec([0, 0, 0.11]).as_matrix()
    elif mismatch == "current_position":
        current[0, 3] = 0.031
    else:
        current[:3, :3] = Rotation.from_rotvec([0, 0, 0.16]).as_matrix()
    decision = select(memory, poses, openings, 8, measured_tcp=current)
    assert decision["source"] == "latest" and decision["reason"] == "pose_agreement_rejected"
    memory.commit(decision, completed_steps=8)
    assert memory.state()["pending_event"] is None


def test_measured_phase_change_discards_old_event():
    memory = CausalEventMemory(config())
    arm_closure(memory)
    poses, openings = references()
    decision = select(memory, poses, openings, 8, measured_opening=0.2)
    assert decision["reason"] == "measured_phase_changed" and decision["source"] == "latest"
    memory.commit(decision, completed_steps=8)
    assert memory.state()["pending_event"] is None


def test_only_completed_stride_commits_memory_and_tampering_is_rejected():
    memory = CausalEventMemory(config())
    poses, openings = references()
    openings[20:] = 0.1
    decision = select(memory, poses, openings, 0)
    with pytest.raises(ValueError, match="committed or aborted"):
        select(memory, poses, openings, 0)
    with pytest.raises(ValueError, match="full stride"):
        memory.commit(decision, completed_steps=7)
    with pytest.raises(ValueError, match="current staged"):
        memory.commit(copy.deepcopy(decision), completed_steps=8)
    decision["openings"][0] = 0.5
    with pytest.raises(ValueError, match="selection was changed"):
        memory.commit(decision, completed_steps=8)
    memory.abort(decision)
    assert memory.state()["pending_event"] is None and memory.state()["last_completed_replan_step"] is None


def test_partial_abort_requires_reset_and_reset_isolates_episodes():
    memory = CausalEventMemory(config())
    arm_closure(memory)
    poses, openings = references()
    decision = select(memory, poses, openings, 8)
    memory.abort(decision, completed_steps=2)
    assert memory.state()["faulted"] and memory.state()["pending_event"] is None
    with pytest.raises(ValueError, match="explicit reset"):
        select(memory, poses, openings, 16)
    memory.reset()
    decision = select(memory, poses, openings, 0)
    assert decision["source"] == "latest" and decision["event"] is None


def test_abort_without_execution_preserves_committed_memory():
    memory = CausalEventMemory(config())
    arm_closure(memory)
    before = memory.state()
    poses, openings = references()
    first = select(memory, poses, openings, 8)
    memory.abort(first)
    assert memory.state() == before
    second = select(memory, poses, openings, 8)
    assert second["reason"] == "event_retained"
    memory.commit(second, completed_steps=8)


def test_altered_decision_metadata_cannot_be_committed():
    memory = CausalEventMemory(config())
    poses, openings = references()
    openings[20:] = 0.1
    decision = select(memory, poses, openings, 0)
    decision["event"]["absolute_endpoint"] = 999
    with pytest.raises(ValueError, match="selection was changed"):
        memory.commit(decision, completed_steps=8)
    memory.abort(decision)


def test_wrong_step_and_invalid_input_do_not_change_completed_history():
    memory = CausalEventMemory(config())
    arm_closure(memory)
    before = memory.state()
    poses, openings = references()
    with pytest.raises(ValueError, match="Consecutive"):
        select(memory, poses, openings, 16)
    openings[0] = float("nan")
    with pytest.raises(ValueError):
        select(memory, poses, openings, 8)
    assert memory.state() == before


def test_no_gripper_event_produces_no_memory_or_intervention():
    memory = CausalEventMemory(config())
    poses, openings = references()
    for step in (0, 8, 16, 24, 32, 40):
        decision = select(memory, poses, openings, step)
        assert decision["source"] == "latest" and decision["event"] is None
        memory.commit(decision, completed_steps=8)
    assert memory.state()["pending_event"] is None


def test_old_event_at_end_of_horizon_is_consumed_before_expiry():
    memory = CausalEventMemory(config())
    arm_closure(memory, index=39)
    poses, openings = references()
    for step in (8, 16, 24):
        decision = select(memory, poses, openings, step)
        assert decision["reason"] == "event_retained"
        memory.commit(decision, completed_steps=8)
    final = select(memory, poses, openings, 32)
    assert final["source"] == "remembered" and final["event"]["absolute_endpoint"] == 40
    memory.commit(final, completed_steps=8)
    assert memory.state()["pending_event"] is None

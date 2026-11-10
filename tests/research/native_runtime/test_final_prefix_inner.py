"""The final recorded prefix period must expose actual inner buffer loss."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.final_prefix_inner import FinalPrefixInner


def fixture():
    model = SimpleNamespace(jnt_range=np.tile([-1.0, 1.0], (7, 1)), joint_name2id=lambda name: int(name[-1]) - 1)
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), time=0.0)
    robot = SimpleNamespace(_ref_joint_pos_indexes=list(range(7)), _ref_joint_vel_indexes=list(range(7)))
    return SimpleNamespace(robots=[robot], sim=SimpleNamespace(model=model, data=data))


def test_only_the_last_prefix_period_is_captured_and_buffer_loss_is_recorded():
    env, recorder = fixture(), FinalPrefixInner(3)
    recorder.observe(env, 1)
    assert recorder.samples == []
    for index in range(25):
        env.sim.data.time = (index + 1) * 0.002
        env.sim.data.qpos[0] = 0.96 if index == 5 else 0.0
        recorder.observe(env, 2)
    result = recorder.finish()
    assert result["prefix_step"] == 2 and result["below_buffer_samples"] == 1 and result["outside_samples"] == 0
    assert result["minimum_margin_rad"] == pytest.approx(0.04)
    env.sim.data.qpos[0] = 0.0
    assert result["samples"][5]["q"][0] == 0.96
    with pytest.raises(ValueError, match="cannot exceed"):
        recorder.observe(env, 2)


def test_incomplete_period_and_changed_limits_are_rejected():
    env, recorder = fixture(), FinalPrefixInner(1)
    recorder.observe(env, 0)
    with pytest.raises(ValueError, match="Exactly 25"):
        recorder.finish()
    env.sim.model.jnt_range[0, 1] = 2.0
    with pytest.raises(ValueError, match="limits must remain"):
        recorder.observe(env, 0)

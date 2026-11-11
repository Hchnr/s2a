"""The replay recorder must observe native work without changing its outcome."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tracking.response_models import replay_contact_torque as replay


class FakeState:
    def __init__(self, data):
        self.data = data

    def flatten(self):
        return np.r_[self.data.time, self.data.qpos, self.data.qvel]


class FakeSimulation:
    def __init__(self):
        self.data = SimpleNamespace(
            time=0.0,
            qpos=np.zeros(7),
            qvel=np.zeros(7),
            ctrl=np.array([1.0, 0, 0, 0, 0, 0, 0]),
            qfrc_constraint=np.arange(7.0),
            qfrc_actuator=np.arange(7.0) * 2,
            qfrc_bias=np.arange(7.0) * 3,
        )
        self.calls = 0

    def get_state(self):
        return FakeState(self.data)

    def step(self):
        self.calls += 1
        self.data.time += 0.002
        self.data.qpos += 0.001


def fake_env():
    sim = FakeSimulation()
    robot = SimpleNamespace(
        controller=SimpleNamespace(torques=np.array([2.0, 0, 0, 0, 0, 0, 0])),
        torques=np.array([1.0, 0, 0, 0, 0, 0, 0]),
        torque_limits=(-np.ones(7), np.ones(7)),
        _ref_joint_actuator_indexes=list(range(7)),
        _ref_joint_vel_indexes=list(range(7)),
    )
    return SimpleNamespace(sim=sim, robots=[robot])


def test_recorder_observes_the_actual_step_and_enforces_the_hard_cap():
    env = fake_env()
    recorder = replay.ReplayRecorder(env, 1, contacts=lambda _sim: [])
    for _ in range(25):
        recorder.step()
    assert env.sim.calls == 25 and len(recorder.samples) == 25
    assert recorder.samples[0]["step"] == 80 and recorder.samples[-1]["inner_index"] == 24
    np.testing.assert_array_equal(recorder.samples[0]["controller_raw_torque_nm"], [2, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(recorder.samples[0]["commanded_torque_nm"], env.robots[0].torques)
    before = env.sim.get_state().flatten().copy()
    with pytest.raises(ValueError, match="inner-step budget"):
        recorder.step()
    np.testing.assert_array_equal(before, env.sim.get_state().flatten())
    assert env.sim.calls == 25


def test_mismatched_actuator_control_is_rejected_before_integration():
    env = fake_env()
    env.sim.data.ctrl[0] = 0.5
    recorder = replay.ReplayRecorder(env, 1, contacts=lambda _sim: [])
    before = env.sim.get_state().flatten().copy()
    with pytest.raises(AssertionError):
        recorder.step()
    np.testing.assert_array_equal(before, env.sim.get_state().flatten())
    assert env.sim.calls == len(recorder.samples) == 0


def test_instrumentation_restores_step_and_preserves_partial_receipt_on_error(monkeypatch):
    env = fake_env()
    original = env.sim.step
    real_recorder = replay.ReplayRecorder
    monkeypatch.setattr(
        replay, "ReplayRecorder", lambda env, periods: real_recorder(env, periods, contacts=lambda _: [])
    )
    receipt = {}

    def callback(env, *_args, **_kwargs):
        env.sim.step()
        raise RuntimeError("explicit failure")

    with pytest.raises(RuntimeError, match="explicit failure"):
        replay.instrumented_continuation(env, None, None, receipt, callback, 1)
    assert env.sim.step == original and env.sim.calls == 1
    assert len(receipt["replay_dynamics"]["samples"]) == 1


def test_contact_reader_classifies_robot_subtree_without_advancing_state():
    import mujoco

    xml = """
    <mujoco><worldbody>
      <geom name="table" type="box" pos="0 0 -0.04" size="0.2 0.2 0.05"/>
      <body name="robot0_base"><body name="robot0_link1" pos="0 0 0.03">
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom name="robot_collision" type="box" size="0.04 0.04 0.04" mass="1"/>
      </body></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
    before = np.r_[data.time, data.qpos.copy(), data.qvel.copy()]
    contacts = replay.contact_reader(sim)
    np.testing.assert_array_equal(before, np.r_[data.time, data.qpos, data.qvel])
    assert contacts and all(contact["kind"] == "robot_environment" for contact in contacts)
    assert {"table", "robot_collision"} == {contacts[0]["geom1"], contacts[0]["geom2"]}
    assert all(len(contact["wrench_contact_frame"]) == 6 for contact in contacts)


def test_runtime_enforces_the_registered_native_budget(monkeypatch):
    from s2a.research.native_runtime import guarded_stream_prefix

    captured = {}

    class Runtime:
        def _run(self, *_args, **kwargs):
            captured.update(kwargs)
            return "bounded"

    monkeypatch.setattr(guarded_stream_prefix, "GuardedStreamPrefixRuntime", Runtime)
    keys = ("runtime_config", "record", "record_xml", "commands", "saved", "receipts", "asset_refs", "controller")
    assert replay.bounded_prefix(dict.fromkeys(keys, object()), Path("outputs/example"), None, periods=46) == "bounded"
    assert captured["continuation_budget"] == 46 and captured["continuation_limit"] == 48
    assert captured["capture_final_prefix_inner"] is True and captured["registered_seed"] == 42


def test_replay_process_does_not_import_candidate_solvers():
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.tracking.response_models import replay_contact_torque as r; r.isolation()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr

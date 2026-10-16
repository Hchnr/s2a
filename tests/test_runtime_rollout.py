import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from s2a.artifacts import REPO
from s2a.control import PIDConfig
from s2a.geometry import advance, pose_error
from s2a.identification import LinearResponse
from s2a.interface import DroidReferenceInterface
from s2a.rollout import simulate
from s2a.runtime import StateToActionAdapter
from s2a.trajectory import ReferenceTrajectory


def test_runtime_chunk_is_anchored_once_gripper_separate_and_dt_command_planned():
    adapter = StateToActionAdapter(np.eye(4), dt_command=0.05, pid_config=PIDConfig(kp=(1.0,) * 6, ki=(0.2,) * 6))
    deltas = np.tile([0.01, 0, 0, 1, 0, 0, 0, 1, 0, 0.3], (3, 1))
    adapter.set_chunk(np.eye(4), deltas, np.array([0.0, 0.1, 0.2, 0.3]))
    first = adapter.update(np.eye(4), 0.0)
    second = adapter.update(np.eye(4), 0.1)
    assert second["dt_feedback"] == 0.1
    assert second["dt_command"] == 0.05
    assert second["gripper_action_space"] == "position"
    assert second["command"][-1] == 0.7
    assert_allclose(second["reference_pose_flange"][:3, 3], [0.01, 0, 0])
    assert second["error"][0] == 0.01
    integral = adapter.pid.state.integral.copy()
    adapter.set_chunk(np.eye(4), deltas, np.array([0.1, 0.2, 0.3, 0.4]))
    assert_allclose(adapter.pid.state.integral, integral)
    assert_allclose(adapter.update(np.eye(4), 0.1)["command"], second["command"])
    assert first["command"].shape == (7,)
    adapter.reset()
    with pytest.raises(ValueError, match="reference"):
        adapter.update(np.eye(4), 0)


def test_software_closed_loop_advances_states_and_pi_rejects_constant_disturbance():
    settings = json.loads((REPO / "configs/droid_episode0.json").read_text())["rollout"]
    times = np.arange(101) * 0.07
    reference = ReferenceTrajectory(np.array([0.0, times[-1]]), np.array([np.eye(4), np.eye(4)]))
    model = LinearResponse(
        np.diag([0.075 / 0.07] * 3 + [0.15 / 0.07] * 3),
        np.empty((6, 0)),
        np.zeros(6),
        0,
        0,
        "diagonal",
        0.1,
        np.ones(6),
        {},
    )
    metrics = {}
    for method, gains in {
        "FF": {"kp": (0.0,) * 6, "ki": (0.0,) * 6, "kd": (0.0,) * 6},
        "P": {"kp": (2.0,) * 6, "ki": (0.0,) * 6, "kd": (0.0,) * 6},
        "PI": {"kp": (2.0,) * 6, "ki": (1.0,) * 6, "kd": (0.0,) * 6},
        "PD": {"kp": (2.0,) * 6, "ki": (0.0,) * 6, "kd": (0.02,) * 6},
        "PID": {"kp": (2.0,) * 6, "ki": (1.0,) * 6, "kd": (0.02,) * 6},
    }.items():
        value, trace = simulate(
            model,
            DroidReferenceInterface(),
            reference,
            times,
            gains=gains,
            dt_command=0.07,
            inverse_lambda=0.1,
            feedforward_kind="analytic",
            scenario="stationary",
            settings=settings,
            seed=42,
        )
        assert value["valid"]
        assert np.any(np.linalg.norm(np.diff(trace["state"][:, :3, 3], axis=0), axis=1) > 0)
        metrics[method] = value
        # The trace contains the true state advanced by the previous candidate's command.
        assert_allclose(pose_error(trace["state"][1], trace["state"][0]) / 0.07, trace["velocity"][0], atol=1e-14)
    assert metrics["P"]["balanced_tracking_error"] < 0.5 * metrics["FF"]["balanced_tracking_error"]
    assert metrics["PI"]["steady_position_m"] < 0.25 * metrics["P"]["steady_position_m"]
    assert metrics["PID"]["steady_position_m"] < 0.25 * metrics["PD"]["steady_position_m"]


def test_noise_scenario_is_seeded_and_uses_no_future_observation():
    settings = json.loads((REPO / "configs/droid_episode0.json").read_text())["rollout"]
    times = np.arange(13) * 0.07
    ref = ReferenceTrajectory(
        np.array([0.0, times[-1]]), np.array([np.eye(4), advance(np.eye(4), [0.01, 0, 0, 0, 0, 0], 1)])
    )
    model = LinearResponse(np.eye(6), np.empty((6, 0)), np.zeros(6), 0, 0, "diagonal", 0.1, np.ones(6), {})
    kwargs = dict(
        gains={"kp": (1.0,) * 6, "ki": (0.0,) * 6, "kd": (0.0,) * 6},
        dt_command=0.07,
        inverse_lambda=0.1,
        feedforward_kind="analytic",
        scenario="observation_noise",
        settings=settings,
        seed=42,
    )
    a, ta = simulate(model, DroidReferenceInterface(), ref, times, **kwargs)
    b, tb = simulate(model, DroidReferenceInterface(), ref, times, **kwargs)
    assert a == b
    assert_allclose(ta["command"], tb["command"], atol=0)

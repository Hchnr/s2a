import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
from loaded_velocity_independent_replay import LoadedVelocityReplay

from s2a.loaded_velocity_control import LoadedVelocityControl


def setup():
    parameters = dict(
        velocity_units="normalized_opening_per_20Hz_period",
        prior="constant_current_velocity",
        mean=[0.0] * 4,
        scale=[1.0] * 4,
        coefficients=np.zeros((15, 2)),
    )
    controller = LoadedVelocityControl(parameters, 0.9, velocity_envelope=0.3)
    replay = LoadedVelocityReplay(parameters, 0.9, asdict(controller.latch.config), 0.3)
    return controller, replay


def result(controller, reference, motor, previous):
    poses = np.repeat(np.eye(4)[None], len(reference), axis=0)
    value = controller.update(0.38, motor, 0.0, poses, reference, 0, np.eye(4), previous_command=previous, dt=0.05)
    saved = json.loads(json.dumps(asdict(value)))
    return saved, poses


def test_scalar_replay_matches_loading_hold_and_confirmed_release():
    controller, replay = setup()
    previous = 0.0
    trials = [
        ([0.38] * 8, [-0.25, 0.25]),
        ([0.38] * 8, [-0.5, 0.5]),
        ([0.38] * 8, [-0.75, 0.75]),
        ([0.38] * 8, [-1.0, 1.0]),
        ([0.38, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], [-1.0, 1.0]),
        ([0.38, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], [-1.0, 1.0]),
    ]
    routes = []
    for reference, motor in trials:
        saved, poses = result(controller, reference, motor, previous)
        proof = replay.step(saved, 0.38, motor, 0.0, poses, reference, 0, np.eye(4), previous, 0.05)
        routes.append(proof["route"])
        previous = saved["command"]
    assert routes[:-1] == ["native_close_load"] * 5 and routes[-1] == "model_tracking"


@pytest.mark.parametrize("field", ["command", "preview", "prediction"])
def test_independent_replay_rejects_corrupted_control_evidence(field):
    controller, replay = setup()
    reference = [0.38] * 8
    motor = [-0.25, 0.25]
    saved, poses = result(controller, reference, motor, 0.0)
    tampered = copy.deepcopy(saved)
    if field == "command":
        tampered["command"] = 0.0
    elif field == "preview":
        tampered["preview"]["guards"]["position_span"] = False
    else:
        tampered["executed_response"]["next_opening"] += 0.01
    with pytest.raises(AssertionError):
        replay.step(tampered, 0.38, motor, 0.0, poses, reference, 0, np.eye(4), 0.0, 0.05)

import numpy as np
import pytest

from s2a.loaded_velocity_control import LoadedVelocityControl


def parameters():
    return dict(
        velocity_units="normalized_opening_per_20Hz_period",
        prior="constant_current_velocity",
        mean=[0.0] * 4,
        scale=[1.0] * 4,
        coefficients=np.zeros((15, 2)),
    )


def invoke(controller, reference, motor=(-0.25, 0.25), previous=0.0, poses=None, velocity=0.0):
    if poses is None:
        poses = np.repeat(np.eye(4)[None], len(reference), axis=0)
    return controller.update(0.38, motor, velocity, poses, reference, 0, np.eye(4), previous_command=previous, dt=0.05)


def loaded():
    controller = LoadedVelocityControl(parameters(), 0.9, velocity_envelope=0.3)
    result = invoke(controller, [0.38] * 8)
    assert result.route == "native_close_load" and result.command == 1
    assert result.plan is None and result.latch["closing_episode"]
    assert result.executed_response["next_motor_target"] == pytest.approx(-0.5)
    return controller


def test_closing_intent_completes_load_instead_of_only_matching_aperture():
    controller = loaded()
    result = invoke(controller, [0.38] * 8, motor=(-1.0, 1.0), previous=1.0)
    assert result.command == 0 and result.route == "native_close_load"
    assert result.executed_response["next_motor_target"] == -1.0


def test_loading_state_survives_generated_chunk_boundary_but_is_not_shared():
    controller = loaded()
    assert invoke(controller, [0.38] * 8, motor=(-0.5, 0.5), previous=1.0).command == 1
    fresh = LoadedVelocityControl(parameters(), 0.38, velocity_envelope=0.3)
    result = invoke(fresh, [0.38] * 8)
    assert result.route == "model_tracking" and result.plan is not None
    assert not result.latch["closing_episode"]


def test_release_requires_confirmed_coupled_rise_and_tracks_original_reference():
    controller = loaded()
    reference = [0.38, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    first = invoke(controller, reference, motor=(-1.0, 1.0))
    assert first.preview["advanced"] and first.route == "native_close_load"
    second = invoke(controller, reference, motor=(-1.0, 1.0))
    assert second.route == "model_tracking" and not second.latch["closing_episode"]
    assert second.plan["reference_endpoints"] == tuple(reference)
    assert second.plan["reference_endpoints"][0] == 0.38 and second.preview["reference"] == 0.65
    assert second.command == second.plan["command"]


def test_pose_span_guard_blocks_release_even_with_future_opening_rise():
    controller = loaded()
    reference = [0.38, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[6, 0, 3] = 0.02
    for _ in range(3):
        result = invoke(controller, reference, motor=(-1.0, 1.0), poses=poses)
        assert not result.preview["advanced"] and not result.preview["guards"]["position_span"]
        assert result.route == "native_close_load" and result.command == 0


def test_tracking_model_failure_cannot_silently_enter_native_loading():
    controller = LoadedVelocityControl(parameters(), 0.38, velocity_envelope=0.3)
    with pytest.raises(ValueError, match="outside fit support"):
        invoke(controller, [0.38] * 8, velocity=0.4)


def test_load_route_reports_unmodeled_velocity_instead_of_claiming_supported_mpc():
    controller = loaded()
    result = invoke(controller, [0.38] * 8, velocity=0.4)
    assert result.route == "native_close_load" and result.plan is None
    assert not result.executed_prediction_in_velocity_envelope
    assert result.executed_response["next_period_velocity"] == 0.4


def test_nonfinite_actual_command_prediction_is_an_error():
    p = parameters()
    p["coefficients"][0, 0] = np.finfo(float).max
    p["coefficients"][-1, 0] = np.finfo(float).max
    controller = LoadedVelocityControl(p, 0.9, velocity_envelope=0.3)
    with pytest.raises(ValueError, match="Nonfinite"):
        invoke(controller, [0.38] * 8)

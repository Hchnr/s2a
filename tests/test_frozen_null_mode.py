"""Mass-weighted uncontrolled mode agrees across different task commands."""

import numpy as np

from s2a.frozen_null_mode import forecast_null_modes
from s2a.joint_limit_response import build_joint_response, predict_joint_response


def test_mass_weighted_mode_matches_full_joint_response_independently_of_task_command():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = a.T @ a + np.eye(7)
    j = rng.normal(size=(6, 7))
    q, initial = rng.normal(size=(2, 7))
    velocity = rng.normal(size=7) * 0.2
    mode = forecast_null_modes(mass, j, q, velocity, initial)
    model = build_joint_response(mass, j, np.ones(6) * 100, np.ones(6) * 20, q, initial, null_kp=10.0, dt=0.002)
    np.testing.assert_allclose(mode["task_forcing_residual"], np.zeros((1, 6)), atol=1e-12, rtol=0)
    for command in rng.normal(size=(4, 6)) * 0.1:
        result = predict_joint_response(model, velocity, command, q - 10, q + 10, substeps=25, enforce_limits=False)
        np.testing.assert_allclose(mode["rows"] @ result["joint_position"], mode["next"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(mode["rows"] @ result["joint_velocity"], mode["next_velocity"], atol=1e-11, rtol=0)

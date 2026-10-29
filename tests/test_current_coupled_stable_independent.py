"""Closed-form response and the actual ill-conditioned verification failure."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from current_coupled_precision_independent import response as inverse_response
from current_coupled_stable_independent import response


def test_uncoupled_critically_damped_response_matches_closed_form():
    result = response(np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.full(6, 100.0), np.full(6, 20.0), np.eye(3))
    t = 0.05
    decay = np.exp(-10 * t)
    expected_f = np.vstack((t * decay * np.eye(6), (1 - 10 * t) * decay * np.eye(6)))
    expected_g = np.vstack(((1 - (1 + 10 * t) * decay) * np.eye(6), 100 * t * decay * np.eye(6)))
    np.testing.assert_allclose(result["initial_velocity_response"], expected_f, atol=1e-11, rtol=0)
    np.testing.assert_allclose(result["goal_forcing_response"], expected_g, atol=1e-11, rtol=0)


def verify_historical_fixture():
    assert sys.version_info[:2] == (3, 11) and np.__version__ == "1.26.4"
    fixture = json.loads((Path(__file__).parent / "fixtures/coupled_near_singular_response.json").read_text())
    d = fixture["inputs"]
    args = [d[k] for k in ("mass", "jacobian", "kp", "kd", "relative_rotation_goal")]
    result = response(*args)
    for key, expected in fixture["expected"].items():
        np.testing.assert_allclose(result[key], expected, atol=1e-8, rtol=0)
    old = inverse_response(*args)
    assert np.max(abs(old["goal_forcing_response"] - fixture["expected"]["goal_forcing_response"])) > 1e-8


def test_actual_near_singular_response_passes_original_tolerance_without_explicit_inverse():
    if sys.version_info[:2] == (3, 11) and np.__version__ == "1.26.4":
        verify_historical_fixture()
        return
    repo = Path(__file__).resolve().parents[1]
    native = repo / "outputs/libero_env311/bin/python"
    assert native.exists(), "Historical evidence regression requires the documented native interpreter"
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONPATH=":".join(str(repo / part) for part in ("src", "scripts", "tests")),
    )
    result = subprocess.run(
        [
            str(native),
            "-c",
            "from test_current_coupled_stable_independent import verify_historical_fixture; verify_historical_fixture()",
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

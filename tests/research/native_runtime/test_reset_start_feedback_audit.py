"""Check the reported endpoint metrics and independent replay safeguards."""

import copy

import numpy as np
import pytest

from scripts.verification.response_models import audit_reset_start_feedback as audit


def test_metrics_use_spatial_error_norms_and_count_endpoints():
    rows = [
        dict(error=[0.003, 0.004, 0, 0.03, 0.04, 0], opening_error=-0.2, task_success=False, target_grasp=True),
        dict(error=[0, 0, 0, 0, 0, 0], opening_error=0.1, task_success=True, target_grasp=False),
    ]
    result = audit.metrics(rows)
    assert result["position_rmse_m"] == pytest.approx(0.005 / np.sqrt(2))
    assert result["rotation_rmse_rad"] == pytest.approx(0.05 / np.sqrt(2))
    assert result["opening_mae"] == pytest.approx(0.15)
    assert result["pose_cap_endpoints"] == 2
    assert result["target_grasp_endpoints"] == result["task_success_endpoints"] == 1
    rows[0]["error"][0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        audit.metrics(rows)


def test_replay_rejects_an_inner_sample_change_even_when_endpoints_match():
    score = dict(opening=0.5, opening_reference=0.5, opening_error=0, qpos=[0, 0], qvel=[0, 0], motor_target=[0, 0])
    sample = dict(time=0.0, q=[0] * 7, qvel=[0] * 7)
    periods = [
        dict(
            step=i,
            command=[0] * 7,
            actual_state=[0] * 79,
            tcp_error=[0] * 6,
            minimum_inner_margin_rad=0.1,
            pose_caps_satisfied=True,
            task_success=False,
            target_grasp=False,
            inner_outside_samples=0,
            inner_below_buffer_samples=0,
            endpoint_receipt={"endpoint_step": i + 1},
            actual_inner=[copy.deepcopy(sample) for _ in range(25)],
            completed_feedback={"score": copy.deepcopy(score)},
        )
        for i in range(8)
    ]
    replay = copy.deepcopy(periods)
    assert all(value == 0 for value in audit.compare_replay(periods, replay).values())
    replay[3]["actual_inner"][12]["qvel"][2] = 1e-5
    with pytest.raises(ValueError, match="inner_qvel"):
        audit.compare_replay(periods, replay)
    replay = copy.deepcopy(periods)
    replay[2]["actual_inner"].pop()
    with pytest.raises(ValueError, match="25 inner"):
        audit.compare_replay(periods, replay)

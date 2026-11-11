"""The failure audit must count completed state records rather than owner success."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime.audit_reset_multitask_live_failure import summarize_periods


def periods():
    result = []
    for step in range(8, 16):
        state = np.zeros(79, dtype=float)
        state[0] = step
        result.append(
            dict(
                step=step,
                actual_inner=[{}] * 25,
                actual_state=state,
                endpoint_receipt=dict(endpoint_step=step + 1, sim_state_sha256=state_sha256(state)),
                tcp_error=[0.001, 0, 0, 0.01, 0, 0],
                pose_caps_satisfied=True,
                minimum_inner_margin_rad=0.5,
                target_grasp=step == 10,
                task_success=False,
            )
        )
    return result


def test_audit_counts_observed_periods_and_rejects_changed_state_or_inner_count():
    rows = periods()
    result = summarize_periods(rows, 8)
    assert result["periods"] == 8 and result["inner_steps"] == 200
    assert result["target_grasp_periods"] == 1 and result["task_success_periods"] == 0
    assert result["position_rmse_m"] == pytest.approx(0.001)
    changed = copy.deepcopy(rows)
    changed[1]["actual_state"][0] += 1
    with pytest.raises(ValueError, match="state and all 25"):
        summarize_periods(changed, 8)
    changed = copy.deepcopy(rows)
    changed[2]["actual_inner"].pop()
    with pytest.raises(ValueError, match="state and all 25"):
        summarize_periods(changed, 8)

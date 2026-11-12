"""Independent audit must reject altered physical and gripper observations."""

import copy

import pytest

from scripts.simulation.runtime.audit_reset_multitask_live_recovery import compare_periods


def period():
    score = dict(
        opening=0.7, opening_reference=0.8, opening_error=-0.1, qpos=[0.01, -0.01], qvel=[0, 0], motor_target=[1, -1]
    )
    return dict(
        step=8,
        endpoint_receipt=dict(endpoint_step=9, sim_state_sha256="a" * 64),
        actual_inner=[dict(time=i * 0.002, q=[0] * 7, qvel=[0] * 7) for i in range(25)],
        pose_caps_satisfied=True,
        target_grasp=False,
        task_success=False,
        inner_outside_samples=0,
        inner_below_buffer_samples=0,
        command=[0] * 7,
        actual_state=[0] * 79,
        tcp_error=[0] * 6,
        completed_feedback=dict(score=score),
    )


@pytest.mark.parametrize("change", ["command", "actual_state", "inner", "gripper", "task_success"])
def test_audit_rejects_a_changed_dispatch_or_observation(change):
    original = period()
    altered = copy.deepcopy(original)
    if change in ("command", "actual_state"):
        altered[change][0] = 0.1
    elif change == "inner":
        altered["actual_inner"][8]["qvel"][0] = 0.1
    elif change == "gripper":
        altered["completed_feedback"]["score"]["motor_target"][0] = 0.1
    else:
        altered["task_success"] = True
    assert all(value == 0 for value in compare_periods([original], [original]).values())
    with pytest.raises(ValueError):
        compare_periods([original], [altered])

"""The diagnosis must preserve real temporal and eligibility distinctions."""

import copy
import json

import pytest

from scripts.verification.trajectory_search import audit_seed_policy_first_difference as audit


def period(command=0.0, plan=0.0):
    return dict(command=[command] * 7, decision=dict(refined_plan=[[plan] * 6 for _ in range(8)]))


def test_plan_can_diverge_before_any_dispatched_command():
    left = [period(), period(), period()]
    right = [period(), period(plan=0.1), period(command=0.2, plan=0.1)]
    assert audit.first_difference(left, right, key="refined_plan") == 1
    assert audit.first_difference(left, right, key="command") == 2
    assert audit.first_difference(left, left, key="command") is None
    with pytest.raises(ValueError):
        audit.first_difference(left, [dict(command=[float("nan")] * 7)], key="command")


def test_threshold_ordering_does_not_claim_a_policy_outcome():
    def case(outcome, value):
        return dict(
            outcome=outcome,
            diagnostic=dict(
                fast_early_selected=True,
                bounded_seed_blocked=True,
                bounded_seed_in_eligible_cache=True,
                seed_slack_agrees=True,
                seed_future_slack=value,
            ),
        )

    rows = [case("fast_only", x) for x in (0.31, 0.36)] + [case("bounded_only", x) for x in (0.33, 0.47)]
    result = audit.threshold_observation(rows, "seed_future_slack")
    assert result["available"] is True
    assert result["larger_values_select_bounded_separable"] is False
    assert result["smaller_values_select_bounded_separable"] is False
    rows[-1]["diagnostic"]["seed_slack_agrees"] = False
    assert audit.threshold_observation(rows, "seed_future_slack")["available"] is False


def test_fixed_inputs_and_zero_new_execution_budget_cannot_change():
    config = json.loads(audit.CONFIG.read_text())
    audit.validate_config(config)
    for key, value in (("physics_steps", 1), ("gpu_calls", 1), ("cases", 17)):
        with pytest.raises(ValueError):
            audit.validate_config(dict(config, **{key: value}))
    modified = copy.deepcopy(config)
    modified["fast_cohort_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit.validate_config(modified)


def test_recorded_source_rosters_and_resets_are_bound():
    config, _identity, audits = audit.inputs()
    assert config["cases"] == len(audits[0]["cases"]) == len(audits[1]["cases"]) == 18


def test_different_actual_state_before_action_divergence_is_rejected():
    def recorded(step, command):
        row = period(command=command)
        row.update(
            step=step,
            endpoint_receipt={},
            actual_state=[0.0],
            tcp_error=[0.0] * 6,
            actual_inner=[dict(q=[0.0], qvel=[0.0], time=0.0) for _ in range(25)],
            pose_caps_satisfied=True,
            target_grasp=False,
            task_success=False,
            inner_outside_samples=0,
            inner_below_buffer_samples=0,
            completed_feedback=dict(
                score={
                    key: 0.0
                    for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target")
                }
            ),
        )
        return row

    def branch(rows):
        return dict(runtime=dict(continuation=dict(segments=[dict(continuation=dict(periods=rows))])))

    first, second = [recorded(0, 0), recorded(1, 0)], [recorded(0, 0), recorded(1, 1)]
    second[0]["actual_state"] = [0.001]
    with pytest.raises(ValueError, match="differ in state"):
        audit.analyze_case(branch(first), branch(second), True, False)

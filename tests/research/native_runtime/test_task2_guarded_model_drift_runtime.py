"""Static drift decomposition must preserve the frozen-model continuation."""

import copy

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from scripts.simulation.runtime import prepare_task2_guarded_model_drift as binding
from scripts.tracking.trajectory_search import audit_task2_guarded_model_drift as runner


def test_reanchoring_at_the_model_prediction_introduces_no_drift():
    mass = np.diag(np.arange(1, 8, dtype=float))
    jacobian = np.column_stack((np.eye(6), np.ones(6) * 0.05))
    kp, kd, q = np.ones(6) * 80, np.ones(6) * 15, np.linspace(-0.2, 0.2, 7)
    initial, velocity = np.zeros(7), np.linspace(-0.02, 0.02, 7)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6) * 0.1, np.ones(6) * 0.1)
    plan = np.array([[0.1, -0.2, 0.1, 0.1, 0.1, 0.1], [0.15, -0.15, 0.12, 0.1, 0.12, 0.1]])
    lower, upper, site = -np.ones(7) * 3, np.ones(7) * 3, np.eye(4)
    model = build_joint_preview(mass, jacobian, kp, kd, q, initial, null_kp=10.0, dt=0.002)
    rollout = rollout_joint_preview(
        model, velocity, np.asarray([interface.decode(row) for row in plan]), site, lower, upper, enforce_limits=False
    )
    q1, v1, site1 = rollout["joint_position"][0], rollout["joint_velocity"][0], rollout["pose"][0]
    next_model = build_joint_preview(mass, jacobian, kp, kd, q1, initial, null_kp=10.0, dt=0.002)
    next_rollout = rollout_joint_preview(
        next_model, v1, np.asarray([interface.decode(plan[1])]), site1, lower, upper, enforce_limits=False
    )
    old = dict(mass=mass, jacobian=jacobian, contract=dict(kp=kp, kd=kd), rollout=rollout)
    current = dict(
        q=q1,
        velocity=v1,
        site=site1,
        limits=np.column_stack((lower, upper)),
        interface=interface,
        plan=plan[1:],
        rollout=next_rollout,
    )
    stats = runner.new_stats()
    result = runner.decompose_next_forecast(old, current, initial, np.eye(4), np.eye(4), stats)
    np.testing.assert_allclose(result["total_difference"], 0, atol=1e-11, rtol=0)
    for value in result["components"].values():
        np.testing.assert_allclose(value, 0, atol=1e-11, rtol=0)
    assert stats["counterfactual_builds_completed"] == 1 and stats["counterfactual_rollouts_completed"] == 2


def test_case_budget_rejects_extra_model_response():
    cases = []
    for spec in binding.CASES:
        n, stats = spec["completed_periods"], runner.new_stats()
        stats.update(
            model_initializations_attempted=1,
            model_initializations_completed=1,
            forward_queries_attempted=2 * n,
            forward_queries_completed=2 * n,
            baseline_rollouts=2 * n,
            counterfactual_builds_attempted=n,
            counterfactual_builds_completed=n,
            counterfactual_rollouts_attempted=2 * n,
            counterfactual_rollouts_completed=2 * n,
            independent_first_responses_charged=2 * n,
            independent_first_responses_completed=2 * n,
            independent_transition_steps_charged=50 * n,
            independent_transition_steps_completed=50 * n,
            independent_scalar_models_built=2 * n,
        )
        cases.append(dict(case=spec, stats=stats, data=dict(transitions=[{}] * n)))
    config = binding.read_config()[1]
    assert runner.summarize(cases, config)["independent_transition_steps_completed"] == 250
    changed = copy.deepcopy(cases)
    changed[0]["stats"]["independent_first_responses_charged"] += 1
    with pytest.raises(ValueError, match="costs"):
        runner.summarize(changed, config)

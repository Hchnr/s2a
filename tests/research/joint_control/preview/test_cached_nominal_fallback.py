"""A cached nominal plan requires the original two-point independent audit."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.joint_control.preview import cached_nominal_fallback as candidate
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import two_step_viability as original
from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def attach(adapter, seed, previous):
    history = AdaptivePlanHistory()
    history.completed_periods = 1
    history.last_plan = np.vstack((previous, seed[:7]))
    history.stage_plan(seed)
    adapter.attach_history(history)
    return history


def nominal_only_inputs():
    args, audit_args, targets = solve_and_audit_inputs()
    args = list(args)
    model, velocity, seed, _previous, _interface, site, _tool, _targets, _twists, lower, upper = args
    peak = seed.copy()
    peak[:, 0] = 0.25
    rollout = rollout_joint_preview(model, velocity, peak, site, lower, upper, enforce_limits=False)
    targets[:] = rollout["pose"]
    targets[0, 0, 3] += 0.0097
    args[2] = peak
    return args, audit_args


def test_exhausted_margin_search_reuses_nominal_cache_and_independently_audits():
    args, audit_args = nominal_only_inputs()
    receipt, events = {}, []
    adapter = candidate.CachedNominalFallbackAudit(receipt)
    attach(adapter, args[2], args[3])
    adapter.enabled = True
    frozen = {key: original.solve_two_step_viability.__globals__[key] for key in guarded._FROZEN_GLOBALS}
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    assert witness["witness_found"] and witness["selection_scope"] == "cached_nominal_two_step"
    assert events.count("attempted") == events.count("completed") == witness["response_evaluations"] == 320
    assert all(original.solve_two_step_viability.__globals__[key] is value for key, value in frozen.items())
    trace = receipt["cached_nominal_traces"][-1]
    assert trace["baseline_status"] == "shared_search_ineligible"
    assert trace["nominal_prefix_evaluations"] > 0 and trace["selected"] and trace["extra_model_responses"] == 0
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert checked["selection_scope"] == "cached_nominal_two_step" and checked["model_margin_enforced"] is False
    assert receipt["shared_shift_traces"][-1]["audit_status"] == "passed"


def test_disabled_candidate_keeps_original_selection_and_budget():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    baseline = SharedShiftFallbackAudit({})
    receipt = {}
    disabled = candidate.CachedNominalFallbackAudit(receipt)
    attach(baseline, args[2], args[3])
    attach(disabled, args[2], args[3])
    left = baseline.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    right = disabled.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert right["selection_scope"] == left["selection_scope"]
    assert right["response_evaluations"] == left["response_evaluations"]
    np.testing.assert_array_equal(right["planned_commands"], left["planned_commands"])
    assert receipt["cached_nominal_traces"] == []


def test_enabled_candidate_does_not_replace_a_successful_baseline():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = candidate.CachedNominalFallbackAudit(receipt)
    attach(adapter, args[2], args[3])
    adapter.enabled = True
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert witness["selection_scope"] == "margin_two_step"
    assert receipt["cached_nominal_traces"][-1]["baseline_witness_found"] is True
    assert receipt["cached_nominal_traces"][-1]["selected"] is False
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["model_margin_enforced"] is True


def test_no_nominal_cache_candidate_still_stops_without_a_witness():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 10.0
    receipt = {}
    adapter = candidate.CachedNominalFallbackAudit(receipt)
    attach(adapter, args[2], args[3])
    adapter.enabled = True
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    trace = receipt["cached_nominal_traces"][-1]
    assert result["witness_found"] is False and adapter._pending is None
    assert trace["nominal_prefix_evaluations"] == 0 and trace["candidate_available"] is False
    assert trace["selected"] is False and trace["extra_model_responses"] == 0


def test_nominal_candidate_still_fails_closed_when_independent_audit_fails(monkeypatch):
    args, audit_args = nominal_only_inputs()
    receipt = {}
    adapter = candidate.CachedNominalFallbackAudit(receipt)
    attach(adapter, args[2], args[3])
    adapter.enabled = True
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    monkeypatch.setattr(
        candidate, "audit_two_step_witness", lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("audit"))
    )
    with pytest.raises(ValueError, match="audit"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert adapter._pending is None and receipt["shared_shift_traces"][-1]["audit_status"] == "rejected"


def test_activation_requires_current_completed_idle_history():
    adapter = candidate.CachedNominalFallbackAudit({})
    owner = object()
    history = AdaptivePlanHistory()
    history.completed_periods = 143
    adapter.attach_history(history)
    inner = SimpleNamespace(
        _owner=owner,
        snapshot=lambda: {"completed_periods": 7, "last_completed_step": 142, "future_guarantee": False},
    )
    refiner = SimpleNamespace(history=inner, solver=adapter.solve)
    tracking = SimpleNamespace(_refiner=refiner, phase="ready", history=history)
    arm = SimpleNamespace(pending=None, coupled=SimpleNamespace(pending=None, completed_periods=143), tracking=tracking)
    gripper = SimpleNamespace(state=lambda: dict(staged=False, faulted=False))
    policy = SimpleNamespace(
        arm_policy=arm,
        inner_history=inner,
        gripper=gripper,
        start_step=136,
        max_steps=8,
        pending=None,
        last_inner_loss=False,
    )
    receipt = {"control_steps_completed": 142}
    with pytest.raises(ValueError, match="completed and idle"):
        adapter.activate(policy, owner, receipt, step=143)
    receipt["control_steps_completed"] = 143
    adapter.activate(policy, owner, receipt, step=143)
    assert adapter.enabled is True and receipt["cached_nominal_activation"]["future_observations"] == 0
    with pytest.raises(ValueError, match="completed and idle"):
        adapter.activate(policy, owner, receipt, step=143)

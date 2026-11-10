"""The quality candidate may dispatch only its independently checked plan."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.preview import quality_feedback_refiner as quality
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan


def fixture(monkeypatch, *, solver=solve_interior_plan, auditor=quality.independently_check_witness):
    contract = {
        "input_min": [-1.0] * 6,
        "input_max": [1.0] * 6,
        "output_min": [-1.0] * 6,
        "output_max": [1.0] * 6,
    }
    q = np.zeros(7)
    jacobian = np.c_[np.eye(6), np.zeros(6)]
    dynamics = {
        "mass": np.eye(7),
        "jacobian": jacobian,
        "robot_joint_position": q,
        "robot_joint_velocity": np.zeros(7),
        "kp": np.full(6, 100.0),
        "kd": np.full(6, 20.0),
    }
    model = SimpleNamespace(jnt_range=np.tile([-1.0, 1.0], (7, 1)), joint_name2id=lambda name: int(name[-1]) - 1)
    controller = SimpleNamespace(initial_joint=q, goal_ori=np.eye(3))
    env = SimpleNamespace(sim=SimpleNamespace(model=model), robots=[SimpleNamespace(controller=controller)])
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))

    def baseline(*args, **_kwargs):
        args[-1].stage_plan(seed)
        return {"planned_commands": seed.copy(), "evaluations": 3, "optimization": None}

    monkeypatch.setattr(
        quality,
        "rotation_goal_from_command",
        lambda command, interface, current, previous: (
            Rotation.from_rotvec(interface.decode(command)[3:]).as_matrix() @ current,
            np.eye(3),
        ),
    )
    history, receipt = CausalInnerError(), {}
    refiner = quality.QualityFeedbackRefiner(
        history, receipt, start_step=10, start_cursor=0, baseline=baseline, solver=solver, auditor=auditor
    )
    transaction = TrackingPreviewTransaction(refiner=refiner)
    poses = np.tile(np.eye(4), (9, 1, 1))
    arguments = (env, dynamics, np.eye(4), np.zeros(6), poses, np.eye(4), 0, seed[:4], np.zeros(6), contract)
    return transaction, history, receipt, arguments


def test_no_witness_clears_unexecuted_plan_and_preserves_actual_history(monkeypatch):
    def solver(*_args, **_kwargs):
        return {"witness_found": False, "status": "no_witness", "response_evaluations": 5}

    transaction, history, receipt, arguments = fixture(monkeypatch, solver=solver)
    with pytest.raises(quality.QualitySelectionStopped) as caught:
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert caught.value.decision()["stop"]["reason"] == "quality_no_witness"
    assert transaction.phase == "faulted" and transaction.history.pending_plan is None
    assert transaction.history.last_plan is None and history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["baseline_response_evaluations"] == 3
    assert receipt["quality_refiner"]["quality_response_evaluations"] == 5


def test_failed_independent_audit_stops_before_a_candidate_is_staged(monkeypatch):
    def solver(*_args, **_kwargs):
        return {"witness_found": True, "status": "model_witness", "response_evaluations": 1}

    def reject(*_args, **_kwargs):
        raise ValueError("tampered witness")

    transaction, history, receipt, arguments = fixture(monkeypatch, solver=solver, auditor=reject)
    with pytest.raises(quality.QualitySelectionStopped) as caught:
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert caught.value.reason == "quality_witness_rejected" and "tampered" in caught.value.detail
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["witness_audits_attempted"] == 1
    assert receipt["quality_refiner"]["endpoint_queries_charged"] == 200


def test_checked_plan_is_committed_then_only_completed_samples_tighten_next_solve(monkeypatch):
    captured = []

    def solver(*args, **kwargs):
        captured.append(kwargs["past_inner_error_rad"].copy())
        if len(captured) == 1:
            return solve_interior_plan(*args, **kwargs)
        return {"witness_found": False, "status": "no_witness", "response_evaluations": 1}

    transaction, history, receipt, arguments = fixture(monkeypatch, solver=solver)
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert result["route"] == "quality_witness" and result["quality"]["verification"]["passed"]
    np.testing.assert_array_equal(transaction.history.pending_plan, result["planned_commands"])
    assert transaction.history.last_plan is None and history.snapshot()["completed_periods"] == 0
    transaction.commit(result["command"])
    env, dynamics, site, _, poses, anchor, _, seed, _, contract = arguments
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    model = build_joint_preview(
        dynamics["mass"],
        dynamics["jacobian"],
        dynamics["kp"],
        dynamics["kd"],
        dynamics["robot_joint_position"],
        env.robots[0].controller.initial_joint,
        null_kp=10,
        dt=0.002,
    )
    nominal = rollout_joint_preview(
        model, np.zeros(7), [interface.decode(result["command"])], site, -np.ones(7), np.ones(7), enforce_limits=False
    )
    history.complete(10, nominal["inner_joint_position"] + 0.001)
    transaction.observe(site, np.zeros(6))
    np.testing.assert_array_equal(transaction.history.last_plan, result["planned_commands"])
    with pytest.raises(quality.QualitySelectionStopped):
        transaction.prepare(
            env,
            dynamics,
            site,
            np.zeros(6),
            poses,
            anchor,
            1,
            seed,
            result["command"],
            contract,
            site_from_tcp=np.eye(4),
        )
    np.testing.assert_array_equal(captured[0], np.zeros((25, 7)))
    np.testing.assert_allclose(captured[1], np.full((25, 7), 0.001), atol=1e-14, rtol=0)
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1

"""The opt-in solver preserves hard gates and its total response budget."""

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.research.trajectory_search import quality_rate_repair_solver as solver
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


def fixture():
    plan = np.zeros((8, 6))
    plan[0, 0] = 0.250000000006
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    poses = np.tile(np.eye(4), (8, 1, 1))
    rollout = {
        "pose": poses.copy(),
        "inner_joint_position": np.zeros((200, 7)),
        "joint_position": np.zeros((8, 7)),
        "joint_velocity": np.zeros((8, 7)),
    }
    result = {
        "witness_found": False,
        "status": "nonlinear_search_no_witness",
        "response_evaluations": 58,
        "optimizer_success": True,
        "terminal_normals": np.empty((0, 7)),
    }
    args = (
        object(),
        np.zeros(7),
        np.zeros((8, 6)),
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        poses,
        np.zeros((8, 6)),
        -np.ones(7),
        np.ones(7),
    )
    return args, result, {"commands": plan, "rollout": rollout}


def test_original_witness_passes_through_without_an_extra_response(monkeypatch):
    args, result, final = fixture()
    result["witness_found"] = True
    monkeypatch.setattr(solver, "_original_with_final_cache", lambda *_args, **_kwargs: (result, final))
    monkeypatch.setattr(
        solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: pytest.fail("extra response")
    )
    assert solver.solve_interior_plan(*args) is result


def test_small_rate_only_violation_is_repaired_and_fully_rechecked(monkeypatch):
    args, result, final = fixture()
    monkeypatch.setattr(solver, "_original_with_final_cache", lambda *_args, **_kwargs: (result, final))
    monkeypatch.setattr(solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: final["rollout"])
    repaired = solver.solve_interior_plan(*args)
    assert repaired["witness_found"] is True
    assert repaired["status"] == "model_witness_rate_repaired"
    assert repaired["response_evaluations"] == 59
    assert repaired["rate_repair"]["requires_independent_audit"] is True
    assert repaired["rate_repair"]["maximum_command_change"] < 1e-8
    assert repaired["rate_repair"]["maximum_rate_after_repair"] < 0.25
    assert repaired["rate_repair"]["failed_after_repair"] == []


@pytest.mark.parametrize("failure", ["budget", "pose", "joint", "large_rate"])
def test_unqualified_or_unbudgeted_candidate_never_uses_a_repair_response(monkeypatch, failure):
    args, result, final = fixture()
    args = list(args)
    if failure == "budget":
        result["response_evaluations"] = 160
    elif failure == "pose":
        args[7] = args[7].copy()
        args[7][0, 0, 3] = 0.02
    elif failure == "joint":
        final["rollout"]["inner_joint_position"][0, 0] = 0.96
    else:
        final["commands"][0, 0] = 0.3
    monkeypatch.setattr(solver, "_original_with_final_cache", lambda *_args, **_kwargs: (result, final))
    monkeypatch.setattr(
        solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: pytest.fail("extra response")
    )
    outcome = solver.solve_interior_plan(*args)
    assert outcome["witness_found"] is False
    assert outcome["response_evaluations"] == result["response_evaluations"]


def test_a_recomputed_pose_failure_still_stops_and_charges_the_response(monkeypatch):
    args, result, final = fixture()
    changed = dict(final["rollout"])
    changed["pose"] = changed["pose"].copy()
    changed["pose"][0, 0, 3] = 0.02
    monkeypatch.setattr(solver, "_original_with_final_cache", lambda *_args, **_kwargs: (result, final))
    monkeypatch.setattr(solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: changed)
    outcome = solver.solve_interior_plan(*args)
    assert outcome["witness_found"] is False
    assert outcome["response_evaluations"] == 59
    assert outcome["rate_repair"]["failed_after_repair"] == ["first_position"]


def test_original_solver_cache_is_observed_without_mutating_its_namespace():
    original_minimize = solver.quality.minimize
    result, captured = solver._original_with_final_cache(*inputs())
    assert solver.quality.minimize is original_minimize
    assert 0 < result["response_evaluations"] <= 160
    if result["optimizer_success"]:
        assert captured["commands"].shape == (8, 6)
        assert captured["rollout"]["inner_joint_position"].shape == (200, 7)

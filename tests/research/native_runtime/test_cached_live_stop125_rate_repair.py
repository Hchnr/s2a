"""Only the registered cached candidate may receive an independent audit."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.tracking.trajectory_search import repair_cached_live_stop125_rate as repair
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base


def test_actual_candidate_is_bound_and_a_changed_selection_is_rejected():
    stopped = admission.load_inputs()["stop_inputs"]
    prior = prefix.read(
        {"path": str(repair.ROOT / repair.FIXED["cache_audit"]), "sha256": repair.FIXED["cache_audit_sha256"]}
    )
    selected = repair.select_candidate(prior, stopped)
    assert selected["commands"].shape == (8, 6)
    assert selected["terminal_normals"].shape == (0, 7)
    changed = copy.deepcopy(prior)
    changed["analysis"]["candidates"]["cache_audit"]["optimizer_final_candidate_index"] = 56
    with pytest.raises(ValueError, match="final candidate"):
        repair.select_candidate(changed, stopped)


def test_independent_audit_controls_witness_status(monkeypatch):
    plan = np.zeros((8, 6))
    plan[0, 0] = 0.250000000006
    inputs = {"candidate": {"commands": plan, "terminal_normals": np.empty((0, 7))}}
    context = {
        "interface": SimpleInterface(),
        "model": object(),
        "velocity": np.zeros(7),
        "site": np.eye(4),
        "site_from_tcp": np.eye(4),
        "limits": np.tile([-2.0, 2.0], (7, 1)),
        "mass": np.eye(7),
        "jacobian": np.zeros((6, 7)),
        "contract": {"kp": np.ones(6), "kd": np.ones(6)},
        "q": np.zeros(7),
        "initial_joint": np.zeros(7),
    }
    stopped = {"previous": np.zeros(7), "past_inner_error_rad": np.zeros((25, 7))}
    monkeypatch.setattr(
        repair, "rollout_joint_preview", lambda *_args, **_kwargs: {"pose": np.tile(np.eye(4), (8, 1, 1))}
    )

    def approved(witness, *_args, **kwargs):
        assert kwargs["enforce_pose_caps"] is True
        assert np.max(np.abs(np.diff(np.vstack((np.zeros(6), witness["planned_commands"])), axis=0))) < 0.25
        return {
            "passed": True,
            "first_pose_caps_satisfied": True,
            "first_pose_caps_enforced": True,
            "independent_endpoint_queries": 200,
            "independent_transition_steps": 2600,
        }

    def rejected(*_args, **_kwargs):
        raise ValueError("pose limit failed")

    for auditor, expected in ((approved, True), (rejected, False)):
        stats = base.new_stats()
        stats.update(preview_responses_attempted=0, preview_responses_completed=0)
        results = {}
        repair.repair_search(inputs)(
            context,
            stopped,
            np.tile(np.eye(4), (40, 1, 1)),
            np.eye(4),
            stats,
            quality_solver=None,
            joint_solver=None,
            auditor=auditor,
            results=results,
        )
        assert results["repair"]["certified_model_witness"] is expected
        assert stats["preview_responses_completed"] == 1
        assert stats["independent_endpoint_queries_charged"] == 200


class SimpleInterface:
    input_min = -np.ones(6)
    input_max = np.ones(6)

    @staticmethod
    def decode(command):
        return command

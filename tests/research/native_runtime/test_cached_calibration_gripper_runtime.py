"""Paired references, post-step scoring, and fixed physical budgets."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from scripts.simulation.runtime import run_cached_calibration_gripper as runner


def example():
    anchor = transform(np.array([0.1, 0.2, 0.3]), Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix())
    poses = np.repeat(anchor[None], 40, axis=0)
    poses[:, 0, 3] += np.arange(1, 41) * 0.001
    chunk = dict(anchor_tcp=anchor, anchor_opening=0.8, effective_targets_tcp=poses, openings=np.full(40, 0.5))
    candidate = dict(pose_coefficients=np.full((7, 6), 0.02), opening_coefficients=np.array([-0.1, 0, 0, 0, 0, 0, 0]))
    config = dict(
        pi_margin_rad=0.0001,
        position_scale_m=0.1,
        rotation_scale_rad=0.5,
        position_cap_m=0.05,
        rotation_cap_rad=0.25,
        opening_cap=0.25,
    )
    return chunk, candidate, config


def test_independent_pair_matches_and_keeps_original_unchanged():
    chunk, candidate, config = example()
    original = chunk["effective_targets_tcp"].copy()
    produced = runner.paired_references(chunk, candidate, config, independent=False)
    checked = runner.paired_references(chunk, candidate, config, independent=True)
    for mode in ("raw", "calibrated"):
        for key in ("poses", "openings"):
            np.testing.assert_allclose(produced[mode][key], checked[mode][key], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(chunk["effective_targets_tcp"], original)
    np.testing.assert_array_equal(produced["raw"]["poses"], original)
    assert produced["calibrated"]["openings"][0] == pytest.approx(0.4)


def test_scores_both_targets_against_the_same_completed_state():
    chunk, candidate, config = example()
    references = runner.paired_references(chunk, candidate, config, independent=True)
    actual = references["raw"]["poses"][0]
    scores = runner.score_both(references, actual, 0.45, 0)
    np.testing.assert_allclose(scores["raw"]["tcp_error"], np.zeros(6), atol=1e-14)
    assert scores["raw"]["opening_error"] == pytest.approx(-0.05)
    assert scores["calibrated"]["opening_error"] == pytest.approx(0.05)
    assert np.linalg.norm(scores["calibrated"]["tcp_error"]) > 0


def test_phase_rejects_budget_overrun_and_checker_new_actions():
    config = dict(max_control_steps_per_phase=176, max_native_steps_per_phase=4400, max_new_commands_per_phase=48)

    def branch(index, mode):
        runtime = dict(
            cleanup_complete=True,
            control_steps_completed=20,
            newly_selected_commands=8,
            counters={"mj_step": {"completed": 500}, "model_construct": {"completed": 1}, "context": {"completed": 1}},
        )
        return dict(complete=True, passed=True, case={"case_index": index}, branch=mode, runtime=runtime)

    branches = [branch(*identity) for identity in runner.BRANCHES]
    assert runner.summarize_phase(branches, "producer", config)["newly_selected_commands"] == 48
    with pytest.raises(ValueError, match="budget"):
        runner.summarize_phase(branches, "checker", config)
    branches[0]["runtime"]["control_steps_completed"] = 1000
    with pytest.raises(ValueError, match="budget"):
        runner.summarize_phase(branches, "producer", config)

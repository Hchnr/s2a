"""Synthetic fixed-grid, endpoint, and full-constraint scoring contracts."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface
from s2a.saved_direction_backtracking import build_batch, score_backtracking_response


def batch_inputs():
    diagnostic, producers = dict(cases={}), {}
    for index, name in enumerate(("prefix", "early", "late")):
        base = np.full((2, 6), 0.01 * index)
        base[0, 0] = -0.0
        trial = base + np.linspace(0.03, 0.09, 12).reshape(2, 6)
        trial[0, 0] = 0.0
        errors = np.zeros((2, 6))
        errors[:, 0] = [0.005, 0.02]
        trace = [dict(commands=base, tcp_errors=errors), dict(commands=trial, tcp_errors=errors.copy())]
        pair = dict(main_subset=True, base_index=0, trial_index=1, prefix_before=1, qbase=[[0.25, 0], [4, 0]])
        diagnostic["cases"][name] = dict(horizon=2, target_index=1, pairs=[dict(main_subset=False), pair])
        producers[name] = dict(answer=dict(horizon=2, evaluated_candidates=trace))
    return diagnostic, producers


def scoring_inputs():
    desired = np.tile(np.eye(4), (2, 1, 1))
    desired[:, 0, 3] = [0.002, 0.012]
    velocity = np.zeros((2, 7))
    velocity[-1, 6] = -0.01
    return dict(
        rollout=dict(
            pose=np.tile(np.eye(4), (2, 1, 1)),
            inner_joint_position=np.zeros((50, 7)),
            joint_position=np.zeros((2, 7)),
            joint_velocity=velocity,
        ),
        commands=np.zeros((2, 6)),
        previous=np.zeros(6),
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
        lower=-np.ones(7),
        upper=np.ones(7),
        past_inner_error_rad=np.zeros((25, 7)),
        terminal_normals=np.eye(7)[6:],
        desired_tcp=desired,
        site_from_tcp=np.eye(4),
        prefix_before=1,
        target_index=1,
        base_target_max=4.0,
    )


def test_batch_keeps_case_pair_alpha_order_and_copies_endpoints_without_clipping():
    diagnostic, producers = batch_inputs()
    producers["early"]["answer"]["evaluated_candidates"][1]["commands"][1, 1] = 1.2
    batch = build_batch(diagnostic, producers)
    assert len(batch) == 18
    for case_index, name in enumerate(("prefix", "early", "late")):
        rows = batch[6 * case_index : 6 * (case_index + 1)]
        trace = producers[name]["answer"]["evaluated_candidates"]
        assert [row["alpha"] for row in rows] == [0, 0.0625, 0.125, 0.25, 0.5, 1]
        assert [row["alpha_index"] for row in rows] == list(range(6))
        assert [row["source_trace_index"] for row in rows] == [0, None, None, None, None, 1]
        assert all(row["case_name"] == name and row["pair_index"] == 1 for row in rows)
        assert all(row["base_target_max"] == 4.0 and row["target_index"] == 1 for row in rows)
        assert rows[0]["commands"].tobytes() == trace[0]["commands"].tobytes()
        assert rows[-1]["commands"].tobytes() == trace[1]["commands"].tobytes()
        assert np.signbit(rows[0]["commands"][0, 0]) and not np.signbit(rows[-1]["commands"][0, 0])
        np.testing.assert_array_equal(
            rows[3]["commands"], trace[0]["commands"] + 0.25 * (trace[1]["commands"] - trace[0]["commands"])
        )
    assert batch[11]["commands"][1, 1] == 1.2


def test_batch_does_not_alias_inputs_other_requests_or_drop_repeated_endpoints():
    diagnostic, producers = batch_inputs()
    trace = producers["prefix"]["answer"]["evaluated_candidates"]
    trace[1]["commands"] = trace[0]["commands"].copy()
    before = deepcopy(producers)
    batch = build_batch(diagnostic, producers)
    assert len(batch) == 18
    assert batch[0]["commands"].tobytes() == batch[5]["commands"].tobytes()
    batch[0]["commands"][:] = 42
    np.testing.assert_array_equal(
        batch[5]["commands"], before["prefix"]["answer"]["evaluated_candidates"][0]["commands"]
    )
    for name, producer in producers.items():
        for row, saved in zip(
            producer["answer"]["evaluated_candidates"], before[name]["answer"]["evaluated_candidates"]
        ):
            np.testing.assert_array_equal(row["commands"], saved["commands"])


@pytest.mark.parametrize("invalid", ["qbase_one_ulp", "error_nan", "command_inf", "index", "horizon"])
def test_batch_rejects_changed_saved_scores_and_malformed_inputs(invalid):
    diagnostic, producers = batch_inputs()
    case = diagnostic["cases"]["prefix"]
    trace = producers["prefix"]["answer"]["evaluated_candidates"]
    if invalid == "qbase_one_ulp":
        case["pairs"][1]["qbase"][1][0] = np.nextafter(4.0, np.inf)
    elif invalid == "error_nan":
        trace[0]["tcp_errors"][0, 0] = np.nan
    elif invalid == "command_inf":
        trace[1]["commands"][0, 0] = np.inf
    elif invalid == "index":
        case["pairs"][1]["base_index"] = -1
    else:
        producers["prefix"]["answer"]["horizon"] = 3
    with pytest.raises(ValueError):
        build_batch(diagnostic, producers)


def test_safe_descent_can_preserve_prefix_without_hitting_target_or_full_witness():
    inputs = scoring_inputs()
    result = score_backtracking_response(**inputs)
    np.testing.assert_allclose(result["normalized_squared_errors"], [[0.04, 0], [1.44, 0]], rtol=0, atol=3e-16)
    assert result["prefix_length"] == 1 and result["prefix_preserved"]
    assert result["joint_feasible"] and result["buffer_satisfied"] and result["feasible_descent"]
    assert result["target_improvement"] == pytest.approx(2.56)
    assert result["target_improved"] and not result["target_passed"] and not result["feasible"]
    assert set(result["audit"]["groups"]) == {
        "box_lower",
        "box_upper",
        "rate_lower",
        "rate_upper",
        "inner_lower",
        "inner_upper",
        "braking_lower",
        "braking_upper",
        "terminal",
    }
    np.testing.assert_array_equal(result["audit"]["groups"]["terminal"]["slack"], [0.01])
    assert result["safety_min"] == 0.01


def test_tcp_transform_and_rotational_caps_are_scored_in_the_registered_frame():
    inputs = scoring_inputs()
    tool = inputs["site_from_tcp"]
    tool[:3, 3] = [0.02, -0.03, 0.04]
    tool[:3, :3] = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
    inputs["desired_tcp"][:] = tool
    inputs["desired_tcp"][1, :3, :3] = Rotation.from_rotvec([0, 0, 0.06]).as_matrix() @ tool[:3, :3]
    result = score_backtracking_response(**inputs)
    np.testing.assert_allclose(result["tcp_errors"][:, :3], 0, rtol=0, atol=0)
    assert result["target_max"] == pytest.approx(1.44, abs=1e-14)
    assert not result["target_passed"] and result["prefix_preserved"]


@pytest.mark.parametrize("outside", [False, True])
def test_pose_caps_have_no_diagnostic_tolerance_even_one_ulp(outside):
    inputs = scoring_inputs()
    inputs["desired_tcp"][:, 0, 3] = 0
    inputs["desired_tcp"][0, 0, 3] = np.nextafter(0.01, np.inf) if outside else 0.01
    result = score_backtracking_response(**inputs)
    assert result["prefix_preserved"] is (not outside)
    assert result["feasible"] is (not outside)
    assert result["feasible_descent"] is (not outside)


@pytest.mark.parametrize("safety", [0.0, 5e-9, 1e-8])
def test_complete_witness_does_not_bypass_the_separate_descent_buffer(safety):
    inputs = scoring_inputs()
    inputs["desired_tcp"][:, 0, 3] = 0
    inputs["rollout"]["joint_velocity"][-1, 6] = -safety
    result = score_backtracking_response(**inputs)
    assert result["feasible"] and result["joint_feasible"] and result["target_improved"]
    assert result["buffer_satisfied"] is (safety >= 1e-8)
    assert result["feasible_descent"] is (safety >= 1e-8)


@pytest.mark.parametrize("above", [False, True])
def test_improvement_threshold_is_strict_and_does_not_change_control_caps(above):
    inputs = scoring_inputs()
    inputs["desired_tcp"][:, 0, 3] = 0
    inputs["base_target_max"] = np.nextafter(1e-12, np.inf) if above else 1e-12
    result = score_backtracking_response(**inputs)
    assert result["target_improved"] is above and result["feasible_descent"] is above
    assert result["target_passed"] and result["feasible"]


@pytest.mark.parametrize("excess,accepted", [(0.5e-12, True), (2e-12, False)])
def test_command_rate_uses_original_one_e_minus_twelve_audit_tolerance(excess, accepted):
    inputs = scoring_inputs()
    inputs["commands"][:, 0] = 0.25 + excess
    result = score_backtracking_response(**inputs)
    group = result["audit"]["groups"]["rate_upper"]
    assert group["tolerance"] == 1e-12 and group["slack"][0, 0] < 0
    assert result["joint_feasible"] is accepted and result["feasible_descent"] is accepted


@pytest.mark.parametrize("excess,accepted", [(0.5e-10, True), (2e-10, False)])
def test_inner_joint_tolerance_is_separate_from_the_extra_buffer(excess, accepted):
    inputs = scoring_inputs()
    inputs["rollout"]["inner_joint_position"][30, 2] = 0.95 + excess
    result = score_backtracking_response(**inputs)
    assert result["audit"]["groups"]["inner_upper"]["tolerance"] == 1e-10
    assert result["joint_feasible"] is accepted
    assert not result["buffer_satisfied"] and not result["feasible_descent"]


def test_terminal_and_braking_failures_are_not_hidden_by_pose_improvement():
    inputs = scoring_inputs()
    inputs["rollout"]["joint_velocity"][-1, 6] = 0.01
    inputs["rollout"]["joint_position"][0, 0] = 0.94
    inputs["rollout"]["joint_velocity"][0, 0] = 0.2
    result = score_backtracking_response(**inputs)
    assert result["target_improved"] and result["prefix_preserved"]
    assert result["audit"]["groups"]["terminal"]["minimum"] == -0.01
    assert result["audit"]["groups"]["braking_upper"]["minimum"] == pytest.approx(-0.01)
    assert not result["joint_feasible"] and not result["feasible_descent"]


def test_profile_margins_and_scoring_do_not_mutate_or_alias_inputs():
    inputs = scoring_inputs()
    inputs["past_inner_error_rad"][:, 2] = np.arange(25) * 0.001
    before = deepcopy(inputs)
    result = score_backtracking_response(**inputs)
    np.testing.assert_allclose(
        result["audit"]["position_margin_rad"][:25, 2], 0.05 + np.arange(25) * 0.001, rtol=0, atol=0
    )
    np.testing.assert_array_equal(result["audit"]["position_margin_rad"][25:, 2], np.full(25, 0.05 + 0.024))
    result["audit"]["groups"]["inner_upper"]["slack"][:] = 12
    result["tcp_errors"][:] = 12
    for name, value in inputs["rollout"].items():
        np.testing.assert_array_equal(value, before["rollout"][name])
    for name in ("commands", "previous", "desired_tcp", "site_from_tcp", "past_inner_error_rad"):
        np.testing.assert_array_equal(inputs[name], before[name])


@pytest.mark.parametrize("invalid", ["base_nan", "base_negative", "target_index", "prefix", "pose_shape", "inner_nan"])
def test_invalid_scoring_contract_is_rejected(invalid):
    inputs = scoring_inputs()
    if invalid == "base_nan":
        inputs["base_target_max"] = np.nan
    elif invalid == "base_negative":
        inputs["base_target_max"] = -1
    elif invalid == "target_index":
        inputs["target_index"] = 2
    elif invalid == "prefix":
        inputs["prefix_before"] = True
    elif invalid == "pose_shape":
        inputs["rollout"]["pose"] = np.eye(4)
    else:
        inputs["rollout"]["inner_joint_position"][0, 0] = np.nan
    with pytest.raises(ValueError):
        score_backtracking_response(**inputs)

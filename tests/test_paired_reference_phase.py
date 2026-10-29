"""Check the fixed schedules, distinct scoring clocks, and warm admission contract."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface
from s2a.paired_reference_phase import (
    assess_phase_warm,
    prepare_paired_reference_phases,
    score_phase_response,
)
from s2a.prefix_preserving_trajectory_plan import solve_prefix_preserving_trajectory_response


def reference_inputs(*, stationary=False):
    targets = np.tile(np.eye(4), (40, 1, 1))
    if not stationary:
        targets[:, 0, 3] = 0.02 * np.arange(40)
        targets[:, :3, :3] = Rotation.from_euler("z", 0.12 * np.arange(40)).as_matrix()
    warm = np.zeros((12, 6))
    warm[:, 0] = np.linspace(-0.2, 0.2, 12)
    warm[:, 3] = 0.1 * (-1.0) ** np.arange(12)
    return dict(
        targets_tcp=targets,
        openings=np.linspace(0.1, 0.9, 40),
        warm12=warm,
        previous=warm[0].copy(),
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
    )


def synthetic_response(commands):
    """A small causal joint integrator; joints 0..5 directly determine SITE pose."""
    increments = np.c_[0.001 * commands, np.full(len(commands), -0.0005)]
    endpoints = np.cumsum(increments, axis=0)
    starts = np.vstack((np.zeros(7), endpoints[:-1]))
    inner = starts[:, None] + (np.arange(1, 26) / 25)[None, :, None] * increments[:, None]
    velocity = increments / 0.05
    poses = np.tile(np.eye(4), (len(commands), 1, 1))
    poses[:, :3, 3] = endpoints[:, :3]
    poses[:, :3, :3] = Rotation.from_rotvec(endpoints[:, 3:6]).as_matrix()
    return dict(
        inner_joint_position=inner.reshape(-1, 7),
        inner_joint_velocity=np.repeat(velocity, 25, axis=0),
        joint_position=endpoints,
        joint_velocity=velocity,
        pose=poses,
        twist=velocity[:, :6],
    )


def admission_inputs(*, stationary=False):
    inputs = reference_inputs(stationary=stationary)
    if stationary:
        inputs["warm12"][:] = 0
        inputs["previous"][:] = 0
    case = prepare_paired_reference_phases(**inputs)["early"]
    return dict(
        rollout=synthetic_response(case["warm"]),
        case=case,
        previous=inputs["previous"],
        interface=inputs["interface"],
        lower=-2 * np.ones(7),
        upper=2 * np.ones(7),
        site_from_tcp=np.eye(4),
        past_inner_error_rad=np.zeros((25, 7)),
        terminal_normals=np.eye(7)[6:],
    )


@pytest.mark.parametrize("name,left", [("early", 5), ("late", 8)])
def test_schedules_keep_every_original_node_and_one_shared_pose_opening_clock(name, left):
    inputs = reference_inputs()
    case = prepare_paired_reference_phases(**inputs)[name]
    insertion = left + 1
    expected_phases = list(range(insertion)) + [left + 0.5] + list(range(insertion, 12))
    expected_indices = list(range(insertion)) + list(range(insertion + 1, 13))
    np.testing.assert_array_equal(case["phases"], expected_phases)
    np.testing.assert_array_equal(case["original_node_endpoint_indices"], expected_indices)
    np.testing.assert_array_equal(case["desired_tcp"][expected_indices], inputs["targets_tcp"][:12])
    np.testing.assert_array_equal(case["opening"][expected_indices], inputs["openings"][:12])
    np.testing.assert_array_equal(case["original_timed_tcp"], inputs["targets_tcp"][:13])
    np.testing.assert_array_equal(case["original_timed_opening"], inputs["openings"][:13])
    np.testing.assert_array_equal(case["warm"][expected_indices], inputs["warm12"])
    np.testing.assert_array_equal(case["warm"][insertion], inputs["warm12"][[left, left + 1]].mean(axis=0))
    np.testing.assert_array_equal(case["unconsumed_suffix"], np.arange(12, 40))
    np.testing.assert_allclose(case["physical_time_s"], np.arange(1, 14) / 20, atol=1e-15, rtol=0)
    np.testing.assert_allclose(case["reference_time_s"], (np.asarray(expected_phases) + 1) / 20, atol=1e-15, rtol=0)
    assert case["duration_s"] == 0.65 and case["reference_time_s"][-1] == pytest.approx(0.60)
    assert case["scheduled_original_node_count"] == 12
    assert case["scheduled_final_original_cursor"] == 11
    assert case["actual_consumed_reference_count"] == 0
    assert np.max(abs(np.diff(np.vstack((inputs["previous"], case["warm"])), axis=0))) <= 0.25
    assert np.all(abs(case["warm"]) <= 1)
    assert case["warm"].tobytes() == case["used_start"].tobytes()


def test_midpoint_takes_short_rotation_arc_in_a_nontrivial_frame_and_same_opening_fraction():
    inputs = reference_inputs()
    base = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
    inputs["targets_tcp"][5, :3, :3] = base @ Rotation.from_euler("z", 170, degrees=True).as_matrix()
    inputs["targets_tcp"][6, :3, :3] = base @ Rotation.from_euler("z", -170, degrees=True).as_matrix()
    inputs["targets_tcp"][[5, 6], :3, 3] = [[0.1, 0.2, 0.3], [0.7, -0.2, 0.5]]
    inputs["openings"][[5, 6]] = [0.25, 0.75]
    early = prepare_paired_reference_phases(**inputs)["early"]
    np.testing.assert_allclose(early["desired_tcp"][6, :3, :3], base @ np.diag([-1, -1, 1]), atol=1e-15, rtol=0)
    np.testing.assert_allclose(early["desired_tcp"][6, :3, 3], [0.4, 0.0, 0.4], atol=1e-15, rtol=0)
    assert early["opening"][6] == 0.5


def test_reference_snapshots_do_not_alias_inputs_or_the_other_case():
    inputs = reference_inputs()
    before = deepcopy(inputs)
    cases = prepare_paired_reference_phases(**inputs)
    for name in ("targets_tcp", "openings", "warm12", "previous"):
        np.testing.assert_array_equal(inputs[name], before[name])
        inputs[name][...] = 0
    np.testing.assert_array_equal(cases["early"]["original_timed_tcp"], before["targets_tcp"][:13])
    np.testing.assert_array_equal(cases["late"]["source_warm12"], before["warm12"])
    late = deepcopy(cases["late"])
    for value in cases["early"].values():
        if isinstance(value, np.ndarray):
            value[...] = -7
    for name, value in late.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(cases["late"][name], value)


def test_only_cursor12_can_change_the_extra_timed_target_and_later_suffix_does_not_plan():
    inputs = reference_inputs()
    original = prepare_paired_reference_phases(**inputs)
    inputs["targets_tcp"][12, 1, 3] += 2
    inputs["openings"][12] = 0.95
    inputs["targets_tcp"][13:, 2, 3] += 100
    inputs["openings"][13:] = 0.99
    changed = prepare_paired_reference_phases(**inputs)
    for name in ("early", "late"):
        for key in ("desired_tcp", "opening", "warm", "source_warm12", "unconsumed_suffix"):
            np.testing.assert_array_equal(original[name][key], changed[name][key])
        np.testing.assert_array_equal(
            original[name]["original_timed_tcp"][:12], changed[name]["original_timed_tcp"][:12]
        )
        assert changed[name]["original_timed_tcp"][-1, 1, 3] == 2
        assert changed[name]["original_timed_opening"][-1] == 0.95


@pytest.mark.parametrize("name", ["early", "late"])
def test_one_case_preparation_is_available_for_independent_producer_processes(name):
    inputs = reference_inputs()
    one = prepare_paired_reference_phases(**inputs, case_name=name)
    assert list(one) == [name]
    np.testing.assert_array_equal(
        one[name]["desired_tcp"], prepare_paired_reference_phases(**inputs)[name]["desired_tcp"]
    )


@pytest.mark.parametrize("invalid", ["missing_timed13", "opening_shape", "opening_range", "pose", "warm_shape", "rate"])
def test_invalid_reference_or_original_commands_are_rejected_before_any_response(invalid):
    inputs = reference_inputs()
    if invalid == "missing_timed13":
        inputs["targets_tcp"] = inputs["targets_tcp"][:12]
        inputs["openings"] = inputs["openings"][:12]
    elif invalid == "opening_shape":
        inputs["openings"] = inputs["openings"][:-1]
    elif invalid == "opening_range":
        inputs["openings"][0] = 1.01
    elif invalid == "pose":
        inputs["targets_tcp"][0, 0, 0] = np.nan
    elif invalid == "warm_shape":
        inputs["warm12"] = inputs["warm12"][:11]
    else:
        inputs["warm12"][1, 2] = 0.2501
    with pytest.raises(ValueError):
        prepare_paired_reference_phases(**inputs)


@pytest.mark.parametrize("boundary", ["box", "first_rate"])
def test_even_admissible_roundoff_cannot_silently_change_the_registered_warm(boundary):
    inputs = reference_inputs()
    inputs["warm12"][:, 0] = (1.0 if boundary == "box" else 0.25) + 5e-13
    inputs["previous"][0] = 1.0 if boundary == "box" else 0.0
    with pytest.raises(ValueError, match="roundoff clipping"):
        prepare_paired_reference_phases(**inputs)


def test_inserting_an_equal_nonzero_rotation_command_does_not_turn_it_into_a_zero_latch():
    inputs = reference_inputs()
    inputs["warm12"][:, 3] = np.nextafter(0.0, 1.0)
    inputs["previous"][3] = inputs["warm12"][0, 3]
    cases = prepare_paired_reference_phases(**inputs)
    for name, insertion in (("early", 6), ("late", 9)):
        assert cases[name]["warm"][insertion, 3] == inputs["warm12"][0, 3] != 0


@pytest.mark.parametrize("matched", ["retimed_path", "original_timed"])
def test_perfect_one_clock_does_not_qualify_the_other_clock(matched):
    case = prepare_paired_reference_phases(**reference_inputs())["early"]
    poses = case["desired_tcp"] if matched == "retimed_path" else case["original_timed_tcp"]
    score = score_phase_response(poses, case, np.eye(4))
    other = "original_timed" if matched == "retimed_path" else "retimed_path"
    assert score[matched]["all_pose_caps_satisfied"]
    assert not score[other]["all_pose_caps_satisfied"]
    assert score[other]["prefix_length"] < 13
    if matched == "retimed_path":
        assert score["predicted_qualified_original_node_prefix"] == 12
        assert score["original_timed"]["normalized_squared_errors"][-1].max() > 1
    assert score["actual_consumed_reference_count"] == 0


def test_scoring_converts_site_to_tcp_and_does_not_count_an_inserted_endpoint_as_an_original_node():
    case = prepare_paired_reference_phases(**reference_inputs())["late"]
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([0.2, 0.1, -0.3]).as_matrix()
    tool[:3, 3] = [0.03, -0.02, 0.1]
    sites = case["desired_tcp"] @ np.linalg.inv(tool)
    score = score_phase_response(sites, case, tool)
    assert score["retimed_path"]["all_pose_caps_satisfied"]
    assert score["original_nodes"]["tcp_errors"].shape == (12, 6)
    assert score["predicted_final_qualified_original_cursor"] == 11
    np.testing.assert_allclose(score["predicted_tcp"], case["desired_tcp"], atol=1e-15, rtol=0)
    sites[9, 1, 3] += 0.011
    failed = score_phase_response(sites, case, tool)
    assert not failed["retimed_path"]["all_pose_caps_satisfied"]
    assert failed["retimed_path"]["prefix_length"] == 9
    assert failed["original_nodes"]["all_pose_caps_satisfied"]
    assert failed["predicted_qualified_original_node_prefix"] == 12
    assert failed["actual_consumed_reference_count"] == 0


def test_pose_caps_are_strict_at_the_last_timed_endpoint():
    case = prepare_paired_reference_phases(**reference_inputs(stationary=True))["early"]
    poses = case["desired_tcp"].copy()
    poses[-1, 0, 3] = 0.01
    assert score_phase_response(poses, case, np.eye(4))["retimed_path"]["all_pose_caps_satisfied"]
    poses[-1, 0, 3] = np.nextafter(0.01, np.inf)
    scored = score_phase_response(poses, case, np.eye(4))
    assert not scored["retimed_path"]["all_pose_caps_satisfied"]
    assert scored["retimed_path"]["prefix_length"] == 12


@pytest.mark.parametrize("failure", ["inner", "braking", "terminal", "buffer"])
def test_incomplete_warm_admission_recomputes_all_periods_and_distinguishes_audit_from_buffer(failure):
    arguments = admission_inputs()
    if failure == "inner":
        arguments["rollout"]["inner_joint_position"][312, 6] = 1.951
    elif failure == "braking":
        arguments["rollout"]["joint_velocity"][8, 6] = 20
    elif failure == "terminal":
        arguments["rollout"]["joint_velocity"][-1, 6] = 1e-5
    else:
        arguments["rollout"]["joint_velocity"][-1, 6] = -5e-9
    result = assess_phase_warm(**arguments)
    assert not result["complete_witness"] and not result["admitted"]
    assert result["status"] == "warm_admission_failed"
    assert result["joint_feasible"] == (failure == "buffer")
    expected_reason = "optimization_safety_buffer_not_met" if failure == "buffer" else "joint_or_command_audit_failed"
    assert result["rejection_reason"] == expected_reason
    assert result["initial"]["rollout"]["inner_joint_position"].shape == (325, 7)
    assert len(result["initial"]["audit"]["groups"]) == 9


def test_safe_incomplete_warm_is_admitted_without_claiming_a_search_or_a_witness():
    arguments = admission_inputs()
    result = assess_phase_warm(**arguments)
    assert result["admitted"] and result["joint_feasible"]
    assert not result["complete_witness"] and result["status"] == "admitted"
    assert result["rejection_reason"] is None
    assert result["safety_min"] >= 1e-8
    assert result["metrics"]["actual_consumed_reference_count"] == 0


@pytest.mark.parametrize("slack,admitted", [(1e-8, True), (np.nextafter(1e-8, 0), False)])
def test_incomplete_warm_uses_the_exact_frozen_optimization_buffer(slack, admitted):
    arguments = admission_inputs()
    arguments["rollout"]["joint_velocity"][-1, 6] = -slack
    result = assess_phase_warm(**arguments)
    assert result["joint_feasible"] and not result["complete_witness"]
    assert result["admitted"] is admitted
    assert result["safety_min"] == slack


def test_preflight_accepts_empty_terminal_rows_but_not_the_shape_rejected_by_the_solver():
    arguments = admission_inputs()
    arguments["terminal_normals"] = np.empty((0, 7))
    assert assess_phase_warm(**arguments)["admitted"]
    arguments["terminal_normals"] = np.empty(0)
    with pytest.raises(ValueError, match="two-dimensional"):
        assess_phase_warm(**arguments)


def test_complete_warm_buffer_exception_matches_the_frozen_solver_initial_record():
    arguments = admission_inputs(stationary=True)
    arguments["rollout"]["joint_velocity"][-1, 6] = 5e-11
    arguments["rollout"]["inner_joint_velocity"][-25:, 6] = 5e-11
    assessed = assess_phase_warm(**arguments)
    assert assessed["status"] == "complete_witness" and assessed["admitted"]
    assert assessed["safety_min"] == -5e-11
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return deepcopy(arguments["rollout"])

    answer = solve_prefix_preserving_trajectory_response(
        predict,
        arguments["previous"],
        arguments["case"]["warm"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        arguments["case"]["desired_tcp"],
        arguments["site_from_tcp"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
    )
    assert answer["witness_found"] and answer["response_evaluations"] == len(calls) == 1
    assert answer["total_iterations"] == 0 and answer["phases"] == []
    initial = assessed["initial"]
    assert set(initial) == set(answer["initial"])
    for key in ("commands", "tcp_errors", "normalized_squared_errors", "safety_slack"):
        np.testing.assert_array_equal(initial[key], answer["initial"][key])
    for key in (
        "prefix_length",
        "maximum_normalized_squared_error",
        "maximum_pose_cap_violation",
        "pose_loss",
        "feasible",
    ):
        assert initial[key] == answer["initial"][key]
    for name, group in initial["audit"]["groups"].items():
        np.testing.assert_array_equal(group["slack"], answer["initial"]["audit"]["groups"][name]["slack"])


def test_admission_snapshots_rollout_and_does_not_modify_causal_profile_or_case():
    arguments = admission_inputs()
    arguments["past_inner_error_rad"][:, 4] = np.linspace(0, 0.005, 25)
    before = deepcopy(arguments)
    result = assess_phase_warm(**arguments)
    margin = result["initial"]["audit"]["position_margin_rad"]
    np.testing.assert_array_equal(margin[:25, 4], 0.05 + arguments["past_inner_error_rad"][:, 4])
    np.testing.assert_array_equal(margin[25:, 4], np.full(300, 0.055))
    np.testing.assert_array_equal(arguments["past_inner_error_rad"], before["past_inner_error_rad"])
    for key, value in before["rollout"].items():
        np.testing.assert_array_equal(arguments["rollout"][key], value)
    # Some source arrays share views (twist and endpoint velocity). Mutate only
    # after checking the entire unchanged source, then check the returned copy.
    for key in before["rollout"]:
        arguments["rollout"][key][...] = -9
    for key, value in before["rollout"].items():
        np.testing.assert_array_equal(result["initial"]["rollout"][key], value)
    arguments["case"]["warm"][...] = -8
    np.testing.assert_array_equal(result["initial"]["commands"], before["case"]["warm"])

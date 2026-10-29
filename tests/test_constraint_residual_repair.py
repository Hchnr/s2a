"""Observed residual corrections use synthetic saved data, without a robot model."""

from copy import deepcopy

import numpy as np
import pytest
from test_tail_translation_reserve import synthetic_case

import s2a.constraint_residual_repair as repair

EARLY_BRAKING_ROW = 4284 + 4 * 7 + 6


def squared(errors):
    return np.column_stack((np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2))


def saved_stencil(*, nominal_deltas=None):
    """A nonlinear trial intercept accompanies a complete signed FD stencil."""
    x, previous = np.zeros((12, 6)), np.zeros(6)
    x[0, 0] = 0.123456789  # Nominal and represented FD displacements differ.
    y = x.copy()
    y[4, 0] = 0.001
    errors = np.zeros((12, 6))
    errors[:, 0], errors[:, 3] = 0.005, 0.025
    errors[9, 0], errors[9, 3] = 0.012, 0.06
    errors[10:, 0], errors[10:, 3] = 0.02, 0.1
    safety = np.full(4369, 0.5)
    safety[EARLY_BRAKING_ROW], safety[-1] = 2e-6, 3e-6

    def entry(commands, values, residual):
        return dict(
            commands=commands.copy(),
            tcp_errors=values.copy(),
            safety_slack=residual.copy(),
            normalized_squared_errors=squared(values),
            audit=dict(feasible=True),
        )

    base = entry(x, errors, safety)
    trial_errors, trial_safety = errors.copy(), safety.copy()
    trial_errors[9, 0], trial_errors[9, 3] = 0.011, 0.055
    trial_safety[EARLY_BRAKING_ROW], trial_safety[-1] = -1.5e-7, -6e-6
    trial = entry(y, trial_errors, trial_safety)
    trial["audit"]["feasible"] = False
    deltas = np.where(np.arange(72) % 2 == 0, 1e-4, -1e-4) if nominal_deltas is None else nominal_deltas.copy()
    probes = []
    for axis, delta in enumerate(deltas):
        command, values, residual = x.copy(), errors.copy(), safety.copy()
        command.flat[axis] += delta
        values[9, 0] += delta * (axis + 1)
        residual[0] += 2 * delta
        residual[EARLY_BRAKING_ROW] += (axis + 1) * delta
        residual[-1] -= 3 * delta
        probes.append(entry(command, values, residual))
    return dict(
        base=base,
        trial=trial,
        probes=probes,
        nominal_deltas=deltas,
        previous=previous,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        stage_lower=x - 0.005,
        stage_upper=x + 0.005,
    )


def block(problem, name):
    start, stop = problem["row_blocks"][name]
    return problem["A"][start:stop], problem["b"][start:stop]


def test_observed_trial_intercepts_repair_early_braking_and_keep_every_hard_row():
    arguments = saved_stencil()
    problem = repair.build_repair_problem(**arguments)
    assert problem["A"].shape == (4678, 73)
    assert problem["Js"].shape == (4369, 72) and problem["Jq"].shape == (24, 72)
    assert problem["new_response_evaluations"] == 0 and not problem["empirical_error_bound_used"]
    safety_A, safety_b = block(problem, "safety")
    np.testing.assert_array_equal(safety_A[:, :72], problem["Js"])
    np.testing.assert_array_equal(safety_b, arguments["trial"]["safety_slack"] - 1e-8)
    assert safety_b[EARLY_BRAKING_ROW] == -1.6e-7
    # A positive correction must increase this earlier braking slack. Reusing
    # the feasible base intercept would incorrectly admit the zero correction.
    point = np.zeros(73)
    assert (safety_A @ point + safety_b)[EARLY_BRAKING_ROW] < 0
    point[0] = 1e-6
    assert (safety_A @ point + safety_b)[EARLY_BRAKING_ROW] > 0
    old_intercept = arguments["base"]["safety_slack"] - 1e-8
    assert old_intercept[EARLY_BRAKING_ROW] > 0
    np.testing.assert_array_equal(problem["B"], np.eye(72))
    np.testing.assert_array_equal(problem["g"], np.zeros(72))
    assert problem["initial_t"] == 0 and not np.any(problem["A"][:-1, -1])
    np.testing.assert_array_equal(problem["A"][-1], np.r_[np.zeros(72), 1])
    assert problem["b"][-1] == 0
    large_t = point.copy()
    large_t[-1] = 1e9
    np.testing.assert_array_equal((problem["A"] @ point)[:-1], (problem["A"] @ large_t)[:-1])


def test_saved_signed_nominal_steps_are_the_fd_denominators_and_pose_rows_are_interleaved():
    arguments = saved_stencil()
    # Old compact traces have tcp_errors but no redundant normalized field.
    del arguments["probes"][3]["normalized_squared_errors"]
    problem = repair.build_repair_problem(**arguments)
    actual = arguments["probes"][0]["commands"].flat[0] - arguments["base"]["commands"].flat[0]
    nominal = arguments["nominal_deltas"][0]
    assert actual != nominal
    difference = arguments["probes"][0]["safety_slack"] - arguments["base"]["safety_slack"]
    np.testing.assert_array_equal(problem["Js"][:, 0], difference / nominal)
    assert problem["Js"][0, 0] != (difference / actual)[0]
    np.testing.assert_array_equal(problem["nominal_deltas"], arguments["nominal_deltas"])
    for axis, nominal in enumerate(arguments["nominal_deltas"]):
        derivative = (2 * 0.012 * (axis + 1) + nominal * (axis + 1) ** 2) / 0.01**2
        assert problem["Jq"][18, axis] == pytest.approx(derivative, rel=2e-13)
        assert not np.any(np.delete(problem["Jq"][:, axis], 18))
    prefix_A, prefix_b = block(problem, "prefix")
    target_A, target_b = block(problem, "target")
    np.testing.assert_array_equal(prefix_A[:, :72], -problem["Jq"][:18])
    np.testing.assert_array_equal(prefix_b, 0.9999 - squared(arguments["trial"]["tcp_errors"])[:9].ravel())
    np.testing.assert_array_equal(target_A[:, :72], -problem["Jq"][18:20])
    assert problem["threshold"] == pytest.approx((1.44 + 1.21) / 2)
    np.testing.assert_array_equal(target_b, problem["threshold"] - squared(arguments["trial"]["tcp_errors"])[9])


def test_rate_and_command_rows_constrain_the_trial_plus_correction_in_original_units():
    arguments = saved_stencil()
    problem = repair.build_repair_problem(**arguments)
    delta = np.linspace(-0.001, 0.001, 72).reshape(12, 6)
    point = np.r_[delta.ravel(), 0]
    commands = arguments["trial"]["commands"] + delta
    change = np.diff(np.vstack((arguments["previous"], commands)), axis=0)
    for name, expected in (
        ("rate_lower", 0.25 + change),
        ("rate_upper", 0.25 - change),
        ("command_lower", commands - problem["command_lower"]),
        ("command_upper", problem["command_upper"] - commands),
    ):
        A, b = block(problem, name)
        np.testing.assert_allclose(A @ point + b, expected.ravel(), rtol=0, atol=6e-17)
    np.testing.assert_array_equal(
        problem["command_lower"], np.maximum(arguments["stage_lower"], arguments["trial"]["commands"] - 0.0025)
    )
    np.testing.assert_array_equal(
        problem["command_upper"], np.minimum(arguments["stage_upper"], arguments["trial"]["commands"] + 0.0025)
    )


def test_original_stage_clips_the_first_command_to_the_previous_command_rate_box():
    nominal = np.where(np.arange(72) % 2 == 0, 1e-4, -1e-4)
    nominal[0] = -1e-4
    arguments = saved_stencil(nominal_deltas=nominal)
    shift = 0.24999 - arguments["base"]["commands"][0, 0]
    for entry in (arguments["base"], arguments["trial"], *arguments["probes"]):
        entry["commands"][0, 0] += shift
    x = arguments["base"]["commands"]
    global_lower, global_upper = -np.ones((12, 6)), np.ones((12, 6))
    global_lower[0], global_upper[0] = -0.25, 0.25
    arguments["stage_lower"] = np.maximum(global_lower, x - 0.005)
    arguments["stage_upper"] = np.minimum(global_upper, x + 0.005)
    problem = repair.build_repair_problem(**arguments)
    assert problem["stage_upper"][0, 0] == problem["command_upper"][0, 0] == 0.25
    assert problem["delta_upper"][0, 0] == pytest.approx(1e-5)
    # A locally small correction can still violate the original first rate.
    point = np.zeros(73)
    point[0] = 2e-5
    answer = repair.fixed_repair_commands(problem, dict(qualified=True, point=point))
    assert not answer["admitted"]
    assert not answer["commands"][0]["admission"]["rate"]
    assert not answer["commands"][0]["admission"]["stage"]


@pytest.mark.parametrize(
    "defect", ["length", "delta", "same_axis", "two_axes", "step", "saved_q", "stage", "anchor", "descent"]
)
def test_invalid_saved_stencil_or_scientific_admission_is_rejected_before_qp(monkeypatch, defect):
    arguments = saved_stencil()
    if defect == "length":
        arguments["probes"].pop()
    elif defect == "delta":
        arguments["nominal_deltas"][0] = 2e-4
    elif defect == "same_axis":
        arguments["probes"][1]["commands"] = arguments["probes"][0]["commands"].copy()
    elif defect == "two_axes":
        arguments["probes"][0]["commands"].flat[1] += 1e-8
    elif defect == "step":
        arguments["probes"][0]["commands"].flat[0] += 1e-10
    elif defect == "saved_q":
        arguments["trial"]["normalized_squared_errors"][0, 0] += 1e-15
    elif defect == "stage":
        arguments["stage_lower"][0, 0] -= 1e-10
    elif defect == "anchor":
        arguments["base"]["safety_slack"][EARLY_BRAKING_ROW] = 0
    else:
        arguments["trial"]["tcp_errors"][9] = arguments["base"]["tcp_errors"][9]
        arguments["trial"]["normalized_squared_errors"] = squared(arguments["trial"]["tcp_errors"])
    monkeypatch.setattr(repair, "solve_epigraph_qp", lambda *_args, **_kwargs: pytest.fail("Admission precedes QP"))
    with pytest.raises(ValueError):
        repair.build_repair_problem(**arguments)


@pytest.mark.parametrize("qualified", [True, False])
def test_solve_returns_the_single_qp_record_without_retry_or_reinterpretation(monkeypatch, qualified):
    problem = repair.build_repair_problem(**saved_stencil())
    result, calls = dict(qualified=qualified, solver_status=7, retained="diagnostic"), []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(repair, "solve_epigraph_qp", solve)
    assert repair.solve_repair(problem) is result and len(calls) == 1
    args, kwargs = calls[0]
    assert all(
        left is right for left, right in zip(args[:4], (problem[key] for key in ("B", "g", "A", "b")), strict=True)
    )
    assert args[4] == 0 and kwargs == dict(max_iterations=100, certificate_tolerance=1e-6)


def test_solver_exception_is_not_retried(monkeypatch):
    calls, error = [], RuntimeError("Synthetic solver interruption")

    def solve(*args, **kwargs):
        calls.append(None)
        raise error

    monkeypatch.setattr(repair, "solve_epigraph_qp", solve)
    with pytest.raises(RuntimeError) as caught:
        repair.solve_repair(repair.build_repair_problem(**saved_stencil()))
    assert caught.value is error and len(calls) == 1


def test_fixed_ray_contains_all_nine_exact_unclipped_commands_and_does_not_alias_inputs():
    problem = repair.build_repair_problem(**saved_stencil())
    point = np.r_[np.full(72, 0.001), 0]
    before = problem["trial_commands"].copy()
    answer = repair.fixed_repair_commands(problem, dict(qualified=True, point=point))
    assert answer["admitted"] and answer["reason"] == "admitted" and answer["planned_request_count"] == 9
    assert [row["alpha"] for row in answer["commands"]] == [2**-i for i in range(9)]
    for index, row in enumerate(answer["commands"]):
        assert row["alpha_index"] == index and all(row["admission"].values())
        np.testing.assert_array_equal(row["commands"], before + 2**-index * point[:72].reshape(12, 6))
    assert len({row["commands"].tobytes() for row in answer["commands"]}) == 9
    answer["commands"][0]["commands"][0, 0] = 0.9
    np.testing.assert_array_equal(problem["trial_commands"], before)


@pytest.mark.parametrize("defect", ["radius", "zero", "stage", "rate"])
def test_one_illegal_full_step_denies_the_entire_fixed_ray_without_clipping(defect):
    problem = repair.build_repair_problem(**saved_stencil())
    point = np.r_[np.full(72, 0.001), 0]
    if defect == "radius":
        point[0] = 0.003
    elif defect == "zero":
        point[:] = 0
    elif defect == "stage":
        problem["trial_commands"][0, 0] = problem["stage_upper"][0, 0]
    else:
        problem["previous"][0] = problem["trial_commands"][0, 0] - 0.25
    answer = repair.fixed_repair_commands(problem, dict(qualified=True, point=point))
    assert not answer["admitted"] and answer["reason"] == "fixed_ray_preflight_failed"
    assert len(answer["commands"]) == 9
    if defect == "radius":
        assert not answer["commands"][0]["admission"]["radius"]
        assert answer["commands"][1]["admission"]["radius"]
        assert answer["commands"][0]["actual_delta"].flat[0] == pytest.approx(0.003)
    elif defect == "zero":
        assert all(not row["admission"]["nonzero"] and not row["admission"]["unique"] for row in answer["commands"])
    else:
        assert not answer["commands"][0]["admission"][defect]


def test_unqualified_qp_never_attempts_to_interpret_its_missing_point():
    assert repair.fixed_repair_commands({}, dict(qualified=False)) == dict(
        admitted=False, reason="qp_unqualified", commands=[], planned_request_count=9
    )


def test_original_actual_caps_are_distinct_from_the_prefix_optimization_margin_and_full_witness():
    entry = deepcopy(saved_stencil()["trial"])
    entry["audit"]["feasible"] = True
    entry["safety_slack"][:] = 2e-8
    entry["tcp_errors"][0, 0] = 0.01 * np.sqrt(1 - 0.5e-4)
    entry["normalized_squared_errors"] = squared(entry["tcp_errors"])
    answer = repair.classify_repair_entry(entry, 1.325)
    assert answer["repair_eligible"] and answer["prefix9_preserved"] and answer["descent_preserved"]
    assert not answer["prefix_optimization_margin_satisfied"] and not answer["feasible"]
    # A valid terminal constraint cannot conceal an earlier braking failure.
    entry["safety_slack"][EARLY_BRAKING_ROW] = -1e-12
    answer = repair.classify_repair_entry(entry, 1.325)
    assert not answer["safety_buffer_satisfied"] and not answer["repair_eligible"]
    entry["safety_slack"][:] = 2e-8
    entry["tcp_errors"][:] = 0
    entry["normalized_squared_errors"] = squared(entry["tcp_errors"])
    assert repair.classify_repair_entry(entry, 1.325)["feasible"]


@pytest.mark.parametrize("defect", ["joint", "prefix", "target"])
def test_joint_audit_prefix_and_retained_descent_are_each_required_for_repair(defect):
    entry = deepcopy(saved_stencil()["trial"])
    entry["audit"]["feasible"] = defect != "joint"
    entry["safety_slack"][:] = 2e-8
    if defect == "prefix":
        entry["tcp_errors"][8, 0] = np.nextafter(0.01, np.inf)
    elif defect == "target":
        entry["tcp_errors"][9, 0] = 0.012
    entry["normalized_squared_errors"] = squared(entry["tcp_errors"])
    answer = repair.classify_repair_entry(entry, 1.325)
    assert not answer["repair_eligible"]
    expected = {"joint": "joint_feasible", "prefix": "prefix9_preserved", "target": "descent_preserved"}[defect]
    assert not answer[expected]
    assert all(
        value
        for key, value in answer.items()
        if key in ("joint_feasible", "safety_buffer_satisfied", "prefix9_preserved", "descent_preserved")
        and key != expected
    )


def test_evaluation_recomputes_every_original_audit_group_and_full_h12_pose_errors():
    arguments = synthetic_case(prior_max_abs_residual=None)
    rollout = arguments["predict"](arguments["anchor"])
    result = repair.evaluate_repair_entry(
        rollout,
        arguments["anchor"],
        arguments["previous"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        arguments["desired"],
        arguments["site_from_tcp"],
        arguments["past_inner_error_rad"],
        arguments["terminal_normals"],
    )
    assert result["audit"]["horizon"] == 12 and len(result["audit"]["groups"]) == 9
    assert result["safety_slack"].shape == (4369,) and result["prefix_length"] == 9 and not result["feasible"]
    assert result["maximum_normalized_squared_error"] == 4 and result["maximum_pose_cap_violation"] == 1
    np.testing.assert_array_equal(result["normalized_squared_errors"], squared(result["tcp_errors"]))
    result["rollout"]["joint_position"][0, 0] = 1
    assert rollout["joint_position"][0, 0] == 0

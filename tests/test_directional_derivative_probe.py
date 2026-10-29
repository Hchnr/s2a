"""Finite CPU response fixtures; these tests do not construct a simulator."""

import json
from copy import deepcopy

import numpy as np
import pytest

import s2a.directional_derivative_probe as probing
from s2a.osc import OSCInterface


def response(commands):
    increments = commands[:, :2]
    endpoint = np.cumsum(increments, axis=0)
    starts = np.vstack((np.zeros(2), endpoint[:-1]))
    inner = starts[:, None] + (np.arange(1, 26) / 25)[None, :, None] * increments[:, None]
    velocity = increments / 0.05
    poses = np.tile(np.eye(4), (len(commands), 1, 1))
    poses[:, :2, 3] = endpoint
    return dict(
        inner_joint_position=inner.reshape(-1, 2),
        inner_joint_velocity=np.repeat(velocity, 25, axis=0),
        joint_position=endpoint,
        joint_velocity=velocity,
        pose=poses,
        twist=np.c_[velocity, np.zeros((len(commands), 4))],
        periods=[dict(position=endpoint[i].copy(), goal=poses[i].copy()) for i in range(len(commands))],
    )


def inputs(predict=response, horizon=1):
    direction = np.zeros((1, horizon, 6))
    direction[0, :, :2] = [0.004, 0.002]
    return dict(
        predict=predict,
        previous=np.zeros(6),
        anchor=np.zeros((horizon, 6)),
        directions=direction,
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
        lower=-2 * np.ones(2),
        upper=2 * np.ones(2),
        desired=np.tile(np.eye(4), (horizon, 1, 1)),
        site_from_tcp=np.eye(4),
        past_inner_error_rad=np.zeros((25, 2)),
        terminal_normals=np.eye(2),
        source_stages=(24,),
        step_sizes=(1e-3, 5e-4),
    )


def test_affine_safety_and_quadratic_pose_have_known_direction_differences():
    arguments = inputs()
    arguments["anchor"][0, :2] = [0.02, 0.01]
    arguments["desired"][0, :2, 3] = [0.03, -0.02]
    result = probing.probe_directional_derivatives(**arguments)
    unit = np.array([1.0, 0.5])
    inner_gradient = (np.arange(1, 26)[:, None] / 25 * unit).ravel()
    safety_gradient = np.r_[inner_gradient, -inner_gradient, 3 * unit, -3 * unit, -20 * unit]
    pose_gradient = 2 * np.dot(np.array([-0.01, 0.03]), unit) / 0.01**2
    pose_quadratic = np.dot(unit, unit) / 0.01**2
    for item in result["differences"]:
        step = item["step"]
        for kind, correction in (("forward", step), ("backward", -step), ("central", 0.0)):
            derivative = item[kind]
            assert derivative["valid"] and derivative["reason"] is None
            np.testing.assert_allclose(derivative["safety_slack"], safety_gradient, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                derivative["normalized_squared_errors"],
                [[pose_gradient + correction * pose_quadratic, 0]],
                rtol=0,
                atol=1e-9,
            )
    assert result["response_evaluations"] == result["planned_request_count"] == 6
    assert result["unique_command_count"] == 5
    assert result["repeatability"]["exact"]
    assert result["total_iterations"] == 0 and not result["control_qualified"]


def test_grid_cache_and_forced_repeat_are_counted_separately_after_a_witness():
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    arguments = inputs(predict)
    arguments["directions"] = np.concatenate([arguments["directions"] * scale for scale in (1, 2, 4)])
    arguments["source_stages"] = (24, 25, 26)
    result = probing.probe_directional_derivatives(**arguments)
    ledger = result["request_ledger"]
    assert result["evaluated_candidates"][0]["feasible"]  # The fixed diagnostic still runs to completion.
    assert result["planned_request_count"] == len(ledger) == 14
    assert result["response_evaluations"] == len(calls) == 6
    assert result["unique_command_count"] == 5
    assert sum(row["outcome"] == "cached" for row in ledger) == 8
    assert [row["sign"] for row in ledger[1:-1]] == [1, -1] * 6
    assert [(row["source_stage"], row["step"]) for row in ledger[1:-1:2]] == [
        (stage, step) for stage in (24, 25, 26) for step in (1e-3, 5e-4)
    ]
    assert ledger[-1]["role"] == "anchor_repeat" and ledger[-1]["outcome"] == "repeated"
    assert ledger[-1]["charged"] and ledger[-1]["repeat_of"] == 0 and ledger[-1]["response_index"] == 5
    assert ledger[-1]["calls_before"] == 5 and ledger[-1]["calls_after"] == 6
    assert calls[0].tobytes() == calls[-1].tobytes()
    np.testing.assert_array_equal(result["direction_scales"], [0.004, 0.008, 0.016])
    np.testing.assert_array_equal(result["pairwise_direction_linf"], np.zeros((3, 3)))
    np.testing.assert_allclose(result["pairwise_direction_cosine"], np.ones((3, 3)), atol=1e-15)


def test_box_rate_rejections_are_unclipped_and_joint_invalid_probes_still_run():
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    arguments = inputs(predict, horizon=2)
    arguments["directions"][:] = 0
    arguments["directions"][0, :, 0] = [1, -1]
    arguments["step_sizes"] = (0.2, 0.1)
    arguments["interface"] = OSCInterface(-0.1 * np.ones(6), 0.1 * np.ones(6), -np.ones(6), np.ones(6))
    arguments["lower"], arguments["upper"] = -0.06 * np.ones(2), 0.06 * np.ones(2)
    result = probing.probe_directional_derivatives(**arguments)
    plus, minus = result["request_ledger"][1:3]
    assert plus["admission"]["reasons"] == ["box_lower", "box_upper", "rate_lower"]
    assert minus["admission"]["reasons"] == ["box_lower", "box_upper", "rate_upper"]
    for row in (plus, minus):
        assert row["outcome"] == "command_rejected" and not row["charged"] and row["response_index"] is None
        assert np.max(np.abs(row["commands"])) == 0.2
    assert len(calls) == result["response_evaluations"] == 4
    assert all(np.max(np.abs(command)) <= 0.1 for command in calls)
    assert not result["evaluated_candidates"][1]["audit"]["feasible"]
    assert not result["evaluated_candidates"][2]["audit"]["feasible"]
    for kind in ("forward", "backward", "central"):
        assert result["differences"][0][kind] == dict(
            valid=False,
            reason="command_rejected",
            safety_slack=None,
            normalized_squared_errors=None,
        )
        assert result["differences"][1][kind]["valid"]


@pytest.mark.parametrize(("extra", "admitted"), [(5e-13, True), (2e-12, False)])
def test_original_rate_tolerance_is_not_replaced_by_an_exact_zero_gate(extra, admitted):
    arguments = inputs()
    arguments["directions"][:] = 0
    arguments["directions"][0, 0, 0] = 1
    arguments["step_sizes"] = (0.25 + extra,)
    result = probing.probe_directional_derivatives(**arguments)
    assert result["request_ledger"][1]["admission"]["admitted"] is admitted
    assert result["differences"][0]["central"]["valid"] is admitted
    assert result["response_evaluations"] == (4 if admitted else 2)


def test_rounded_zero_displacement_cannot_masquerade_as_zero_derivative():
    arguments = inputs()
    arguments["anchor"][0, 0] = 0.1
    arguments["directions"][:] = 0
    arguments["directions"][0, 0, 0] = 1
    arguments["step_sizes"] = (1e-20,)
    result = probing.probe_directional_derivatives(**arguments)
    assert result["response_evaluations"] == 2 and result["unique_command_count"] == 1
    for row in result["request_ledger"][1:3]:
        assert row["outcome"] == "cached" and row["response_index"] == 0 and row["zero_displacement"]
        assert row["relative_displacement_error_inf"] == 1
    assert all(
        result["differences"][0][kind]["reason"] == "zero_displacement" for kind in ("forward", "backward", "central")
    )


def test_anchor_repeat_mismatch_is_retained_without_retry_or_averaging():
    calls = []

    def drifting(commands):
        calls.append(commands.copy())
        result = response(commands)
        if len(calls) == 4:
            result["joint_velocity"][-1, 1] += 1e-6
        return result

    arguments = inputs(drifting)
    arguments["step_sizes"] = (1e-3,)
    result = probing.probe_directional_derivatives(**arguments)
    repeat = result["repeatability"]
    assert len(calls) == 4 and result["response_evaluations"] == 4
    assert not repeat["exact"] and result["derivative_reliability_limited"]
    assert repeat["arrays"]["rollout.joint_velocity"]["max_abs_difference"] == 1e-6
    assert repeat["arrays"]["audit.groups.terminal.slack"]["max_abs_difference"] == 1e-6
    assert result["evaluated_candidates"][0]["rollout"]["joint_velocity"][-1, 1] == 0
    assert result["evaluated_candidates"][-1]["rollout"]["joint_velocity"][-1, 1] == 1e-6
    assert result["differences"][0]["central"]["valid"]


def test_repeat_numeric_scope_survives_json_empty_arrays_and_nested_periods():
    arguments = inputs()
    arguments["terminal_normals"] = np.empty((0, 2))
    result = probing.probe_directional_derivatives(**arguments)
    archived = json.loads(
        json.dumps(result, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
    )
    reconstructed = probing._repeat_comparison(
        archived["evaluated_candidates"][0],
        archived["evaluated_candidates"][-1],
        archived["response_evaluations"] - 1,
    )
    assert reconstructed == result["repeatability"]
    assert reconstructed["arrays"]["audit.groups.terminal.active_indices"]["shape"] == [0]
    assert reconstructed["arrays"]["rollout.periods[0].goal"]["exact"]
    assert "no dtype" in reconstructed["comparison_scope"]


def test_predict_receives_copies_and_reused_callback_storage_cannot_rewrite_history():
    buffer, observed = {}, []

    def predict(commands):
        observed.append(commands.copy())
        buffer.clear()
        buffer.update(response(commands))
        commands[:] = 99
        return buffer

    arguments = inputs(predict)
    before = deepcopy(arguments)
    result = probing.probe_directional_derivatives(**arguments)
    for commands, record in zip(observed, result["evaluated_candidates"], strict=True):
        np.testing.assert_array_equal(record["commands"], commands)
        np.testing.assert_array_equal(record["rollout"]["joint_position"], response(commands)["joint_position"])
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)
    assert result["repeatability"]["exact"]


@pytest.mark.parametrize("defect", ["budget", "zero_direction", "unordered_steps", "invalid_anchor"])
def test_invalid_fixed_grid_is_rejected_before_any_response(defect):
    calls = []
    arguments = inputs(lambda commands: calls.append(commands))
    if defect == "budget":
        arguments["max_responses"] = 5
    elif defect == "zero_direction":
        arguments["directions"][:] = 0
    elif defect == "unordered_steps":
        arguments["step_sizes"] = (1e-4, 1e-3)
    else:
        arguments["anchor"][0, 0] = 0.26
    with pytest.raises(ValueError):
        probing.probe_directional_derivatives(**arguments)
    assert calls == []

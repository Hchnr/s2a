"""Causal native adapter and fresh acceptance checks without a simulator."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
from test_prefix_trust_control_execution import CountingDynamics
from test_tail_translation_execution import complete_case, complete_rollout

from s2a.directional_derivative_probe import _canonical
from s2a.interior_trajectory_search import DEFAULT_CONFIG, SearchEvaluationError


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (
        importlib.import_module("probe_tracking_v5_interior_trajectory_search"),
        importlib.import_module("check_tracking_v5_interior_trajectory_search"),
    )


def counters():
    return dict(prediction_calls=0, response_evaluations=0, numerical_inner_steps=0, query_count=0)


def test_actual_callback_resets_every_state_and_bills_initial_repeat(modules):
    producer, _ = modules
    source, case = complete_case()
    before = _canonical(copy.deepcopy(source))
    calls, dynamics, progress = [], CountingDynamics(), counters()

    def simulated_search(evaluate, historical, Js, Jq, previous, lower, upper, config):
        assert dynamics.queries == 1
        np.testing.assert_array_equal(previous, source["previous_command"][:6])
        commands = np.asarray(historical["commands"]).copy()
        changed = commands.copy()
        changed[0, 0] += 1e-4
        entries = [evaluate(command) for command in (commands, changed, commands)]
        # Caller state, bounds and stored derivatives also remain isolated.
        previous[:] = 9
        lower[:] = -9
        Js[:] = 9
        return dict(
            evaluated_candidates=entries,
            response_evaluations=3,
            unique_command_count=2,
            witness_indices=[],
            qualified_witness_indices=[],
        )

    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    producer.run_search(
        source,
        case["historical_anchor"],
        Js,
        Jq,
        DEFAULT_CONFIG,
        case["interface"],
        dynamics,
        progress,
        search_fn=simulated_search,
        rollout_fn=complete_rollout(source, case, calls),
    )
    assert progress["prediction_calls"] == progress["response_evaluations"] == len(calls) == 3
    assert progress["numerical_inner_steps"] == 900 and progress["query_count"] == 937
    assert progress["unique_command_count"] == 2
    assert calls[0]["commands"].tobytes() == calls[-1]["commands"].tobytes()
    for call in calls:
        for key, source_key in (
            ("q", "current_q"),
            ("velocity", "current_velocity"),
            ("latch", "latched_rotation_goal"),
        ):
            np.testing.assert_array_equal(call[key], source[source_key])
    assert _canonical(source) == before and not Js.any()


def test_partial_call_failure_preserves_cost_and_complete_response_prefix(modules):
    producer, _ = modules
    source, case = complete_case()
    calls, dynamics, progress = [], CountingDynamics(), counters()
    normal = complete_rollout(source, case, calls)

    def interrupted(*args, **kwargs):
        if calls:
            dynamics.queries += 19
            raise RuntimeError("partial integration failed")
        return normal(*args, **kwargs)

    def fake_search(evaluate, historical, *args):
        prefix = dict(evaluated_candidates=[evaluate(historical["commands"])])
        try:
            evaluate(historical["commands"])
        except RuntimeError as error:
            raise SearchEvaluationError("partial search", prefix) from error

    with pytest.raises(SearchEvaluationError, match="partial search"):
        producer.run_search(
            source,
            case["historical_anchor"],
            np.zeros((4369, 72)),
            np.zeros((24, 72)),
            DEFAULT_CONFIG,
            case["interface"],
            dynamics,
            progress,
            search_fn=fake_search,
            rollout_fn=interrupted,
        )
    assert progress["prediction_calls"] == 2 and progress["response_evaluations"] == 1
    assert progress["numerical_inner_steps"] == 300 and progress["query_count"] == 332
    assert len(progress["answer"]["evaluated_candidates"]) == 1


def test_callback_rejects_illegal_commands_before_response_and_cannot_overrun_budget(modules):
    producer, _ = modules
    source, case = complete_case()
    dynamics, progress, calls = CountingDynamics(), counters(), []
    config = dict(DEFAULT_CONFIG, max_responses=2)

    def invalid_search(evaluate, historical, *args):
        invalid = np.asarray(historical["commands"]).copy()
        invalid[4, 0] = 0.9
        with pytest.raises(ValueError):
            evaluate(invalid)
        assert progress["prediction_calls"] == 0 and dynamics.queries == 1
        evaluate(historical["commands"])
        evaluate(historical["commands"])
        evaluate(historical["commands"])

    with pytest.raises(ValueError, match="global response budget"):
        producer.run_search(
            source,
            case["historical_anchor"],
            np.zeros((4369, 72)),
            np.zeros((24, 72)),
            config,
            case["interface"],
            dynamics,
            progress,
            search_fn=invalid_search,
            rollout_fn=complete_rollout(source, case, calls),
        )
    assert progress["response_evaluations"] == len(calls) == 2 and progress["query_count"] == 625


@pytest.mark.parametrize("defect", [None, "safety", "ratio", "ceiling", "repeat", "qualification"])
@pytest.mark.parametrize("mode", ["restore", "pose"])
def test_independent_actual_metrics_drive_acceptance_and_witness_gates(modules, defect, mode):
    _, checker = modules
    config = dict(DEFAULT_CONFIG)

    def metric(qmax, reserve):
        q = np.full((12, 2), 0.5)
        q[-1, 1] = qmax
        safety = np.full(4369, config["safety_buffer"] + config["reserve_target"] * reserve)
        safety[-1] = config["safety_buffer"] + config["reserve_target"] * reserve / config["terminal_time_scale"]
        return dict(safety_slack=safety, normalized_squared_errors=q, joint_feasible=True, feasible=False)

    metrics = (
        [metric(4.0, 2.0), metric(3.0, 2.0), metric(4.0, 2.0)]
        if mode == "pose"
        else [metric(4.0, 0.1), metric(4.0, 0.5), metric(4.0, 0.1)]
    )
    rollouts = [dict(value=np.array([1.0])), dict(value=np.array([2.0])), dict(value=np.array([1.0]))]
    answer = dict(
        evaluated_candidates=[{}, {}, {}],
        pose_ceiling=np.full(24, 5.0),
        initial_comparison=dict(exact=True),
        repeatability=dict(exact=True),
        qualified_witness_indices=[],
        stages=[
            dict(
                mode=mode,
                anchor_before_index=0,
                attempts=[dict(response_index=1, predicted_reduction=1.0 if mode == "pose" else 0.4, accepted=True)],
            )
        ],
    )
    if defect == "safety":
        metrics[1]["safety_slack"][0] = np.nextafter(config["safety_buffer"], -np.inf)
    elif defect == "ratio":
        if mode == "pose":
            metrics[1]["normalized_squared_errors"][-1, 1] = 3.95
        else:
            metrics[1] = metric(4.0, 0.11)
    elif defect == "ceiling":
        metrics[1]["normalized_squared_errors"][0, 0] = 5.01
    elif defect == "repeat":
        rollouts[-1]["value"][0] += 1e-12
    elif defect == "qualification":
        answer["qualified_witness_indices"] = [1]
    if defect:
        with pytest.raises(ValueError):
            checker.verify_fresh_decisions(answer, rollouts, metrics, config)
    else:
        assert checker.verify_fresh_decisions(answer, rollouts, metrics, config)["passed"]

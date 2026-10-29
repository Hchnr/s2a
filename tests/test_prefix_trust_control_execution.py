"""Synthetic H12 execution admission and billing, without a dynamics response."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from s2a.prefix_trust_control_evidence import reference
from tests.support.trajectory_rollout import CountingDynamics, captured_rollout, source_fixture


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_tracking_v5_prefix_trust_control")


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("execute_tracking_v5_prefix_trust_control")


def three_response_solver(source, dynamics, calls):
    def solver(predict, previous, warm, interface, lower, upper, desired, tool, **options):
        calls.append(True)
        assert dynamics.queries == 1  # No outer warm response or free preheating.
        np.testing.assert_array_equal(previous, source["previous_command"][:6])
        np.testing.assert_array_equal(lower, source["joint_limits"][:, 0])
        np.testing.assert_array_equal(upper, source["joint_limits"][:, 1])
        np.testing.assert_array_equal(desired, source["desired_tcp"])
        np.testing.assert_array_equal(tool, source["site_from_tcp"])
        assert options["max_evaluations"] == 4000
        assert options["max_iterations"] == 60
        assert options["finite_difference_step"] == 1e-4
        np.testing.assert_array_equal(options["past_inner_error_rad"], source["past_inner_error_rad"])
        np.testing.assert_array_equal(options["terminal_normals"], source["terminal_normals"])
        commands = [warm.copy() for _ in range(3)]
        commands[1][1, 0] += 0.01
        commands[2][1, 0] -= 0.01
        for command in commands:
            predict(command)
        # Solver inputs are caller-owned snapshots too.
        previous[:] = 99
        warm[:] = 99
        desired[:] = 99
        options["past_inner_error_rad"][:] = 99
        return dict(
            response_evaluations=3,
            total_iterations=2,
            evaluated_candidates=[dict(commands=command) for command in commands],
        )

    return solver


@pytest.mark.parametrize("case_name", ["baseline", "candidate"])
def test_both_arms_bill_the_same_first_warm_and_reset_every_response(producer, case_name):
    source, warm, interface = source_fixture()
    original, warm_bytes = deepcopy(source), warm.tobytes()
    dynamics, rollout_calls, solver_calls = CountingDynamics(), [], []

    result = producer.run_case(
        case_name,
        source,
        warm,
        interface,
        dynamics,
        solver_fn=three_response_solver(source, dynamics, solver_calls),
        rollout_fn=captured_rollout(source, rollout_calls),
    )

    assert len(solver_calls) == 1
    assert result["prediction_calls"] == result["response_evaluations"] == len(rollout_calls) == 3
    assert result["numerical_inner_steps"] == 900
    assert result["query_count"] == dynamics.queries == 937
    assert result["total_iterations"] == 2
    assert len(dynamics.current_inputs) == 1
    assert rollout_calls[0]["commands"].tobytes() == warm_bytes == warm.tobytes()
    for call in rollout_calls:
        np.testing.assert_array_equal(call["q"], original["current_q"])
        np.testing.assert_array_equal(call["velocity"], original["current_velocity"])
        np.testing.assert_array_equal(call["latch"], original["latched_rotation_goal"])
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original[key])


@pytest.mark.parametrize("defect", ["omit", "reorder", "replace", "iterations"])
def test_all_actual_responses_must_be_retained_in_order_and_within_the_budget(producer, defect):
    source, warm, interface = source_fixture()
    dynamics, calls = CountingDynamics(), []
    good = three_response_solver(source, dynamics, [])

    def bad_solver(*args, **kwargs):
        answer = good(*args, **kwargs)
        if defect == "omit":
            answer["response_evaluations"] -= 1
            answer["evaluated_candidates"].pop()
        elif defect == "reorder":
            answer["evaluated_candidates"][1:] = answer["evaluated_candidates"][:0:-1]
        elif defect == "replace":
            answer["evaluated_candidates"][1]["commands"][1, 0] = np.nextafter(0.11, np.inf)
        else:
            answer["total_iterations"] = 61
        return answer

    progress = {}
    with pytest.raises(ValueError):
        producer.run_case(
            "baseline",
            source,
            warm,
            interface,
            dynamics,
            solver_fn=bad_solver,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert progress["prediction_calls"] == progress["response_evaluations"] == len(calls) == 3
    assert progress["query_count"] == 937


@pytest.mark.parametrize("defect", ["changed_first_warm", "duplicate"])
def test_first_warm_bytes_and_solver_owned_cache_cannot_be_bypassed(producer, defect):
    source, warm, interface = source_fixture()
    calls, progress, dynamics = [], {}, CountingDynamics()

    def solver(predict, previous, supplied, *args, **kwargs):
        if defect == "changed_first_warm":
            supplied[0, 0] = np.nextafter(supplied[0, 0], np.inf)
            predict(supplied)
        else:
            predict(supplied)
            predict(supplied)
        raise AssertionError("Invalid predictor request must be rejected")

    with pytest.raises(ValueError):
        producer.run_case(
            "candidate",
            source,
            warm,
            interface,
            dynamics,
            solver_fn=solver,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    expected = int(defect == "duplicate")
    assert len(calls) == progress["response_evaluations"] == expected
    assert progress["query_count"] == 1 + 312 * expected


@pytest.mark.parametrize("query_error", [-1, 1])
def test_each_completed_response_charges_exactly_312_queries(producer, query_error):
    source, warm, interface = source_fixture()
    dynamics, calls, progress = CountingDynamics(), [], {}
    with pytest.raises(ValueError):
        producer.run_case(
            "baseline",
            source,
            warm,
            interface,
            dynamics,
            solver_fn=three_response_solver(source, dynamics, []),
            rollout_fn=captured_rollout(source, calls, query_error=query_error),
            progress=progress,
        )
    assert len(calls) == progress["response_evaluations"] == 1
    assert progress["numerical_inner_steps"] == 300
    assert progress["query_count"] == 313 + query_error


def test_response_4001_is_rejected_before_new_work_without_resetting_billing(producer):
    source, warm, interface = source_fixture()
    dynamics, progress = CountingDynamics(), {}
    performed = []

    def rollout(*args, **kwargs):
        performed.append(True)
        args[7].queries += 312
        return {}

    def solver(predict, previous, supplied, *args, **kwargs):
        for index in range(4001):
            command = supplied.copy()
            command[1, 0] += index * 1e-7
            predict(command)
        raise AssertionError("The fixed response budget must stop this loop")

    with pytest.raises(ValueError):
        producer.run_case(
            "candidate", source, warm, interface, dynamics, solver_fn=solver, rollout_fn=rollout, progress=progress
        )
    assert len(performed) == progress["response_evaluations"] == 4000
    assert progress["prediction_calls"] == 4001
    assert progress["numerical_inner_steps"] == 1200000
    assert progress["query_count"] == 1248001


def test_failed_rollout_preserves_observed_queries_in_failure_receipt(producer):
    source, warm, interface = source_fixture()
    dynamics, progress = CountingDynamics(), {}

    def rollout(*args, **kwargs):
        args[7].queries += 17
        raise RuntimeError("synthetic incomplete response")

    with pytest.raises(RuntimeError, match="synthetic incomplete"):
        producer.run_case(
            "baseline",
            source,
            warm,
            interface,
            dynamics,
            solver_fn=three_response_solver(source, dynamics, []),
            rollout_fn=rollout,
            progress=progress,
        )
    assert progress["prediction_calls"] == 1
    assert progress["response_evaluations"] == 0
    assert progress["query_count"] == 18


@pytest.mark.parametrize("defect", ["clip", "rate"])
def test_invalid_warm_is_rejected_without_any_query(producer, defect):
    source, warm, interface = source_fixture()
    if defect == "clip":
        warm[0, 0] = 0.25 + 1e-13
    else:
        warm[1, 0] = 0.5
    dynamics = CountingDynamics()
    with pytest.raises(ValueError):
        producer.run_case("baseline", source, warm, interface, dynamics, solver_fn=lambda *a, **k: None)
    assert dynamics.queries == 0


def qa_receipt(execution, monkeypatch, tmp_path, defect=None):
    monkeypatch.setattr(execution, "REPO", tmp_path)
    monkeypatch.setattr(execution, "NATIVE_PYTHON", tmp_path / "native-python")
    sources = {}
    for relative in execution.SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen source\n")
        sources[relative] = reference(path)["sha256"]
    log = tmp_path / "pytest.log"
    log.write_text("synthetic QA passed\n")
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        actual_wait=True,
        returncode=0,
        command=[str(execution.NATIVE_PYTHON), "-m", "pytest", *execution.QA_TESTS],
        runtime=dict(python="3.11.15", numpy="1.26.4", scipy="1.15.3"),
        environment=dict(execution.CPU_ENVIRONMENT),
        source_sha256=sources,
        log_sha256=reference(log)["sha256"],
    )
    if defect == "not_waited":
        qa["actual_wait"] = False
    elif defect == "missing_test":
        qa["command"].remove(execution.QA_TESTS[-1])
    elif defect == "missing_source":
        del qa["source_sha256"][execution.SOURCES[0]]
    elif defect == "gpu":
        qa["environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif defect == "wrong_numpy":
        qa["runtime"]["numpy"] = "2.0.0"
    elif defect == "changed_source":
        (tmp_path / execution.SOURCES[0]).write_text("changed after QA\n")
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(qa))
    return reference(path)


def test_owner_accepts_complete_current_roster_cpu_qa(execution, monkeypatch, tmp_path):
    assert execution.verify_native_qa(qa_receipt(execution, monkeypatch, tmp_path))["passed"]


@pytest.mark.parametrize(
    "defect", ["not_waited", "missing_test", "missing_source", "gpu", "wrong_numpy", "changed_source"]
)
def test_owner_rejects_incomplete_or_changed_qa(execution, monkeypatch, tmp_path, defect):
    with pytest.raises(ValueError):
        execution.verify_native_qa(qa_receipt(execution, monkeypatch, tmp_path, defect))

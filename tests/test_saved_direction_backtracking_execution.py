"""Synthetic accounting and lifecycle checks; no robot response is executed."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from s2a import saved_direction_backtracking_evidence as evidence


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_tracking_v5_saved_direction_backtracking")


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("execute_tracking_v5_saved_direction_backtracking")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return evidence.reference(path)


def closed_receipts(tmp_path, *, defect=None):
    """All references remain valid even when a lifecycle field is falsified."""
    source = tmp_path / "source.py"
    source.write_text("FROZEN = 1\n")
    launch = dict(
        automatic_retry=False,
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        child=dict(pid=123456, start_ticks=7890),
        command=["native-python", "synthetic-checker.py"],
    )
    exited = dict(actual_wait=True, automatic_retry=False, returncode=0)
    release = dict(reaped=True, returncode=0, signal_sent=False, original_error=None, pid=123456, start_ticks=7890)
    receipt = dict(complete=True, passed=True)
    if defect == "not_waited":
        exited["actual_wait"] = False
    elif defect == "not_reaped":
        release["reaped"] = False
    elif defect == "signal":
        release["signal_sent"] = True
    elif defect == "wrong_pid":
        release["pid"] += 1
    elif defect == "reused_pid":
        release["start_ticks"] += 1
    elif defect == "retry":
        launch["automatic_retry"] = True
    elif defect == "gpu":
        launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif defect == "failed_wait":
        exited["returncode"] = 1
    elif defect == "failed_result":
        receipt["passed"] = False
    registration = write_json(
        tmp_path / "registration.json", dict(source_sha256={str(source): evidence.reference(source)["sha256"]})
    )
    result = write_json(tmp_path / "verified.json", receipt)
    owner = dict(
        complete=True,
        passed=True,
        registration_ref=registration,
        result_ref=result,
        owned_command={
            "launch": write_json(tmp_path / "launch.json", launch),
            "exit": write_json(tmp_path / "exit.json", exited),
            "release": write_json(tmp_path / "release.json", release),
        },
    )
    if defect == "failed_owner":
        owner["passed"] = False
    return write_json(tmp_path / "owner.json", owner), source


def test_closed_owner_requires_real_bound_lifecycle_and_rechecks_sources(tmp_path):
    owner_ref, source = closed_receipts(tmp_path)
    refs = {}

    owner, result = evidence.closed_owner(owner_ref, refs)

    assert owner["result_ref"] == evidence.reference(tmp_path / "verified.json")
    assert result == dict(complete=True, passed=True)
    assert len(refs) == 7
    evidence.finish(refs)
    source.write_text("FROZEN = 2\n")
    with pytest.raises(ValueError):
        evidence.finish(refs)


@pytest.mark.parametrize(
    "defect",
    [
        "not_waited",
        "not_reaped",
        "signal",
        "wrong_pid",
        "reused_pid",
        "retry",
        "gpu",
        "failed_wait",
        "failed_result",
        "failed_owner",
    ],
)
def test_complete_files_cannot_substitute_for_successful_matching_cpu_lifecycle(tmp_path, defect):
    owner_ref, _ = closed_receipts(tmp_path, defect=defect)
    with pytest.raises(ValueError):
        evidence.closed_owner(owner_ref, {})


def test_changed_source_is_rejected_before_prerequisite_admission(tmp_path):
    owner_ref, source = closed_receipts(tmp_path)
    source.write_text("FROZEN = 99\n")
    with pytest.raises(ValueError):
        evidence.closed_owner(owner_ref, {})


def endpoint_fixture(horizon=12):
    commands = np.zeros((horizon, 6))
    errors = np.zeros((horizon, 6))
    errors[-1, 0] = 0.01
    saved = dict(
        commands=commands.copy(),
        tcp_errors=errors,
        safety_min=0.1,
        prefix_length=horizon,
        joint_feasible=True,
        feasible=True,
    )
    request = dict(commands=commands.copy(), source_trace_index=0)
    classification = {key: deepcopy(value) for key, value in saved.items() if key != "commands"}
    return request, classification, [saved]


def test_endpoint_comparison_covers_only_fields_the_old_trace_actually_saved(producer):
    request, classification, trace = endpoint_fixture()
    classification["tcp_errors"][0, 1] += 0.5e-10
    classification["safety_min"] += 0.5e-10

    result = producer.compare_saved_endpoint(request, classification, trace)

    assert result["source_trace_index"] == 0
    assert set(result["fields_checked"]) == {
        "commands",
        "tcp_errors",
        "safety_min",
        "prefix_length",
        "joint_feasible",
        "feasible",
    }
    assert result["old_full_rollout_available"] is False
    request["source_trace_index"] = None
    assert producer.compare_saved_endpoint(request, classification, trace) is None


@pytest.mark.parametrize(
    "defect", ["command_ulp", "signed_zero", "tcp_error", "safety", "boundary_label", "joint_label"]
)
def test_endpoint_match_rejects_changed_bytes_and_strict_labels_despite_numeric_tolerance(producer, defect):
    request, classification, trace = endpoint_fixture()
    if defect == "command_ulp":
        request["commands"][0, 0] = np.nextafter(0.0, 1.0)
    elif defect == "signed_zero":
        request["commands"][0, 0] = -0.0
    elif defect == "tcp_error":
        classification["tcp_errors"][0, 1] = 2e-10
    elif defect == "safety":
        classification["safety_min"] += 2e-10
    elif defect == "boundary_label":
        classification["tcp_errors"][-1, 0] = np.nextafter(0.01, np.inf)
        classification["prefix_length"] -= 1
        classification["feasible"] = False
    elif defect == "joint_label":
        classification["joint_feasible"] = False
        classification["feasible"] = False
    with pytest.raises((ValueError, AssertionError)):
        producer.compare_saved_endpoint(request, classification, trace)


def fixed_case(horizon):
    """One direction's complete grid, with two old endpoints and four new points."""
    q = np.arange(7, dtype=float) / 100.0
    velocity = -q / 10.0
    latch = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source = dict(
        current_q=q,
        current_velocity=velocity,
        current_quantities=dict(mass=np.eye(7), jacobian=np.zeros((6, 7)), pose=np.eye(4)),
        latched_rotation_goal=latch,
        previous_command=np.zeros(7),
        initial_joint=np.zeros(7),
        kp=np.full(6, 150.0),
        kd=np.full(6, 24.0),
        torque_limits=np.tile([-80.0, 80.0], (7, 1)),
        joint_limits=np.tile([-2.0, 2.0], (7, 1)),
        past_inner_error_rad=np.full((25, 7), 0.001),
        terminal_normals=np.empty((0, 7)),
        desired_tcp=np.repeat(np.eye(4)[None], horizon, axis=0),
        site_from_tcp=np.eye(4),
        trace=[],
        case_contract=None,
    )
    requests = []
    for index, alpha in enumerate((0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)):
        commands = np.full((horizon, 6), 0.1 * alpha)
        source_index = 0 if index == 0 else 1 if index == 5 else None
        requests.append(
            dict(
                case_name="prefix" if horizon == 12 else "early",
                pair_index=0,
                base_index=0,
                trial_index=1,
                alpha_index=index,
                alpha=alpha,
                prefix_before=horizon - 1,
                target_index=horizon - 1,
                horizon=horizon,
                commands=commands,
                source_trace_index=source_index,
                base_target_max=1.4,
            )
        )
        if source_index is not None:
            source["trace"].append(
                dict(
                    commands=commands.copy(),
                    tcp_errors=np.zeros((horizon, 6)),
                    safety_min=0.1,
                    prefix_length=horizon,
                    joint_feasible=True,
                    feasible=True,
                )
            )
    return requests, source


class FakeDynamics:
    def __init__(self):
        self.queries = 0
        self.current_inputs = []

    def __call__(self, q, velocity):
        self.queries += 1
        self.current_inputs.append((q.copy(), velocity.copy()))
        return dict(mass=np.eye(7), jacobian=np.zeros((6, 7)), pose=np.eye(4))


def fake_response_functions(source, *, query_error=0):
    calls, score_calls = [], []

    def rollout(
        q,
        velocity,
        commands,
        interface,
        initial_joint,
        kp,
        kd,
        dynamics,
        torque_lower,
        torque_upper,
        *,
        latched_rotation_goal,
    ):
        horizon = len(commands)
        calls.append(
            dict(q=q.copy(), velocity=velocity.copy(), latch=latched_rotation_goal.copy(), commands=commands.copy())
        )
        np.testing.assert_array_equal(initial_joint, source["initial_joint"])
        np.testing.assert_array_equal(kp, source["kp"])
        np.testing.assert_array_equal(kd, source["kd"])
        np.testing.assert_array_equal(torque_lower, source["torque_limits"][:, 0])
        np.testing.assert_array_equal(torque_upper, source["torque_limits"][:, 1])
        dynamics.queries += 26 * horizon + query_error
        # A stateful or aliasing harness would leak these synthetic end states
        # into the next alpha. A new response must always receive fresh copies.
        q[:] = 9.0
        velocity[:] = -9.0
        latched_rotation_goal[:] = 0.0
        return dict(
            unsafe=len(calls) == 2,
            inner_joint_position=np.full((25 * horizon, 7), 9.0),
            inner_joint_velocity=np.full((25 * horizon, 7), -9.0),
            joint_position=np.full((horizon, 7), 9.0),
            joint_velocity=np.full((horizon, 7), -9.0),
            pose=np.repeat(np.eye(4)[None], horizon, axis=0),
            twist=np.zeros((horizon, 6)),
        )

    def score(rollout, commands, previous, interface, lower, upper, **options):
        horizon = len(commands)
        score_calls.append(commands.copy())
        np.testing.assert_array_equal(previous, source["previous_command"][:6])
        np.testing.assert_array_equal(lower, source["joint_limits"][:, 0])
        np.testing.assert_array_equal(upper, source["joint_limits"][:, 1])
        for key in ("past_inner_error_rad", "terminal_normals", "desired_tcp", "site_from_tcp"):
            np.testing.assert_array_equal(options[key], source[key])
        joint = not rollout["unsafe"]
        return dict(
            audit=dict(feasible=joint),
            tcp_errors=np.zeros((horizon, 6)),
            normalized_squared_errors=np.zeros((horizon, 2)),
            safety_min=0.1 if joint else -0.1,
            prefix_length=horizon,
            joint_feasible=joint,
            prefix_preserved=True,
            buffer_satisfied=joint,
            target_max=0.0,
            target_improvement=1.4,
            target_improved=True,
            target_passed=True,
            feasible=joint,
            feasible_descent=joint,
        )

    return rollout, score, calls, score_calls


@pytest.mark.parametrize("horizon", [12, 13])
def test_every_alpha_restarts_current_state_and_charges_endpoints_without_early_stopping(producer, horizon):
    requests, source = fixed_case(horizon)
    original = deepcopy(source)
    saved_commands = [request["commands"].copy() for request in requests]
    dynamics = FakeDynamics()
    rollout, score, calls, scored = fake_response_functions(source)

    result = producer.evaluate_case(requests, source, object(), dynamics, rollout_fn=rollout, score_fn=score)

    assert result["response_evaluations"] == len(calls) == len(scored) == len(result["entries"]) == 6
    assert result["numerical_inner_steps"] == 6 * horizon * 25
    assert result["query_count"] == dynamics.queries == 1 + 6 * horizon * 26
    assert len(dynamics.current_inputs) == 1
    np.testing.assert_array_equal(dynamics.current_inputs[0][0], original["current_q"])
    np.testing.assert_array_equal(dynamics.current_inputs[0][1], original["current_velocity"])
    for index, call in enumerate(calls):
        np.testing.assert_array_equal(call["q"], original["current_q"])
        np.testing.assert_array_equal(call["velocity"], original["current_velocity"])
        np.testing.assert_array_equal(call["latch"], original["latched_rotation_goal"])
        np.testing.assert_array_equal(call["commands"], saved_commands[index])
        np.testing.assert_array_equal(scored[index], saved_commands[index])
        np.testing.assert_array_equal(requests[index]["commands"], saved_commands[index])
    for key in ("current_q", "current_velocity", "latched_rotation_goal", "previous_command", "past_inner_error_rad"):
        np.testing.assert_array_equal(source[key], original[key])
    # The first response is already a complete synthetic witness; the second
    # is unsafe. Neither condition is permission to truncate the fixed grid.
    assert result["entries"][0]["classification"]["feasible"]
    assert not result["entries"][1]["classification"]["joint_feasible"]
    assert result["entries"][-1]["classification"]["feasible"]


@pytest.mark.parametrize("query_error", [-1, 1])
def test_each_response_must_charge_exactly_26_queries_per_period(producer, query_error):
    requests, source = fixed_case(12)
    dynamics = FakeDynamics()
    rollout, score, calls, _ = fake_response_functions(source, query_error=query_error)

    with pytest.raises(ValueError):
        producer.evaluate_case(requests, source, object(), dynamics, rollout_fn=rollout, score_fn=score)

    assert len(calls) == 1


def qa_receipt(execution, monkeypatch, tmp_path, *, defect=None):
    monkeypatch.setattr(execution, "REPO", tmp_path)
    monkeypatch.setattr(execution, "NATIVE_PYTHON", tmp_path / "native-python")
    source_sha = {}
    for relative in execution.SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n")
        source_sha[relative] = evidence.reference(path)["sha256"]
    log = tmp_path / "pytest.log"
    log.write_text("synthetic tests passed\n")
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        actual_wait=True,
        returncode=0,
        command=[str(execution.NATIVE_PYTHON), "-m", "pytest", *execution.QA_TESTS],
        runtime=dict(python="3.11.15", numpy="1.26.4", scipy="1.15.3"),
        environment=dict(execution.CPU_ENVIRONMENT),
        source_sha256=source_sha,
        log_sha256=evidence.reference(log)["sha256"],
    )
    if defect == "gpu":
        qa["environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif defect == "missing_test":
        qa["command"].remove(execution.QA_TESTS[-1])
    elif defect == "missing_source":
        del qa["source_sha256"][execution.SOURCES[0]]
    elif defect == "wrong_numpy":
        qa["runtime"]["numpy"] = "2.0.0"
    elif defect == "failed_tests":
        qa["returncode"] = 1
    elif defect == "not_waited":
        qa["actual_wait"] = False
    elif defect == "changed_source":
        (tmp_path / execution.SOURCES[0]).write_text("changed after QA\n")
    return write_json(tmp_path / "qa.json", qa)


def test_owner_admits_only_source_bound_complete_cpu_qa(execution, monkeypatch, tmp_path):
    ref = qa_receipt(execution, monkeypatch, tmp_path)
    qa = execution.verify_native_qa(ref)
    assert qa["passed"]


@pytest.mark.parametrize(
    "defect", ["gpu", "missing_test", "missing_source", "wrong_numpy", "failed_tests", "not_waited", "changed_source"]
)
def test_owner_rejects_insufficient_qa_before_launch(execution, monkeypatch, tmp_path, defect):
    ref = qa_receipt(execution, monkeypatch, tmp_path, defect=defect)
    with pytest.raises(ValueError):
        execution.verify_native_qa(ref)

"""CPU-only wiring, causal reset, complete billing, and zero-response admission."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_prefix_trust_control_execution import CountingDynamics
from test_tail_translation_execution import complete_case, complete_rollout

from s2a.constraint_residual_repair import evaluate_repair_entry
from s2a.directional_derivative_probe import _canonical
from s2a.osc import OSCInterface


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_tracking_v5_constraint_residual_repair")


def progress():
    return dict(prediction_calls=0, response_evaluations=0, numerical_inner_steps=0, query_count=0)


def synthetic():
    source, case = complete_case()
    commands = np.asarray(case["anchor"]).copy()
    rows = []
    for index in range(9):
        command = commands.copy()
        command[0, 0] += 0.001 * 2**-index
        rows.append(dict(alpha=2**-index, commands=command))
    return source, case, rows


@pytest.mark.parametrize("history_matches", [True, False])
def test_every_request_resets_causal_state_and_repeat_is_billed(producer, history_matches):
    source, case, rows = synthetic()
    before = _canonical(copy.deepcopy(source))
    trial = copy.deepcopy(case["historical_anchor"])
    if not history_matches:
        trial["rollout"]["joint_velocity"][0, 0] += 1e-12
    calls, dynamics, result = [], CountingDynamics(), progress()
    answer = producer.run_responses(
        source,
        trial,
        rows,
        4.0,
        case["interface"],
        dynamics,
        result,
        rollout_fn=complete_rollout(source, case, calls),
    )
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 11
    assert result["numerical_inner_steps"] == 3300 and result["query_count"] == dynamics.queries == 3433
    assert result["unique_command_count"] == answer["unique_command_count"] == 10
    assert answer["historical_exact"] is history_matches and answer["repeat_exact"]
    assert answer["selected_response_index"] == (1 if history_matches else None)
    assert not answer["witness_found"] and not answer["control_qualified"]
    expected = [case["anchor"], *[row["commands"] for row in rows], case["anchor"]]
    assert [row["commands"].tobytes() for row in calls] == [command.tobytes() for command in expected]
    for call in calls:
        for key, original in (("q", "current_q"), ("velocity", "current_velocity"), ("latch", "latched_rotation_goal")):
            np.testing.assert_array_equal(call[key], source[original])
    assert _canonical(source) == before


@pytest.mark.parametrize("defect", ["last_rate", "repeat", "missing"])
def test_entire_command_sequence_admitted_before_first_query(producer, defect):
    source, case, rows = synthetic()
    if defect == "last_rate":
        rows[-1]["commands"][5, 0] = 0.9
    elif defect == "repeat":
        rows[-1]["commands"] = rows[0]["commands"].copy()
    else:
        rows.pop()
    dynamics = CountingDynamics()
    with pytest.raises(ValueError):
        producer.run_responses(source, case["historical_anchor"], rows, 2.0, case["interface"], dynamics, progress())
    assert dynamics.queries == 0


def test_partial_response_failure_retains_actual_queries_and_attempt(producer):
    source, case, rows = synthetic()
    calls, dynamics, result = [], CountingDynamics(), progress()
    normal = complete_rollout(source, case, calls)

    def interrupted(*args, **kwargs):
        if calls:
            dynamics.queries += 17
            raise RuntimeError("Interrupted partial response")
        return normal(*args, **kwargs)

    with pytest.raises(RuntimeError, match="Interrupted partial"):
        producer.run_responses(
            source, case["historical_anchor"], rows, 2.0, case["interface"], dynamics, result, rollout_fn=interrupted
        )
    assert result["prediction_calls"] == 2 and result["response_evaluations"] == 1
    assert result["numerical_inner_steps"] == 300 and result["query_count"] == 330
    assert len(result["answer"]["evaluated_candidates"]) == 1
    assert result["answer"]["request_ledger"][-1]["completed"] is False


def test_archived_compact_stencil_and_full_entry_recompute_without_new_qp_or_model(producer):
    if not producer.INPUT.is_file():
        pytest.skip("Archived experiment inputs are unavailable")
    raw = producer.INPUT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == producer.BUNDLE_SHA
    bundle = json.loads(raw)
    problem = producer.prepare_problem(bundle)
    assert problem["threshold"] == 1.3145080074672406
    assert problem["A"].shape == (4678, 73)
    assert problem["Js"].shape == (4369, 72) and problem["Jq"].shape == (24, 72)
    assert not np.any(problem["A"][:-1, -1])
    source, trial, control = (bundle[key] for key in ("source", "trial", "control"))
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    limits = np.asarray(source["joint_limits"])
    rebuilt = evaluate_repair_entry(
        trial["rollout"],
        trial["commands"],
        np.asarray(source["previous_command"])[:6],
        interface,
        limits[:, 0],
        limits[:, 1],
        source["desired_tcp"],
        source["site_from_tcp"],
        source["past_inner_error_rad"],
        source["terminal_normals"],
    )
    assert _canonical(rebuilt) == trial


def fake_main_inputs(producer, monkeypatch, tmp_path):
    source, case, _ = synthetic()
    control = {key: getattr(case["interface"], key) for key in ("input_min", "input_max", "output_min", "output_max")}
    bundle = dict(source=source, source_registration={}, control=control)
    monkeypatch.setattr(producer, "load_inputs", lambda: (bundle, {}))
    monkeypatch.setattr(producer, "prepare_problem", lambda bundle: {"threshold": 2.0})
    monkeypatch.setattr(producer, "model_registration", lambda reg, refs: ({}, {}))
    monkeypatch.setattr(producer, "load_environment_guard", lambda refs: {})
    monkeypatch.setattr(producer, "finish", lambda refs: None)
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="a" * 64))
    monkeypatch.setattr(producer, "strict_native_environment", lambda: None)
    for key, value in producer.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(producer, "ROOT", tmp_path / "exclusive")

    def forbidden(*args, **kwargs):
        raise AssertionError("No runtime or model on zero-response path")

    monkeypatch.setattr(producer, "MujocoRobotDynamics", forbidden)
    monkeypatch.setattr(producer, "restore_cv2_environment", forbidden)
    monkeypatch.setattr(producer, "run_responses", forbidden)


def test_dry_runs_no_qp_and_constructs_no_runtime(producer, monkeypatch, tmp_path):
    fake_main_inputs(producer, monkeypatch, tmp_path)

    def forbidden(problem):
        raise AssertionError("No QP in dry")

    monkeypatch.setattr(producer, "solve_repair", forbidden)
    producer.main(dry=True)
    assert not producer.ROOT.exists()


def test_failed_qp_is_recorded_once_with_zero_runtime_cost(producer, monkeypatch, tmp_path):
    fake_main_inputs(producer, monkeypatch, tmp_path)
    calls = []

    def failed(problem):
        calls.append(problem)
        return dict(qualified=False)

    monkeypatch.setattr(producer, "solve_repair", failed)
    producer.main()
    saved = json.loads((producer.ROOT / "verified.json").read_text())
    assert len(calls) == saved["qp_solves"] == 1
    assert saved["complete"] and saved["passed"] and saved["answer"] is None
    for key in (
        "prediction_calls",
        "response_evaluations",
        "query_count",
        "model_construction_attempts",
        "numerical_inner_steps",
    ):
        assert saved[key] == 0
    assert not saved["witness_found"] and not saved["control_qualified"]

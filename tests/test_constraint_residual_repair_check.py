"""Independent correction algebra, complete billing, and strict saved/fresh gates."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_constraint_residual_repair_execution import progress, synthetic
from test_prefix_trust_control_execution import CountingDynamics
from test_tail_translation_execution import complete_rollout

from s2a.directional_derivative_probe import _canonical


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (
        importlib.import_module("check_tracking_v5_constraint_residual_repair"),
        importlib.import_module("probe_tracking_v5_constraint_residual_repair"),
    )


@pytest.fixture
def archive(modules):
    _, producer = modules
    source, case, rows = synthetic()
    answer = producer.run_responses(
        source,
        case["historical_anchor"],
        rows,
        4.0,
        case["interface"],
        CountingDynamics(),
        progress(),
        rollout_fn=complete_rollout(source, case, []),
    )
    problem = dict(
        previous=np.asarray(source["previous_command"])[:6],
        input_min=case["interface"].input_min,
        input_max=case["interface"].input_max,
        trial_commands=np.asarray(case["anchor"]),
        threshold=4.0,
    )
    return (
        json.loads(json.dumps(_canonical(answer))),
        case["historical_anchor"],
        problem,
        dict(admitted=True, commands=rows),
    )


def fresh_fields(answer):
    return [deepcopy(row["rollout"]) for row in answer["evaluated_candidates"]], [
        dict(
            safety_slack=deepcopy(row["safety_slack"]),
            normalized_squared_errors=deepcopy(row["normalized_squared_errors"]),
            joint_feasible=row["audit"]["feasible"],
        )
        for row in answer["evaluated_candidates"]
    ]


def test_all_eleven_saved_and_fresh_responses_are_checked(modules, archive):
    checker, _ = modules
    answer, trial, problem, plan = archive
    ledger = checker.audit_repair_ledger(answer, trial=trial, problem=problem, command_plan=plan)
    fresh = checker.verify_replayed_repair(answer, *fresh_fields(answer), problem["threshold"])
    assert ledger["passed"] and fresh["passed"]
    assert ledger["selected_response_index"] == fresh["selected_response_index"] == 1
    assert ledger["witness_indices"] == []
    assert ledger["control_qualified"] is fresh["control_qualified"] is False
    assert ledger["full_responses_independently_replayed_here"] is False


@pytest.mark.parametrize(
    "defect", ["missing", "request", "repeat_key", "alpha", "selected", "classification", "history"]
)
def test_saved_ledger_rejects_omissions_and_forged_strict_results(modules, archive, defect):
    checker, _ = modules
    answer, trial, problem, plan = archive
    if defect == "missing":
        answer["evaluated_candidates"].pop()
    elif defect == "request":
        answer["request_ledger"][5]["completed"] = False
    elif defect == "repeat_key":
        answer["request_ledger"][-1]["command_key"] = "00"
    elif defect == "alpha":
        answer["request_ledger"][2]["alpha"] = 0.25
    elif defect == "selected":
        answer["selected_response_index"] = 2
    elif defect == "classification":
        answer["classifications"][1]["descent_preserved"] = False
    else:
        trial["rollout"]["periods"][0]["clipped"][0, 0] = True
    with pytest.raises(ValueError):
        checker.audit_repair_ledger(answer, trial=trial, problem=problem, command_plan=plan)


@pytest.mark.parametrize("defect", ["safety", "prefix", "target", "joint", "margin", "repeat", "missing"])
def test_fresh_route_recomputes_every_strict_gate(modules, archive, defect):
    checker, _ = modules
    answer, _, problem, _ = archive
    rollouts, metrics = fresh_fields(answer)
    if defect == "safety":
        metrics[1]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif defect == "prefix":
        metrics[1]["normalized_squared_errors"][0][0] = np.nextafter(1.0, np.inf)
    elif defect == "target":
        metrics[1]["normalized_squared_errors"][9][0] = np.nextafter(4.0, np.inf)
    elif defect == "joint":
        metrics[1]["joint_feasible"] = False
    elif defect == "margin":
        metrics[1]["normalized_squared_errors"][0][0] = np.nextafter(0.9999, np.inf)
    elif defect == "repeat":
        rollouts[-1]["periods"][0]["clipped"][0][0] = True
    else:
        metrics.pop()
    with pytest.raises(ValueError):
        checker.verify_replayed_repair(answer, rollouts, metrics, problem["threshold"])


def zero_receipt():
    return dict(
        qp_solves=1,
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        numerical_inner_steps=0,
        query_count=0,
        partial_response_queries=0,
        model_construction_attempts=0,
        model_constructions_completed=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        automatic_retry=False,
        answer=None,
        witness_indices=[],
        witness_found=False,
    )


def test_failed_qp_zero_route_has_no_model_or_fake_trajectory_evidence(modules):
    checker, _ = modules
    receipt = zero_receipt()
    checked = checker.audit_execution_costs(receipt, admitted=False)
    assert checked["passed"] and checked["response_evaluations"] == checked["query_count"] == 0
    assert not checked["full_responses_independently_replayed_here"]
    plan = dict(admitted=False, reason="qp_unqualified", commands=[], planned_request_count=9)
    assert checker.audit_command_plan({}, dict(qualified=False), plan)["admitted"] is False
    for key in ("query_count", "model_construction_attempts", "response_evaluations", "qp_solves"):
        forged = deepcopy(receipt)
        forged[key] += 1
        with pytest.raises(ValueError):
            checker.audit_execution_costs(forged, admitted=False)
    receipt["answer"] = {"feasible": True}
    with pytest.raises(ValueError):
        checker.audit_execution_costs(receipt, admitted=False)


@pytest.fixture(scope="module")
def archived_input():
    path = Path(__file__).resolve().parents[1] / "outputs/constraint_residual_repair_inputs_20261029/bundle.json"
    if not path.is_file():
        pytest.skip("Saved experiment stencil is unavailable")
    import hashlib

    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023"
    return json.loads(raw)


def test_original_saved_stencil_rebuilds_exact_matrix_without_solver_or_predictor(modules, archived_input, monkeypatch):
    checker, producer = modules
    expected = producer.prepare_problem(archived_input)

    def forbidden(*args, **kwargs):
        raise AssertionError("Checker must not call the correction builder or solver")

    monkeypatch.setattr(producer, "prepare_problem", forbidden)
    monkeypatch.setattr(producer, "solve_repair", forbidden)
    actual = checker.audit_repair_problem(archived_input, expected)
    assert actual["A"].shape == (4678, 73)
    assert actual["threshold"] == 1.3145080074672406
    assert actual["row_blocks"]["safety"] == [0, 4369]
    assert not np.any(actual["A"][:-1, -1])


@pytest.mark.parametrize(
    "defect", ["safety_row", "prefix_intercept", "target", "radius", "soft_safety", "nominal_step", "fd_axis", "stage"]
)
def test_independent_problem_binding_rejects_small_matrix_and_stencil_changes(modules, archived_input, defect):
    checker, producer = modules
    bundle = deepcopy(archived_input)
    expected = producer.prepare_problem(bundle)
    if defect == "safety_row":
        expected["A"][0, 0] = np.nextafter(expected["A"][0, 0], np.inf)
    elif defect == "prefix_intercept":
        expected["b"][4369] += 1e-12
    elif defect == "target":
        expected["threshold"] += 1e-12
    elif defect == "radius":
        expected["radius"] = 0.005
    elif defect == "soft_safety":
        expected["A"][0, -1] = 1
    elif defect == "nominal_step":
        bundle["nominal_deltas"][2] = 1e-4
    elif defect == "fd_axis":
        bundle["probes"][0]["commands"][1][0] += 1e-12
    else:
        bundle["phase"]["stage_upper_command"][0][0] += 1e-12
    with pytest.raises(ValueError):
        checker.audit_repair_problem(bundle, expected)


def test_preflight_rejects_substituted_alpha_or_clipped_command(modules, archived_input):
    checker, producer = modules
    problem = checker.audit_repair_problem(archived_input)
    qp = dict(qualified=True, point=np.r_[np.full(72, 1e-5), 0.0])
    plan = producer.fixed_repair_commands(problem, qp)
    checker.audit_command_plan(problem, qp, plan)
    forged = deepcopy(plan)
    forged["commands"][4]["alpha"] *= 2
    with pytest.raises(ValueError):
        checker.audit_command_plan(problem, qp, forged)
    forged = deepcopy(plan)
    forged["commands"][0]["commands"][0, 0] += 1e-14
    with pytest.raises(ValueError):
        checker.audit_command_plan(problem, qp, forged)

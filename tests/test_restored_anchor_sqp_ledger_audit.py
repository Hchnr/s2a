"""Independent first-record and restarted SQP ledger tests using synthetic arrays."""

import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_accepted_point_sqp_ledger_audit import saved_classifications
from test_prefix_optimization_reserve_plan import inputs, response

from s2a.accepted_point_sqp_plan import solve_accepted_point_sqp_response
from s2a.restored_anchor_sqp_plan import (
    InitialRecordAdmissionError,
    canonical_initial_comparison,
    make_initial_record_hook,
    solve_restored_anchor_sqp_response,
)

REPO = Path(__file__).resolve().parents[1]
EXPECTED_REF = dict(path="synthetic/tail.json", sha256="a" * 64, locator="answer.evaluated_candidates[5]")


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("restored_anchor_sqp_ledger_audit")


def canonical(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist()))


def build(arguments, **budgets):
    expected = solve_accepted_point_sqp_response(**deepcopy(arguments), max_stages=1)["initial"]
    hook = make_initial_record_hook(expected, expected_ref=EXPECTED_REF)
    answer = solve_restored_anchor_sqp_response(**arguments, initial_record_hook=hook, **budgets)
    return answer, saved_classifications(answer), expected


@pytest.fixture(scope="module")
def archives():
    results = {"warm": build(inputs())}
    arguments = inputs()
    arguments["desired"][:, 0, 3] = 0.015
    results["qp_witness"] = build(arguments, max_stages=5)

    def nonlinear(commands):
        value = response(commands)
        value["pose"][:] = np.eye(4)
        u = commands[0, 0]
        value["pose"][0, 0, 3] = u - 1000 * u**2
        return value

    arguments = inputs(nonlinear, horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    results["curvature"] = build(arguments, max_stages=2)
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    results["partial_fd"] = build(arguments, max_evaluations=9)
    return results


def check(audit, record):
    answer, classified, expected = record
    return audit.audit_restored_anchor_sqp_ledger(
        answer, classified, expected_initial=expected, expected_ref=EXPECTED_REF
    )


@pytest.mark.parametrize("kind", ["warm", "qp_witness", "curvature", "partial_fd"])
def test_restart_hook_and_original_algebra_both_pass_independent_audit(audit, archives, kind):
    answer, _, _ = archives[kind]
    result = check(audit, archives[kind])
    assert result["initial_record_check"]["passed"]
    assert result["initial_record_check"]["full_response_independently_replayed_here"] is False
    assert result["request_count"] == len(answer["request_ledger"])
    assert result["total_qp_iterations"] == answer["total_qp_iterations"]
    assert result["accepted_anchor_indices"] == [row["response_index"] for row in answer["accepted_anchors"]]
    assert result["termination_reason"] == answer["termination_reason"]
    assert not result["full_ray_and_anchor_audits_checked_here"]
    if kind == "curvature":
        assert result["curvature_checks"] and result["qp_certificate_checks"]
    if kind == "partial_fd":
        assert not answer["phases"][-1]["linearization"]["complete"]
    loaded = canonical(archives[kind])
    assert check(audit, loaded)["initial_record_check"] == result["initial_record_check"]


def test_one_ulp_outside_tight_bound_is_copied_and_independently_accepted(audit):
    arguments = inputs()
    arguments["warm"][0, 0] = np.nextafter(0.25, np.inf)
    arguments["warm"][0, 3] = -0.0
    arguments["desired"] = response(arguments["warm"])["pose"]
    captured = []

    def capture(entry):
        captured.append(audit._canonical_value(entry))
        raise ValueError("synthetic fixture extraction only")

    with pytest.raises(InitialRecordAdmissionError):
        solve_restored_anchor_sqp_response(**arguments, initial_record_hook=capture)
    calls = []
    original_predict = arguments["predict"]

    def predict(commands):
        calls.append(commands.copy())
        return original_predict(commands)

    arguments["predict"] = predict
    answer = solve_restored_anchor_sqp_response(
        **arguments, initial_record_hook=make_initial_record_hook(captured[0], expected_ref=EXPECTED_REF)
    )
    assert len(calls) == answer["response_evaluations"] == 1
    assert calls[0].tobytes() == arguments["warm"].tobytes()
    assert np.clip(arguments["warm"], answer["lower_command"], answer["upper_command"]).tobytes() != calls[0].tobytes()
    result = check(audit, (answer, saved_classifications(answer), captured[0]))
    assert result["initial_record_check"]["comparison"]["exact"] and result["witness_indices"] == [0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("checked", 1),
        ("exact", False),
        ("hook_calls", 2),
        ("request_index", 1),
        ("request_count", 2),
        ("response_count", 0),
        ("stages_before", 1),
        ("iterations_before", 1),
        ("qp_iterations_before", 1),
    ],
)
def test_fabricated_or_late_hook_boundary_is_rejected(audit, archives, field, value):
    record = deepcopy(archives["qp_witness"])
    record[0]["initial_record_guard"][field] = value
    with pytest.raises(ValueError, match="hook report"):
        check(audit, record)


@pytest.mark.parametrize(
    "defect", ["nonminimum_safety", "nonminimum_audit", "metric", "command_sign", "source", "report_path"]
)
def test_first_record_cannot_hide_complete_field_or_source_mismatch(audit, archives, defect):
    answer, classified, expected = deepcopy(archives["warm"])
    if defect == "nonminimum_safety":
        row = answer["initial"]["safety_slack"]
        index = int(np.argmax(row))
        row[index] += 1e-13
    elif defect == "nonminimum_audit":
        answer["initial"]["audit"]["groups"]["inner_lower"]["slack"].flat[0] += 1e-13
    elif defect == "metric":
        answer["initial"]["pose_loss"] = np.nextafter(0.0, 1.0)
    elif defect == "command_sign":
        answer["initial"]["commands"][0, 0] = -0.0
    elif defect == "source":
        answer["initial_record_guard"]["expected_ref"]["sha256"] = "b" * 64
    else:
        del answer["initial_record_guard"]["comparison"]["arrays"]["safety_slack"]
    with pytest.raises(ValueError):
        check(audit, (answer, classified, expected))


@pytest.mark.parametrize(
    "mutation", ["bool_type", "integer_type", "shape", "missing", "scalar", "nonfinite", "signed_zero"]
)
def test_independent_complete_comparison_preserves_type_structure_and_signed_command_bytes(audit, mutation):
    expected = dict(
        commands=[[0.0] * 6], labels=[[True, False]], integers=[1, 26], metric=0.5, nest={"row": [0.1, 0.2]}
    )
    observed = deepcopy(expected)
    if mutation == "bool_type":
        observed["labels"][0][0] = 1
    elif mutation == "integer_type":
        observed["integers"][0] = 1.0
    elif mutation == "shape":
        observed["nest"]["row"] = [[0.1, 0.2]]
    elif mutation == "missing":
        del observed["nest"]
    elif mutation == "scalar":
        observed["metric"] = 0.5000000000000001
    elif mutation == "nonfinite":
        observed["nest"]["row"][0] = np.inf
    else:
        observed["commands"][0][0] = -0.0
    result = audit.compare_initial_records(expected, observed)
    assert not result["exact"]
    assert result == canonical_initial_comparison(expected, observed)
    if mutation in ("bool_type", "integer_type"):
        assert not result["scalar_categories_equal"]
    if mutation == "signed_zero":
        assert result["max_abs_difference"] == 0 and not result["commands_bytes_equal"]


def test_old_math_or_cache_corruption_still_fails_after_initial_gate(audit, archives):
    answer, classified, expected = deepcopy(archives["curvature"])
    answer["phases"][1]["curvature_update"]["displacement"][0] += 1e-4
    with pytest.raises(ValueError):
        check(audit, (answer, classified, expected))
    answer, classified, expected = deepcopy(archives["qp_witness"])
    answer["request_ledger"][1]["outcome"] = "new_response"
    with pytest.raises(ValueError):
        check(audit, (answer, classified, expected))


def test_ledger_inverse_ast_preserves_all_original_math_and_routing():
    original = ast.parse((REPO / "scripts/accepted_point_sqp_ledger_audit.py").read_text())
    restored = ast.parse((REPO / "scripts/restored_anchor_sqp_ledger_audit.py").read_text())
    helper_names = {"_canonical_value", "compare_initial_records", "audit_initial_record_guard"}
    restored.body = [
        node
        for node in restored.body
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name in helper_names
            or isinstance(node, ast.ImportFrom)
            and node.module == "collections.abc"
        )
    ]
    old_function = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit_accepted_point_sqp_ledger"
    )
    new_function = next(
        node
        for node in restored.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit_restored_anchor_sqp_ledger"
    )
    new_function.name = old_function.name
    new_function.args.kwonlyargs = new_function.args.kwonlyargs[2:]
    new_function.args.kw_defaults = new_function.args.kw_defaults[2:]
    new_function.body = [
        node
        for node in new_function.body
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "initial_check" for target in node.targets)
        )
    ]
    copy_index = next(
        i
        for i, node in enumerate(new_function.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "copied"
    )
    clip_index = next(
        i
        for i, node in enumerate(old_function.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "clipped"
    )
    # The isolated admission replacement is exactly three adjacent statements.
    assert ast.unparse(new_function.body[copy_index].value) == "warm.copy()"
    new_function.body[copy_index : copy_index + 3] = deepcopy(old_function.body[clip_index : clip_index + 3])
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and node.value == "restored_anchor_lagrangian_sqp":
            node.value = "accepted_point_lagrangian_sqp"
        if isinstance(node, ast.Call):
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "initial_record_check"]
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_native_engine_full_ray_and_entire_trace_replay_loop_are_unchanged():
    original = ast.parse((REPO / "scripts/check_tracking_v5_accepted_point_sqp.py").read_text())
    restored = ast.parse((REPO / "scripts/check_tracking_v5_restored_anchor_sqp.py").read_text())
    for name in ("case_engine", "full_ray_records"):
        old = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in restored.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)

    def replay_loop(tree):
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and ast.unparse(node.iter) == "enumerate(trace)"
        )

    assert ast.dump(replay_loop(original), include_attributes=False) == ast.dump(
        replay_loop(restored), include_attributes=False
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_large_integer_labels_do_not_lose_one_unit_through_float_promotion(audit, mixed):
    expected = dict(commands=[[0.0] * 6], labels=[2**63 + 1])
    if mixed:
        expected["labels"].insert(0, 0.5)
    observed = deepcopy(expected)
    observed["labels"][-1] += 1
    result = audit.compare_initial_records(expected, observed)
    assert result["exact"] is False
    assert result["arrays"]["labels"]["categories_equal"] is True
    assert result["arrays"]["labels"]["mismatched_indices"] == [[int(mixed)]]
    assert result["arrays"]["labels"]["max_abs_difference"] == 1.0
    assert result == canonical_initial_comparison(expected, observed)

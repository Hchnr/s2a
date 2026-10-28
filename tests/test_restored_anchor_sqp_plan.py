"""Restored-start gates and unchanged SQP behavior on synthetic CPU responses."""

import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import test_accepted_point_sqp_plan as original_tests
from test_prefix_optimization_reserve_plan import inputs, response

import s2a.accepted_point_sqp_plan as original
import s2a.restored_anchor_sqp_plan as planning
from s2a.geometry import pose_error
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

REFERENCE = dict(path="synthetic.json", sha256="0" * 64, locator="answer.evaluated_candidates[5]")


def initial_entry(arguments, rollout=None):
    """Build the archived full analytic initial record without invoking predict."""
    commands = arguments["warm"].copy()
    rollout = response(commands) if rollout is None else deepcopy(rollout)
    audit = audit_joint_preview_constraints(
        rollout,
        commands,
        arguments["previous"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
        braking_time_s=0.1,
    )
    errors = pose_error(arguments["desired"], rollout["pose"] @ arguments["site_from_tcp"])
    normalized = np.stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
    )
    failures = np.flatnonzero(~np.all(normalized <= 1.0, axis=1))
    prefix = int(failures[0]) if len(failures) else len(commands)
    return dict(
        commands=commands,
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=normalized,
        safety_slack=np.concatenate(
            [
                audit["groups"][name]["slack"].ravel()
                for name in ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
            ]
        ),
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(normalized)),
        maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
        pose_loss=float(np.sum(normalized)),
        feasible=bool(audit["feasible"] and prefix == len(commands)),
    )


def solve(arguments, expected=None, **kwargs):
    expected = initial_entry(arguments) if expected is None else expected
    return planning.solve_restored_anchor_sqp_response(
        **arguments, initial_record_hook=planning.make_initial_record_hook(expected, expected_ref=REFERENCE), **kwargs
    )


def test_roundoff_legal_warm_is_copied_byte_exact_and_initial_witness_costs_one_call(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    arguments = inputs(predict, horizon=1)
    arguments["warm"][0, 0] = np.nextafter(0.25, np.inf)
    arguments["desired"] = response(arguments["warm"])["pose"]
    warm_bytes = arguments["warm"].tobytes()
    assert np.clip(arguments["warm"], -0.25, 0.25).tobytes() != warm_bytes
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("initial witness precedes QP"))
    answer = solve(arguments)
    assert len(calls) == answer["response_evaluations"] == 1
    assert answer["witness_found"] and answer["total_iterations"] == answer["total_qp_iterations"] == 0
    assert answer["phases"] == [] and len(answer["request_ledger"]) == 1
    for array in (answer["supplied_warm"], answer["used_start"], answer["initial"]["commands"], calls[0]):
        assert array.tobytes() == warm_bytes
    assert not np.any(answer["start_roundoff_clip"])
    assert answer["initial_record_guard"]["hook_calls"] == 1
    assert answer["initial_record_guard"]["comparison"]["commands_bytes_equal"]


@pytest.mark.parametrize("change", ["nonminimum_audit_row", "metric", "nested_period", "boolean_category"])
def test_full_initial_mismatch_archives_one_response_and_stops_before_fd_qp(monkeypatch, change):
    calls = []

    def enriched(commands):
        value = response(commands)
        value["periods"] = [dict(goal=np.eye(4), clipped=np.array([False, True]))]
        return value

    def predict(commands):
        calls.append(commands.copy())
        return enriched(commands)

    arguments = inputs(predict, horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    expected = initial_entry(arguments, enriched(arguments["warm"]))
    if change == "nonminimum_audit_row":
        expected["audit"]["groups"]["inner_lower"]["slack"].flat[1] += 1e-8
    elif change == "metric":
        expected["pose_loss"] += 1e-9
    elif change == "nested_period":
        expected["rollout"]["periods"][0]["goal"][0, 3] = 1e-9
    else:
        expected["rollout"]["periods"][0]["clipped"] = np.array([0, 1])
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("mismatch must precede QP"))
    with pytest.raises(planning.InitialRecordAdmissionError) as caught:
        solve(arguments, expected)
    failure = caught.value.failure
    assert len(calls) == failure["response_evaluations"] == 1
    assert failure["answer"] is None and failure["phases"] == []
    assert failure["total_iterations"] == failure["total_qp_iterations"] == failure["total_stages"] == 0
    assert [row["role"] for row in failure["request_ledger"]] == ["warm"]
    assert failure["request_ledger"][0]["outcome"] == "new_response"
    assert failure["expected_ref"] == REFERENCE and not failure["comparison"]["exact"]
    assert isinstance(caught.value.__cause__, ValueError)
    assert planning.canonical_initial_comparison(
        initial_entry(arguments, enriched(arguments["warm"])), failure["observed_entry"]
    )["exact"]
    assert "inner_steps" not in failure and "dynamics_queries" not in failure


def test_hook_receives_deep_readonly_snapshot_and_cannot_mutate_cached_initial():
    arguments = inputs(horizon=1)
    archived = initial_entry(arguments)
    archived["rollout"]["periods"] = [dict(goal=np.eye(4), clipped=np.array([False]))]
    arguments["predict"] = lambda _commands: deepcopy(archived["rollout"])
    bound = planning.make_initial_record_hook(archived, expected_ref=REFERENCE)
    retained = []

    def hook(record):
        with pytest.raises(TypeError):
            record["prefix_length"] = -1
        with pytest.raises(TypeError):
            record["rollout"]["periods"][0]["extra"] = True
        with pytest.raises(ValueError):
            record["commands"][0, 0] = 1
        with pytest.raises(ValueError):
            record["rollout"]["periods"][0]["goal"].setflags(write=True)
        result = bound(record)
        retained.append((record, result))
        return result

    answer = planning.solve_restored_anchor_sqp_response(**arguments, initial_record_hook=hook)
    retained[0][1]["comparison"]["exact"] = False
    assert answer["initial_record_guard"]["comparison"]["exact"]
    assert not np.shares_memory(retained[0][0]["commands"], answer["initial"]["commands"])
    assert planning.canonical_initial_comparison(archived, answer["initial"])["exact"]


def test_arbitrary_hook_exception_preserves_original_error_and_charged_record():
    arguments = inputs(horizon=1)
    original_error = RuntimeError("synthetic archive unavailable")

    def hook(_record):
        raise original_error

    hook.expected_ref = REFERENCE
    with pytest.raises(planning.InitialRecordAdmissionError) as caught:
        planning.solve_restored_anchor_sqp_response(**arguments, initial_record_hook=hook)
    assert caught.value.__cause__ is original_error
    assert caught.value.failure["original_exception"] == dict(type="RuntimeError", message=str(original_error))
    assert caught.value.failure["response_evaluations"] == 1
    assert caught.value.failure["comparison"] is None


@pytest.mark.parametrize("invalid", ["warm", "hook"])
def test_input_admission_errors_consume_no_response(invalid):
    arguments = inputs(lambda *_a: pytest.fail("input admission precedes responses"), horizon=1)

    def hook(_record):
        pytest.fail("input admission precedes hook")

    if invalid == "warm":
        arguments["warm"][0, 0] = 0.25 + 2e-12
    else:
        hook = None
    with pytest.raises((TypeError, ValueError)):
        planning.solve_restored_anchor_sqp_response(**arguments, initial_record_hook=hook)


def test_complete_canonical_comparison_survives_json_but_rejects_bool_integer_and_signed_zero():
    entry = initial_entry(inputs(horizon=1))
    entry["rollout"]["clipped"] = np.array([[True, False]])
    encoded = json.loads(json.dumps(entry, default=lambda value: value.tolist()))
    assert planning.canonical_initial_comparison(entry, encoded)["exact"]
    changed = deepcopy(encoded)
    changed["rollout"]["clipped"] = [[1, 0]]
    report = planning.canonical_initial_comparison(entry, changed)
    assert not report["exact"] and not report["scalar_categories_equal"]
    assert report["arrays"]["rollout.clipped"]["category_mismatch_indices"] == [[0, 0], [0, 1]]
    changed = deepcopy(encoded)
    changed["commands"][0][0] = -0.0
    report = planning.canonical_initial_comparison(entry, changed)
    assert report["arrays"]["commands"]["exact"] and not report["commands_bytes_equal"] and not report["exact"]


@pytest.mark.parametrize(
    "before,after",
    [
        ([0.0, 2**63], [0.0, 2**63 + 1]),
        (np.array([2**63], dtype=np.uint64), np.array([2**63 - 1], dtype=np.int64)),
        ([True, 1.0], [1, 1.0]),
    ],
)
def test_canonical_numeric_lists_preserve_exact_leaf_values_before_float_diagnostics(before, after):
    expected = dict(commands=[[0.0]], payload=before)
    observed = dict(commands=[[0.0]], payload=after)
    report = planning.canonical_initial_comparison(expected, observed)
    assert not report["exact"] and not report["arrays"]["payload"]["exact"]
    assert report["arrays"]["payload"]["mismatched_indices"]
    assert report["arrays"]["payload"]["max_abs_difference"] == (
        0.0 if isinstance(before, list) and before[0] is True else 1.0
    )


def test_initial_gate_is_before_original_warm_safety_rejection():
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    arguments["past_inner_error_rad"][:] = 3.0
    expected = initial_entry(arguments)
    expected["pose_loss"] += 1
    with pytest.raises(planning.InitialRecordAdmissionError):
        solve(arguments, expected)
    with pytest.raises(ValueError, match="Incomplete warm trajectory"):
        solve(arguments)


@pytest.mark.parametrize(
    "case",
    [
        "test_real_qp_finds_full_caps_witness_without_hidden_response_queries",
        "test_real_backtracking_uses_accepted_displacement_and_same_dual_at_both_ends",
        "test_cursor_growth_discards_previous_target_curvature",
        "test_response_budget_mid_stencil_keeps_pending_pair_without_updating",
        "test_fd_full_witness_precedes_qp_and_does_not_require_prefix_reserve",
    ],
)
def test_registered_initial_gate_preserves_existing_meaningful_sqp_regressions(monkeypatch, case):
    # These inherited tests assess SQP algebra/control flow, not historical binding.
    # A self-comparison gate allows them to retain their original response counters.
    def solve_with_gate(*args, **kwargs):
        def gate(record):
            comparison = planning.canonical_initial_comparison(record, record)
            return dict(passed=True, exact=True, comparison=comparison, expected_ref=REFERENCE)

        return planning.solve_restored_anchor_sqp_response(*args, initial_record_hook=gate, **kwargs)

    monkeypatch.setattr(original_tests, "planning", planning)
    monkeypatch.setattr(original_tests, "solve", solve_with_gate)
    getattr(original_tests, case)(monkeypatch)


def test_partial_fd_global_ranking_can_select_a_cached_probe_instead_of_last_anchor():
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    answer = solve(arguments, max_evaluations=9)
    assert not answer["final_selected_is_last_anchor"]
    last = answer["accepted_anchors"][-1]["response_index"]
    candidates = answer["evaluated_candidates"]
    safe = [item for item in candidates if item["joint_feasible"]]
    best = min(
        safe,
        key=lambda item: (-item["prefix_length"], np.max(np.sum(item["tcp_errors"][:, :3] ** 2, axis=1) / 0.01**2)),
    )
    assert answer["selected"]["commands"].tobytes() == best["commands"].tobytes()
    assert answer["selected"]["commands"].tobytes() != candidates[last]["commands"].tobytes()
    assert answer["phases"][-1]["status"] == "response_budget"


def test_ast_inverse_proves_only_registered_algorithm_changes():
    baseline, restored = Path(original.__file__).read_text(), Path(planning.__file__).read_text()
    proof = planning.verify_restored_algorithm_ast(baseline, restored)
    assert proof["passed"] and not proof["helper_equivalence_claimed"]
    assert planning.damped_bfgs is original.damped_bfgs
    assert planning.solve_epigraph_qp is original.solve_epigraph_qp


@pytest.mark.parametrize("change", ["clip", "buffer", "extra_response", "late_hook", "optional_hook", "core_rebind"])
def test_ast_inverse_rejects_algorithm_or_gate_tampering(change):
    baseline, restored = Path(original.__file__).read_text(), Path(planning.__file__).read_text()
    module = ast.parse(restored)
    solver = next(
        item
        for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == "solve_restored_anchor_sqp_response"
    )
    if change in ("clip", "buffer"):
        name = "start" if change == "clip" else "safety_buffer"
        node = next(
            item
            for item in solver.body
            if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name) and item.targets[0].id == name
        )
        node.value = ast.parse("np.clip(warm.ravel(), lo, hi)" if change == "clip" else "1e-7", mode="eval").body
    elif change in ("extra_response", "late_hook"):
        index = next(
            index
            for index, item in enumerate(solver.body)
            if isinstance(item, ast.Assign)
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "initial_record_guard"
        )
        if change == "extra_response":
            solver.body.insert(index, ast.parse("predict(start.reshape(h, 6))").body[0])
        else:
            solver.body[index], solver.body[index + 1] = solver.body[index + 1], solver.body[index]
    elif change == "optional_hook":
        index = next(index for index, item in enumerate(solver.args.kwonlyargs) if item.arg == "initial_record_hook")
        solver.args.kw_defaults[index] = ast.Constant(None)
    else:
        module.body.append(ast.parse("solve_epigraph_qp = lambda *args: None").body[0])
    with pytest.raises(ValueError, match="AST difference"):
        planning.verify_restored_algorithm_ast(baseline, ast.unparse(ast.fix_missing_locations(module)))

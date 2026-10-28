"""Independent ledger/metadata checks on causal toy responses, never native physics."""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_preserving_trajectory_plan import inputs, optimizer_result, response
from test_prefix_trajectory_audit_ledger import classify as independently_classify_toy

import s2a.prefix_trust_region_plan as planning


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_prefix_trust_control")


def classifications(answer, arguments):
    # The reused independent toy check reconstructs FK errors and all nine
    # constraint groups without importing the production constraint auditor.
    checked = independently_classify_toy(answer, arguments)
    for metrics, row in zip(checked, answer["evaluated_candidates"], strict=True):
        metrics["tcp_errors"] = np.asarray(row["tcp_errors"])
    return checked


def arguments():
    values = inputs()
    values["desired"][1, 0, 3] = 0.1
    return values


@pytest.fixture
def recentered_archive(monkeypatch):
    def optimizer(objective, start, **kwargs):
        trial = start.copy()
        trial[6] += 0.005
        objective(trial)
        kwargs["callback"](trial)
        return optimizer_result(trial)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    answer = planning.solve_prefix_trust_region_response(**values, max_iterations=2)
    return answer, classifications(answer, values)


def test_true_descent_recenter_without_prefix_growth_and_shared_cache(checker, recentered_archive):
    answer, metrics = recentered_archive
    checked = checker.audit_trust_search_ledger(answer, metrics)
    assert checked["termination_reason"] == "iteration_budget"
    assert checked["accepted_anchor_indices"] == [0, 1, 2]
    assert checked["initial_prefix_length"] == checked["selected_prefix_length"] == 1
    assert answer["response_evaluations"] == 3
    assert len(answer["request_ledger"]) > 3
    assert [phase["radius"] for phase in answer["phases"]] == [0.01, 0.01]
    assert all(phase["acceptance_reason"] == "target_descent" for phase in answer["phases"])


def test_no_descent_shrinks_three_times_then_stops(checker, monkeypatch):
    def optimizer(_objective, start, **kwargs):
        kwargs["callback"](start)
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    answer = planning.solve_prefix_trust_region_response(**values, max_iterations=20)
    checked = checker.audit_trust_search_ledger(answer, classifications(answer, values))
    assert checked["termination_reason"] == "trust_region_stalled"
    assert answer["response_evaluations"] == 1
    assert [phase["radius"] for phase in answer["phases"]] == [0.01, 0.005, 0.0025]
    assert checked["accepted_anchor_indices"] == [0]


@pytest.mark.parametrize("interruption", ["probe_budget", "final_budget", "callback_budget"])
def test_response_budget_preserves_last_completed_witness(checker, monkeypatch, interruption):
    def optimizer(objective, start, **kwargs):
        # Twelve distinct points plus the initial warm fill H2's minimum
        # thirteen-response budget. Only the final point is a complete witness.
        for index in range(1, 13):
            point = start.copy()
            point[6] = index * 1e-4
            objective(point)
        missing = start.copy()
        missing[7] = 1e-4
        if interruption == "probe_budget":
            kwargs["jac"](start)
        elif interruption == "callback_budget":
            kwargs["callback"](missing)
        return optimizer_result(missing, nit=2)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = inputs()
    values["desired"][1, 0, 3] = 0.01115
    answer = planning.solve_prefix_trust_region_response(**values, max_evaluations=13)
    checked = checker.audit_trust_search_ledger(answer, classifications(answer, values))
    assert checked["termination_reason"] == "witness"
    assert answer["phases"][0]["status"] == "response_budget"
    assert checked["selected_trace_index"] == 12
    assert answer["request_ledger"][-1]["outcome"] == "budget_rejected"
    assert answer["total_iterations"] == (
        2 if interruption == "final_budget" else int(interruption == "callback_budget")
    )


def test_derivative_cache_is_rebuilt_under_new_stage_bounds(checker, monkeypatch):
    calls = []

    def optimizer(_objective, start, **kwargs):
        kwargs["jac"](start)
        kwargs["constraints"][1]["jac"](start)
        calls.append(start.copy())
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    # Flat response removes genuine descent, retaining one anchor as radii shrink.
    values["predict"] = lambda commands: response(np.zeros_like(commands))
    answer = planning.solve_prefix_trust_region_response(**values)
    checked = checker.audit_trust_search_ledger(answer, classifications(answer, values))
    assert len(calls) == 3 and checked["termination_reason"] == "trust_region_stalled"
    bases = [row for row in answer["request_ledger"] if row["role"] == "gradient_base"]
    assert [row["stage_index"] for row in bases] == [0, 1, 2]
    assert len([row for row in answer["request_ledger"] if row["role"] == "gradient_probe"]) == 36
    assert answer["response_evaluations"] == 13  # Cached probes are charged only once.
    forged = deepcopy(answer)
    probe = next(row for row in forged["request_ledger"] if row["role"] == "gradient_probe" and row["stage_index"] == 1)
    probe["nominal_delta"] = -probe["nominal_delta"]
    with pytest.raises(ValueError, match="nominal step"):
        checker.audit_trust_search_ledger(forged, classifications(answer, values))


@pytest.mark.parametrize(
    "corruption",
    ["radius", "anchor", "touched", "cache", "response_count", "iteration", "termination", "rank", "acceptance"],
)
def test_forged_stage_or_request_metadata_is_rejected(checker, recentered_archive, corruption):
    source, metrics = recentered_archive
    answer = deepcopy(source)
    if corruption == "radius":
        answer["phases"][1]["radius"] = 0.005
    elif corruption == "anchor":
        answer["phases"][1]["anchor_before_index"] = 0
    elif corruption == "touched":
        answer["phases"][0]["touched_response_indices"] = [0]
    elif corruption == "cache":
        answer["request_ledger"][1]["outcome"] = "new_response"
    elif corruption == "response_count":
        answer["request_ledger"][-1]["response_evaluations_after"] += 1
    elif corruption == "iteration":
        answer["phases"][0]["iterations"] += 1
    elif corruption == "termination":
        answer["termination_reason"] = "witness"
    elif corruption == "rank":
        answer["selected"] = answer["initial"]
    else:
        answer["accepted_anchors"][-1]["acceptance_reason"] = "prefix_growth"
    with pytest.raises(ValueError):
        checker.audit_trust_search_ledger(answer, metrics)


def test_tiny_cap_boundary_difference_is_not_relaxed(checker, recentered_archive):
    answer, metrics = deepcopy(recentered_archive)
    saved = answer["evaluated_candidates"][0]
    saved["tcp_errors"][0, 0] = 0.01
    saved["prefix_length"] = 1
    metrics[0]["tcp_errors"][0, 0] = np.nextafter(0.01, np.inf)
    metrics[0]["normalized_squared_errors"][0, 0] = np.nextafter(1.0, np.inf)
    metrics[0]["prefix_length"] = 0
    with pytest.raises(ValueError, match="Strict saved and independent"):
        checker.validate_trace_entry(saved, metrics[0])


def test_nested_strings_recurse_locally_without_patching_frozen_modules(checker):
    import check_tracking_v5_saved_direction_backtracking as frozen

    original = frozen.compare_contract
    actual = {"decisions": [{"rejection_reasons": ["joint_or_command", "safety_buffer"]}]}
    checker.compare_contract(actual, deepcopy(actual))
    changed = deepcopy(actual)
    changed["decisions"][0]["rejection_reasons"].reverse()
    with pytest.raises(ValueError):
        checker.compare_contract(actual, changed)
    assert frozen.compare_contract is original


@pytest.mark.parametrize("case,fail_classification", [("baseline", False), ("candidate", False), ("candidate", True)])
def test_main_contract_and_failure_accounting_without_native_queries(
    checker, monkeypatch, tmp_path, case, fail_classification
):
    """Exercise orchestration with prevalidated toy data and an explicit fake engine."""
    from types import SimpleNamespace

    import s2a.prefix_preserving_trajectory_plan as baseline
    import s2a.prefix_trust_control_evidence as evidence

    monkeypatch.setattr(checker, "REPO", tmp_path)
    values = inputs(horizon=12)  # The complete warm requires no optimizer call.
    solver = (
        baseline.solve_prefix_preserving_trajectory_response
        if case == "baseline"
        else planning.solve_prefix_trust_region_response
    )
    answer = solver(**values)
    warm = values["warm"].copy()
    actual_metrics = classifications(answer, values)[0]
    actual_metrics.update(
        {
            key: answer["initial"][key]
            for key in ("safety_slack", "maximum_normalized_squared_error", "maximum_pose_cap_violation", "pose_loss")
        }
    )
    prefix_reg = {"control_ref": {"path": "synthetic-control", "sha256": "fixture"}}
    current = {"pose": np.eye(4)}
    original = dict(
        answer=answer,
        current_quantities=current,
        previous_command=np.zeros(7),
        terminal_normals=values["terminal_normals"],
        past_inner_error_rad=values["past_inner_error_rad"],
    )
    backtracking = {"cases": {"prefix": {"entries": [None] * 55 + [{"request": {"commands": warm}}]}}}

    def reference(path):
        return {"path": str(path), "sha256": "fixture"}

    producer_ref = reference(checker.REPO / f"outputs/tracking_v5_prefix_trust_{case}_20261028/verified.json")
    producer_reg_ref = reference("synthetic-producer-registration")
    producer = dict(
        case_name=case,
        registration_ref=producer_reg_ref,
        warm=warm,
        answer=answer,
        current_quantities=current,
        prediction_calls=1,
        response_evaluations=1,
        total_iterations=0,
        numerical_inner_steps=300,
        query_count=313,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        source_refs_unchanged=[],
    )
    launch_ref = reference("synthetic-launch")
    owner = dict(result_ref=producer_ref, owned_command={"launch": launch_ref})
    producer_reg = dict(
        case_name=case,
        plan_ref=reference(evidence.PLAN),
        seed=42,
        backtracking_ref=reference(
            checker.REPO / "outputs/tracking_v5_saved_direction_backtracking_20261028/verified.json"
        ),
        source_refs=[],
        input_refs=[],
    )
    guard_ref, prefix_check_reg_ref = reference("synthetic-guard"), reference("synthetic-prefix-check-reg")
    objects = {
        evidence.PREFIX_CHECK_REF["path"]: {"registration_ref": prefix_check_reg_ref},
        prefix_check_reg_ref["path"]: {},
        producer_reg_ref["path"]: producer_reg,
        launch_ref["path"]: {
            "command": [
                str(evidence.NATIVE_PYTHON),
                str(checker.REPO / "scripts/probe_tracking_v5_prefix_trust_control.py"),
                "--case",
                case,
            ]
        },
        checker.CONTROL_REF["path"]: {
            "cv2_guard_registration": {"exact_probe_ref": guard_ref, "runtime_source_refs": []}
        },
        prefix_reg["control_ref"]["path"]: {"contract": {"input_min": -np.ones(6), "input_max": np.ones(6)}},
    }
    published, calls = [], {"environment": 0, "reconstruct": 0, "full_rollout": 0, "audit": 0}
    monkeypatch.setattr(evidence, "load_prerequisites", lambda: (original, prefix_reg, backtracking, {}))
    monkeypatch.setattr(evidence, "reference", reference)
    monkeypatch.setattr(evidence, "read", lambda ref: objects[ref["path"]])
    monkeypatch.setattr(evidence, "remember", lambda ref, refs: (refs.update({ref["path"]: ref}) or ref))
    monkeypatch.setattr(evidence, "finish", lambda refs: None)
    monkeypatch.setattr(evidence, "closed_owner", lambda ref, refs: (owner, producer))
    monkeypatch.setattr(checker, "strict_native_environment", lambda: None)
    for key, value in evidence.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    def admit(*_args):
        calls["environment"] += 1
        return {"synthetic": True}

    def publish(path, record):
        published.append((path, deepcopy(record)))
        return reference(path)

    def engine(_name, _original, _reg, _check_reg, result, _refs, _compare):
        result["mj_forward_queries"] += 1

        def reconstruct(commands):
            assert commands.tobytes() == warm.tobytes()
            calls["reconstruct"] += 1
            result["numerical_inner_steps"] += 300
            result["mj_forward_queries"] += 312
            return deepcopy(answer["initial"]["rollout"]), []

        def compare_rollout(actual, archived):
            calls["full_rollout"] += 1
            checker.compare_contract(actual, archived)

        def audit(archived, metrics):
            calls["audit"] += 1
            assert archived["feasible"] is metrics["joint_feasible"]

        def classify(_commands, _rollout):
            result_metrics = deepcopy(actual_metrics)
            if fail_classification:
                result_metrics["prefix_length"] = 11
                result_metrics["normalized_squared_errors"][11, 0] = 1.01
                result_metrics["feasible"] = False
            return result_metrics

        return (
            reconstruct,
            compare_rollout,
            classify,
            audit,
            current,
            SimpleNamespace(nopython_signatures=["synthetic"]),
        )

    monkeypatch.setattr(checker, "restore_cv2_environment", admit)
    monkeypatch.setattr(checker, "publish", publish)
    monkeypatch.setattr(checker, "case_engine", engine)
    # The real dry path must finish before either native admission or reconstruction.
    checker.main(case, dry=True)
    assert published == [] and calls == {"environment": 0, "reconstruct": 0, "full_rollout": 0, "audit": 0}
    if fail_classification:
        with pytest.raises(ValueError, match="Strict saved and independent"):
            checker.main(case)
    else:
        checker.main(case)
    result = published[-1][1]
    assert calls["environment"] == calls["reconstruct"] == 1
    assert result["response_evaluations"] == 1  # Even when classification fails afterward.
    assert result["numerical_inner_steps"] == 300 and result["query_count"] == 313
    assert result["verified_responses"] == int(not fail_classification)
    assert result["complete"] is result["passed"] is (not fail_classification)
    if fail_classification:
        assert len(result["boundary_classification_disagreements"]) == 1
    else:
        assert calls["full_rollout"] == 3  # Three archive aliases, one response.
        assert result["full_candidates_checked"] == ["closest_safety_attempt", "initial", "selected"]
        assert result["accepted_anchor_audits_checked"] == int(case == "candidate")

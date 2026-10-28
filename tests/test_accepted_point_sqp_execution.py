"""Synthetic admission, reset, owner billing and complete independent replay wiring."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_prefix_callback_backtracking_execution as prior
from test_prefix_callback_backtracking_execution import write_json
from test_prefix_trust_control_execution import (
    CountingDynamics,
    captured_rollout,
    source_fixture,
    three_response_solver,
)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        producer=importlib.import_module("probe_tracking_v5_accepted_point_sqp"),
        evidence=importlib.import_module("s2a.accepted_point_sqp_evidence"),
        owner=importlib.import_module("execute_tracking_v5_accepted_point_sqp"),
    )


def decorate_receipt(result, stage):
    result["prefix_optimization_margin"] = 1e-4
    if stage == "diagnose":
        result["answer"].update(
            prefix_optimization_margin=1e-4,
            method="accepted_point_lagrangian_sqp",
            qp_iteration_limit=100,
            qp_certificate_tolerance=1e-6,
            total_qp_iterations=7,
        )
    return result


def owner_run(modules, monkeypatch, tmp_path, stage, **kwargs):
    original_receipt = prior.receipt

    def receipt(stage, count=3):
        return decorate_receipt(original_receipt(stage, count), stage)

    monkeypatch.setattr(prior, "receipt", receipt)
    return prior.owner_run(modules, monkeypatch, tmp_path, stage, **kwargs)


def test_default_new_solver_keeps_same_actual368_reset_and_exact_charge(modules, monkeypatch):
    source, warm, interface = source_fixture()
    before, warm_bytes = copy.deepcopy(source), warm.tobytes()
    dynamics, calls, solver_calls = CountingDynamics(), [], []
    standin = three_response_solver(source, dynamics, solver_calls)

    def solver(*args, **kwargs):
        assert kwargs["max_stages"] == kwargs["max_iterations"] == 60
        assert kwargs["max_evaluations"] == 4000
        return standin(*args, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "s2a.accepted_point_sqp_plan",
        SimpleNamespace(solve_accepted_point_sqp_response=solver),
    )
    result = modules.producer.run_search(source, warm, interface, dynamics, rollout_fn=captured_rollout(source, calls))
    assert solver_calls == [True]
    assert result["prediction_calls"] == result["response_evaluations"] == len(calls) == 3
    assert result["numerical_inner_steps"] == 900 and result["query_count"] == dynamics.queries == 937
    assert calls[0]["commands"].tobytes() == warm_bytes == warm.tobytes()
    for row in calls:
        np.testing.assert_array_equal(row["q"], before["current_q"])
        np.testing.assert_array_equal(row["velocity"], before["current_velocity"])
        np.testing.assert_array_equal(row["latch"], before["latched_rotation_goal"])
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])


def test_duplicate_raw_response_cannot_hide_from_new_search_accounting(modules):
    source, warm, interface = source_fixture()
    dynamics, calls, progress = CountingDynamics(), [], {}

    def duplicate(predict, previous, start, *args, **kwargs):
        predict(start)
        predict(start.copy())

    with pytest.raises(ValueError, match="duplicate raw responses"):
        modules.producer.run_search(
            source,
            warm,
            interface,
            dynamics,
            solver_fn=duplicate,
            rollout_fn=captured_rollout(source, calls),
            progress=progress,
        )
    assert progress["prediction_calls"] == 2 and progress["response_evaluations"] == len(calls) == 1
    assert progress["numerical_inner_steps"] == 300 and progress["query_count"] == 313


@pytest.mark.parametrize("stage,timeout", [("diagnose", 1200), ("check", 5400)])
def test_owner_uses_one_new_entry_fixed_cpu_and_full_budget(modules, monkeypatch, tmp_path, stage, timeout):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    assert run.calls[0]["command"] == modules.owner.entry_command(stage)
    assert "_tracking_v5_accepted_point_sqp.py" in run.calls[0]["command"][1]
    env = run.calls[0]["environment"]
    assert all(env[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in env and "PYOPENGL_PLATFORM" not in env
    directory = run.root(check=stage == "check", owner=True)
    result = json.loads((directory / "result.json").read_text())
    registration = json.loads((directory / "registration.json").read_text())
    assert result["complete"] is result["passed"] is True
    assert registration["automatic_retry"] is False
    assert registration["qp_iteration_limit"] == 100 and registration["qp_total_iteration_limit"] == 6000
    assert registration["qp_certificate_tolerance"] == 1e-6


def test_owner_rejects_partial_replay_even_if_its_internal_counts_agree(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "check", partial=True)
    with pytest.raises(ValueError, match="No partial replay"):
        run.run()
    assert len(run.calls) == 1
    result = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False


def test_owner_rejects_different_qa_before_child_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "check", different_qa=True)
    with pytest.raises(ValueError, match="identical frozen source-bound QA"):
        run.run()
    assert run.calls == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", "old_solver"),
        ("qp_iteration_limit", 101),
        ("qp_certificate_tolerance", 1e-5),
        ("total_qp_iterations", 6001),
    ],
)
def test_owner_cannot_accept_wrong_algorithm_or_qp_budget(modules, key, value):
    receipt = decorate_receipt(prior.receipt("diagnose"), "diagnose")
    receipt["answer"][key] = value
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(receipt, "diagnose")


def native_qa(modules, monkeypatch, tmp_path):
    native = tmp_path / "native_python"
    names = (
        "execute_tracking_v5_prefix_optimization_reserve",
        "execute_tracking_v5_prefix_callback_backtracking",
        "execute_tracking_v5_prefix_trust_check_v2",
        "execute_tracking_v5_prefix_trust_control",
    )
    for module in [modules.owner, *(importlib.import_module(name) for name in names)]:
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", native)
    sources = {
        path: write_json(tmp_path / path, dict(synthetic_source=path))["sha256"] for path in modules.owner.QA_SOURCES
    }
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        actual_wait=True,
        returncode=0,
        command=[str(native), "-m", "pytest", *modules.owner.QA_TESTS],
        runtime=dict(python="3.11.15", numpy="1.26.4", scipy="1.15.3"),
        environment=dict(modules.owner.CPU_ENVIRONMENT),
        source_sha256=sources,
        log_sha256=log["sha256"],
    )
    return qa


def test_native_qa_requires_every_new_source_and_each_of_five_tests(modules, monkeypatch, tmp_path):
    qa = native_qa(modules, monkeypatch, tmp_path)
    target = tmp_path / "qa/result.json"
    assert len(modules.evidence.SOURCES) == 14
    assert len(modules.evidence.QA_TESTS) - len(modules.evidence.RESERVE_QA_TESTS) == 5
    assert modules.owner.verify_native_qa(write_json(target, qa)) == qa
    for source in modules.evidence.SOURCES:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            modules.owner.verify_native_qa(write_json(target, changed))
    for test in modules.evidence.SOURCES[-5:]:
        changed = copy.deepcopy(qa)
        changed["command"].remove(test)
        with pytest.raises(ValueError):
            modules.owner.verify_native_qa(write_json(target, changed))


def test_native_qa_rejects_changed_current_source(modules, monkeypatch, tmp_path):
    qa = native_qa(modules, monkeypatch, tmp_path)
    ref = write_json(tmp_path / "qa/result.json", qa)
    (tmp_path / modules.evidence.SOURCES[0]).write_text("changed after synthetic QA")
    with pytest.raises(ValueError):
        modules.owner.verify_native_qa(ref)


def reserve_receipts(modules, monkeypatch, tmp_path, defect=None):
    """Small real-hash fixtures exercise closed_owner without reading old output."""
    evidence = modules.evidence
    monkeypatch.setattr(evidence, "REPO", tmp_path)
    monkeypatch.setattr(evidence, "NATIVE_PYTHON", tmp_path / "native_python")
    qa_ref = write_json(tmp_path / "old_qa.json", dict(synthetic=True))
    source_ref = write_json(tmp_path / "old_source.json", dict(frozen=True))
    sources = {ref["path"]: ref["sha256"] for ref in (qa_ref, source_ref)}

    def owned(stage, registration, result_ref):
        root = tmp_path / (stage + "_owner")
        reg_ref = write_json(root / "registration.json", dict(registration, stage=stage, source_sha256=sources))
        entry = "check" if stage == "check" else "probe"
        launch = write_json(
            root / "launch.json",
            dict(
                automatic_retry=False,
                selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                command=[
                    str(evidence.NATIVE_PYTHON),
                    str(tmp_path / f"scripts/{entry}_tracking_v5_prefix_optimization_reserve.py"),
                ],
                child=dict(pid=1234, start_ticks=5678),
            ),
        )
        exited = write_json(
            root / "exit.json",
            dict(actual_wait=not (stage == "check" and defect == "no_wait"), returncode=0, automatic_retry=False),
        )
        release = write_json(
            root / "release.json",
            dict(
                reaped=not (stage == "check" and defect == "not_reaped"),
                returncode=0,
                signal_sent=stage == "check" and defect == "signal",
                original_error=None,
                pid=1234,
                start_ticks=5678,
            ),
        )
        return write_json(
            root / "result.json",
            dict(
                complete=True,
                passed=True,
                stage=stage,
                registration_ref=reg_ref,
                result_ref=result_ref,
                owned_command=dict(launch=launch, exit=exited, release=release),
            ),
        )

    producer_ref = write_json(
        tmp_path / "producer.json",
        dict(complete=True, passed=True, response_evaluations=4000, source_refs_unchanged=[source_ref]),
    )
    producer_owner_ref = owned("diagnose", dict(native_component_qa_ref=qa_ref), producer_ref)
    bindings = dict(producer_owner_ref=producer_owner_ref, producer_ref=producer_ref)
    report_registration = write_json(tmp_path / "report_registration.json", bindings)
    report = dict(
        complete=True,
        passed=True,
        registration_ref=report_registration,
        response_evaluations=4000,
        verified_responses=4000,
        numerical_inner_steps=1200000,
        query_count=1248001,
        mj_forward_queries=1248001,
        complete_ray_rollouts_checked=177,
        accepted_anchor_audits_checked=54,
        boundary_classification_disagreements=[],
        witness_found=False,
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        source_refs_unchanged=[source_ref],
    )
    for label, key, value in (
        ("partial", "verified_responses", 3999),
        ("missing_ray", "complete_ray_rollouts_checked", 176),
        ("missing_anchor", "accepted_anchor_audits_checked", 53),
        ("unbilled", "query_count", 1248000),
        ("disagreement", "boundary_classification_disagreements", [{}]),
    ):
        if defect == label:
            report[key] = value
    report_ref = write_json(tmp_path / "checker.json", report)
    check_owner_ref = owned(
        "check", dict(bindings, native_component_qa_ref={} if defect == "changed_qa" else qa_ref), report_ref
    )
    closure = dict(
        complete=True,
        passed=True,
        sources_unchanged=defect != "incomplete_closure",
        producer_owner_ref=producer_owner_ref,
        checker_owner_ref=check_owner_ref,
        checker_result_ref=report_ref,
        qa_ref=qa_ref,
        verified_responses=4000,
        producer_new_responses_by_role=dict(warm=1, gradient_probe=3822, objective=53, backtracking=124),
        source_refs_unchanged=[source_ref],
    )
    closure_ref = write_json(tmp_path / "closure.json", closure)
    for name, ref in (
        ("RESERVE_OWNER_REF", producer_owner_ref),
        ("RESERVE_PRODUCER_REF", producer_ref),
        ("RESERVE_CHECK_OWNER_REF", check_owner_ref),
        ("RESERVE_CHECK_REF", report_ref),
        ("RESERVE_CLOSURE_REF", closure_ref),
        ("RESERVE_QA_REF", qa_ref),
    ):
        monkeypatch.setattr(evidence, name, ref)
    if defect == "changed_source":
        Path(source_ref["path"]).write_text("changed after completed replay")
    if defect == "changed_closure":
        Path(closure_ref["path"]).write_text("changed after closure reference")
    return dict(producer_ref=producer_ref, checker_ref=report_ref, closure_ref=closure_ref, source_ref=source_ref)


def test_complete_reserve_lifecycle_is_hash_bound_and_fully_registered(modules, monkeypatch, tmp_path):
    saved = reserve_receipts(modules, monkeypatch, tmp_path)
    refs = {}
    result = modules.evidence.closed_reserve_replay(refs)
    assert result == {key: saved[key] for key in ("producer_ref", "checker_ref", "closure_ref")}
    assert refs[saved["source_ref"]["path"]] == saved["source_ref"]


@pytest.mark.parametrize(
    "defect",
    [
        "partial",
        "missing_ray",
        "missing_anchor",
        "no_wait",
        "not_reaped",
        "signal",
        "changed_qa",
        "changed_source",
        "changed_closure",
        "incomplete_closure",
        "unbilled",
        "disagreement",
    ],
)
def test_incomplete_or_changed_reserve_replay_cannot_admit_search(modules, monkeypatch, tmp_path, defect):
    reserve_receipts(modules, monkeypatch, tmp_path, defect)
    with pytest.raises(ValueError):
        modules.evidence.closed_reserve_replay({})


def checker_run(modules, monkeypatch, tmp_path):
    checker = importlib.import_module("check_tracking_v5_accepted_point_sqp")
    monkeypatch.setitem(sys.modules, "check_tracking_v5_prefix_callback_backtracking", checker)
    run = prior.new_checker_run(modules, monkeypatch, tmp_path)
    metadata = sys.modules["s2a.prefix_callback_backtracking_evidence"]
    metadata.PREFIX_OPTIMIZATION_MARGIN = 1e-4
    monkeypatch.setitem(sys.modules, "s2a.accepted_point_sqp_evidence", metadata)
    owner, _ = metadata.closed_owner(None, None)
    metadata.read(owner["owned_command"]["launch"])["command"][1] = str(
        tmp_path / "scripts/probe_tracking_v5_accepted_point_sqp.py"
    )
    registration = metadata.read(run.producer["registration_ref"])
    registration.update(experiment="accepted_point_sqp", prefix_optimization_margin=1e-4)
    decorate_receipt(run.producer, "diagnose")
    answer = run.producer["answer"]
    answer["previous_command"] = np.zeros(6)
    safety = np.array([0.1, 0.3, 0.2])
    records = [answer[name] for name in ("initial", "selected", "closest_safety_attempt")]
    records += answer["evaluated_candidates"] + answer["accepted_anchors"]
    records += [attempt["entry"] for attempt in run.attempts if attempt["entry"] is not None]
    for entry in records:
        entry["safety_slack"] = safety.copy()
    original_engine = run.main.__globals__["case_engine"]

    def engine(*args):
        values = list(original_engine(*args))
        original_classify = values[2]

        def classify(commands, rollout):
            return dict(original_classify(commands, rollout), safety_slack=safety.copy())

        values[2] = classify
        return tuple(values)

    run.main.__globals__["case_engine"] = engine
    ledger_calls = []

    def ledger(received, classifications, *, compare):
        assert received is answer and len(classifications) == 2
        assert all(np.array_equal(item["safety_slack"], safety) for item in classifications)
        ledger_calls.append(True)
        return dict(witness_indices=[])

    run.main.__globals__["audit_accepted_point_sqp_ledger"] = ledger
    run.ledger_calls = ledger_calls
    return run


def test_new_checker_main_replays_all_commands_and_full_rays_then_calls_new_ledger(modules, monkeypatch, tmp_path):
    run = checker_run(modules, monkeypatch, tmp_path)
    run.main()
    result = run.published[-1][1]
    assert result["complete"] is result["passed"] is True
    assert result["response_evaluations"] == result["verified_responses"] == 2
    assert result["query_count"] == 625 and result["numerical_inner_steps"] == 600
    assert result["complete_ray_rollouts_checked"] == 3 and len(run.audits) == 7
    assert run.ledger_calls == [True] and result["control_qualified"] is False


@pytest.mark.parametrize("defect", ["safety_vector", "previous_command", "previous_signed_zero", "cached_ray"])
def test_checker_does_not_forward_unverified_data_to_algorithm_ledger(modules, monkeypatch, tmp_path, defect):
    run = checker_run(modules, monkeypatch, tmp_path)
    answer = run.producer["answer"]
    if defect == "safety_vector":
        # Preserve the same minimum: a scalar-only audit would miss this change.
        answer["evaluated_candidates"][0]["safety_slack"][1] = 0.4
    elif defect == "previous_command":
        answer["previous_command"][0] = 0.1
    elif defect == "previous_signed_zero":
        answer["previous_command"][0] = -0.0
    else:
        run.attempts[1]["entry"]["rollout"] = dict(index=99)
    with pytest.raises((ValueError, AssertionError)):
        run.main()
    assert run.ledger_calls == []
    assert run.published[-1][1]["complete"] is False

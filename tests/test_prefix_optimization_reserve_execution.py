"""Synthetic admission and entry wiring for the fixed optimization-only reserve."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_prefix_callback_backtracking_execution as prior_tests
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
        producer=importlib.import_module("probe_tracking_v5_prefix_optimization_reserve"),
        evidence=importlib.import_module("s2a.prefix_optimization_reserve_evidence"),
        owner=importlib.import_module("execute_tracking_v5_prefix_optimization_reserve"),
    )


def owner_run(modules, monkeypatch, tmp_path, stage, **kwargs):
    original_receipt = prior_tests.receipt

    def receipt(stage, count=3):
        result = original_receipt(stage, count)
        result["prefix_optimization_margin"] = 1e-4
        if stage == "diagnose":
            result["answer"]["prefix_optimization_margin"] = 1e-4
        return result

    monkeypatch.setattr(prior_tests, "receipt", receipt)
    return prior_tests.owner_run(modules, monkeypatch, tmp_path, stage, **kwargs)


def callback_receipts(modules, monkeypatch, tmp_path, *, defect=None):
    """Build hash-bound metadata only; the historical physics/search is not run."""
    evidence = modules.evidence
    monkeypatch.setattr(evidence, "REPO", tmp_path)
    monkeypatch.setattr(evidence, "NATIVE_PYTHON", tmp_path / "native_python")

    def root(case_name="candidate", *, check=False, owner=False):
        return tmp_path / ("old_callback" + ("_check" if check else "") + ("_owner" if owner else ""))

    monkeypatch.setattr(evidence, "callback_root", root)
    qa_ref = write_json(tmp_path / "old_qa.json", dict(synthetic=True))
    source_ref = write_json(tmp_path / "frozen_source.json", dict(frozen=True))
    source_hashes = {source_ref["path"]: source_ref["sha256"]}

    def owned(stage, registration, result_ref, *, bad=False):
        directory = root(check=stage == "check", owner=True)
        reg_ref = write_json(directory / "registration.json", registration)
        entry = "check" if stage == "check" else "probe"
        command = [
            str(evidence.NATIVE_PYTHON),
            str(tmp_path / f"scripts/{entry}_tracking_v5_prefix_callback_backtracking.py"),
        ]
        if bad and defect == "wrong_launch":
            command[1] = str(tmp_path / "some_other_checker.py")
        launch = write_json(
            directory / "launch.json",
            dict(
                automatic_retry=False,
                selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                command=command,
                child=dict(pid=1234, start_ticks=5678),
            ),
        )
        exited = write_json(
            directory / "exit.json",
            dict(actual_wait=not (bad and defect == "no_wait"), returncode=0, automatic_retry=False),
        )
        release = write_json(
            directory / "release.json",
            dict(
                reaped=True,
                returncode=0,
                signal_sent=bad and defect == "signalled",
                original_error=None,
                pid=1234,
                start_ticks=5678,
            ),
        )
        owner = dict(
            complete=not (bad and defect == "live"),
            passed=not (bad and defect == "live"),
            case_name="candidate",
            stage=stage,
            registration_ref=reg_ref,
            result_ref=result_ref,
            owned_command=dict(launch=launch, exit=exited, release=release),
        )
        owner_ref = write_json(directory / "result.json", owner)
        return owner_ref, reg_ref

    normalized = np.tile([0.9991, 0.2], (12, 1))
    normalized[9:, 0] = 1.4
    if defect == "insufficient_reserve":
        normalized[4, 1] = 1.0 - 0.5e-4
    producer = dict(
        complete=True,
        passed=True,
        answer=dict(initial=dict(prefix_length=9, normalized_squared_errors=normalized.tolist())),
        source_refs_unchanged=[source_ref],
    )
    producer_ref = write_json(root() / "verified.json", producer)
    producer_owner_ref, _ = owned(
        "diagnose",
        dict(native_component_qa_ref=qa_ref, source_sha256=source_hashes),
        producer_ref,
    )
    report_reg = write_json(
        root(check=True) / "registration.json",
        dict(producer_ref=producer_ref, producer_owner_ref=producer_owner_ref),
    )
    report = dict(
        complete=True,
        passed=True,
        case_name="candidate",
        registration_ref=report_reg,
        response_evaluations=1012,
        verified_responses=1012,
        numerical_inner_steps=303600,
        query_count=315745,
        mj_forward_queries=315745,
        boundary_classification_disagreements=[],
        witness_found=False,
        control_qualified=False,
        complete_ray_rollouts_checked=75,
        accepted_anchor_audits_checked=13,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        source_refs_unchanged=[source_ref],
        ledger_check=dict(
            total_backtracking_attempts=75,
            total_backtracking_new_responses=60,
            accepted_anchor_indices=list(range(13)),
            witness_indices=[],
            termination_reason="trust_region_stalled",
        ),
    )
    for label, field, value in (
        ("partial", "verified_responses", 1011),
        ("missing_ray", "complete_ray_rollouts_checked", 74),
        ("missing_anchor", "accepted_anchor_audits_checked", 12),
        ("unbilled_query", "query_count", 315744),
        ("strict_disagreement", "boundary_classification_disagreements", [{}]),
        ("physical_step", "physics_steps", 1),
    ):
        if defect == label:
            report[field] = value
    if defect == "incomplete_ledger":
        report["ledger_check"]["total_backtracking_attempts"] = 74
    result_ref = write_json(root(check=True) / "verified.json", report)
    registration = dict(
        case_name="candidate",
        stage="check",
        native_component_qa_ref={} if defect == "changed_qa" else qa_ref,
        source_sha256=source_hashes,
        producer_owner_ref=producer_owner_ref,
        producer_ref=producer_ref,
    )
    owner_ref, check_registration_ref = owned("check", registration, result_ref, bad=True)
    if defect == "changed_registration":
        check_registration_ref = write_json(tmp_path / "another_registration.json", registration)
    if defect == "changed_source":
        Path(source_ref["path"]).write_text("changed after completion")
    if defect == "changed_producer":
        producer_ref = write_json(tmp_path / "another_producer.json", producer)
    for key, value in (
        ("CALLBACK_PRODUCER_REF", producer_ref),
        ("CALLBACK_OWNER_REF", producer_owner_ref),
        ("CALLBACK_CHECK_REG_REF", check_registration_ref),
        ("CALLBACK_QA_REF", qa_ref),
    ):
        monkeypatch.setattr(evidence, key, value)
    return dict(owner_ref=owner_ref, result_ref=result_ref, source_ref=source_ref)


def test_complete_callback_gate_requires_all_saved_work_and_original_warm_reserve(modules, monkeypatch, tmp_path):
    saved = callback_receipts(modules, monkeypatch, tmp_path)
    refs = {}
    result = modules.evidence.closed_callback_replay(refs)
    assert result["owner_ref"] == saved["owner_ref"] and result["result_ref"] == saved["result_ref"]
    assert result["minimum_initial_prefix_headroom"] == pytest.approx(0.0009)
    assert refs[saved["source_ref"]["path"]] == saved["source_ref"]


@pytest.mark.parametrize(
    "defect",
    [
        "live",
        "partial",
        "no_wait",
        "signalled",
        "changed_source",
        "changed_registration",
        "changed_producer",
        "changed_qa",
        "wrong_launch",
        "missing_ray",
        "missing_anchor",
        "unbilled_query",
        "strict_disagreement",
        "physical_step",
        "incomplete_ledger",
        "insufficient_reserve",
    ],
)
def test_incomplete_changed_or_unqualified_callback_evidence_cannot_admit_new_search(
    modules, monkeypatch, tmp_path, defect
):
    callback_receipts(modules, monkeypatch, tmp_path, defect=defect)
    with pytest.raises(ValueError):
        modules.evidence.closed_callback_replay({})


def test_new_solver_adapter_preserves_reset_warm_and_complete_response_accounting(modules, monkeypatch):
    source, warm, interface = source_fixture()
    source_before, warm_bytes = copy.deepcopy(source), warm.tobytes()
    dynamics, rollout_calls, solver_calls = CountingDynamics(), [], []
    original_solver = three_response_solver(source, dynamics, solver_calls)

    def solver(*args, **kwargs):
        assert kwargs["max_stages"] == kwargs["max_iterations"] == 60
        return original_solver(*args, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "s2a.prefix_optimization_reserve_plan",
        SimpleNamespace(solve_prefix_optimization_reserve_response=solver),
    )
    result = modules.producer.run_search(
        source, warm, interface, dynamics, rollout_fn=captured_rollout(source, rollout_calls)
    )
    assert solver_calls == [True] and len(rollout_calls) == result["response_evaluations"] == 3
    assert result["prediction_calls"] == 3 and result["query_count"] == dynamics.queries == 937
    assert result["numerical_inner_steps"] == 900
    assert rollout_calls[0]["commands"].tobytes() == warm_bytes == warm.tobytes()
    for row in rollout_calls:
        np.testing.assert_array_equal(row["q"], source_before["current_q"])
        np.testing.assert_array_equal(row["velocity"], source_before["current_velocity"])
        np.testing.assert_array_equal(row["latch"], source_before["latched_rotation_goal"])
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, source_before[key])


@pytest.mark.parametrize("stage,timeout", [("diagnose", 1200), ("check", 5400)])
def test_owner_launches_only_the_new_entry_with_its_registered_timeout(modules, monkeypatch, tmp_path, stage, timeout):
    run = owner_run(modules, monkeypatch, tmp_path, stage)
    run.run()
    assert len(run.calls) == 1 and run.calls[0]["timeout"] == timeout
    command = run.calls[0]["command"]
    assert command == modules.owner.entry_command(stage) and len(command) == 2
    assert "prefix_optimization_reserve.py" in command[1]
    environment = run.calls[0]["environment"]
    assert all(environment[key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in environment and "PYOPENGL_PLATFORM" not in environment
    result = json.loads((run.root(check=stage == "check", owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is True


def test_owner_rejects_changed_qa_before_starting_the_new_replay(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path, "check", different_qa=True)
    with pytest.raises(ValueError, match="identical frozen source-bound QA"):
        run.run()
    assert run.calls == []


def reserve_checker_run(modules, monkeypatch, tmp_path, *, near_cap_witness=False):
    checker = importlib.import_module("check_tracking_v5_prefix_optimization_reserve")
    # Reuse the old pure fake-response fixture while executing this new main.
    monkeypatch.setitem(sys.modules, "check_tracking_v5_prefix_callback_backtracking", checker)
    run = prior_tests.new_checker_run(modules, monkeypatch, tmp_path)
    metadata = sys.modules["s2a.prefix_callback_backtracking_evidence"]
    metadata.PREFIX_OPTIMIZATION_MARGIN = 1e-4
    monkeypatch.setitem(sys.modules, "s2a.prefix_optimization_reserve_evidence", metadata)
    owner, _ = metadata.closed_owner(None, None)
    metadata.read(owner["owned_command"]["launch"])["command"][1] = str(
        tmp_path / "scripts/probe_tracking_v5_prefix_optimization_reserve.py"
    )
    registration = metadata.read(run.producer["registration_ref"])
    registration.update(experiment="prefix_optimization_reserve", prefix_optimization_margin=1e-4)
    run.producer["prefix_optimization_margin"] = 1e-4
    answer = run.producer["answer"]
    answer["prefix_optimization_margin"] = 1e-4
    namespace = run.main.__globals__
    proof = dict(
        source_refs=[
            metadata.reference(tmp_path / name)
            for name in ("old_solver.py", "new_solver.py", "audit.py", "frozen_ledger.py")
        ],
        inverse_ast_equal=True,
        prefix_optimization_margin=1e-4,
        scope="Synthetic solver-only AST proof; no main or dynamics equivalence claim",
        allowed_changes=["optimizer-only prefix residual"],
    )
    proof_calls, ledger_calls = [], []

    def source_proof():
        proof_calls.append(True)
        return copy.deepcopy(proof)

    def ledger(actual, classifications, *, compare):
        assert actual is answer and len(classifications) == len(run.rows) == 2
        assert actual["prefix_optimization_margin"] == 1e-4
        ledger_calls.append(True)
        return dict(witness_indices=[0, 1] if near_cap_witness else [])

    namespace["optimization_reserve_source_proof"] = source_proof
    namespace["audit_optimization_reserve_ledger"] = ledger
    if near_cap_witness:
        # Deliberately misses the optimizer reserve while satisfying original caps.
        error = 0.01 * np.sqrt(1.0 - 0.5e-4)
        errors = np.tile([error, 0, 0, 0, 0, 0], (12, 1))
        normalized = np.tile([(error / 0.01) ** 2, 0], (12, 1))
        metrics = dict(
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            prefix_length=12,
            feasible=True,
            joint_feasible=True,
            safety_min=0.1,
            safety_slack=np.array([0.1]),
            maximum_normalized_squared_error=float(normalized.max()),
            maximum_pose_cap_violation=0.0,
            pose_loss=float(normalized.sum()),
        )
        records = [answer[key] for key in ("initial", "selected", "closest_safety_attempt")]
        records += answer["evaluated_candidates"] + answer["accepted_anchors"]
        records += [attempt["entry"] for attempt in run.attempts if attempt["entry"] is not None]
        for record in records:
            for key in record.keys() & metrics.keys():
                record[key] = copy.deepcopy(metrics[key])
        original_engine = namespace["case_engine"]

        def engine(*args):
            values = list(original_engine(*args))
            values[2] = lambda commands, rollout: copy.deepcopy(metrics)
            return tuple(values)

        namespace["case_engine"] = engine
    run.metadata, run.registration, run.proof = metadata, registration, proof
    run.proof_calls, run.ledger_calls = proof_calls, ledger_calls
    return run


def test_main_binds_new_solver_proof_and_checks_all_rays_without_tightening_original_caps(
    modules, monkeypatch, tmp_path
):
    run = reserve_checker_run(modules, monkeypatch, tmp_path, near_cap_witness=True)
    run.main()
    registration, result = run.published[0][1], run.published[-1][1]
    assert result["complete"] is result["passed"] is result["witness_found"] is True
    assert result["prefix_optimization_margin"] == 1e-4
    assert run.proof_calls == run.ledger_calls == [True]
    assert result["optimization_reserve_source_proof"] == registration["optimization_reserve_source_proof"] == run.proof
    assert all(ref in registration["input_refs"] for ref in run.proof["source_refs"])
    assert all(ref in result["source_refs_unchanged"] for ref in run.proof["source_refs"])
    assert "main_ast_proof" not in result and "main_ast_proof" not in run.proof
    assert result["response_evaluations"] == result["verified_responses"] == 2
    assert result["complete_ray_rollouts_checked"] == 3 and len(run.audits) == 7
    assert result["numerical_inner_steps"] == 600 and result["query_count"] == 625
    assert result["control_qualified"] is False


def test_dry_checker_binds_the_new_proof_without_querying_or_publishing(modules, monkeypatch, tmp_path, capsys):
    run = reserve_checker_run(modules, monkeypatch, tmp_path)
    run.main(dry=True)
    result = json.loads(capsys.readouterr().out)
    assert run.proof_calls == [True] and run.ledger_calls == []
    assert result["optimization_reserve_source_proof"] == run.proof
    assert all(ref in run.finishes[-1].values() for ref in run.proof["source_refs"])
    assert result["new_queries"] == result["new_responses"] == 0
    assert run.rows == run.queries == run.published == []


@pytest.mark.parametrize("location", ["registration", "producer", "answer"])
def test_changed_reserve_metadata_is_rejected_before_any_response(modules, monkeypatch, tmp_path, location):
    run = reserve_checker_run(modules, monkeypatch, tmp_path)
    target = {"registration": run.registration, "producer": run.producer, "answer": run.producer["answer"]}[location]
    target["prefix_optimization_margin"] = 1e-5
    with pytest.raises(ValueError):
        run.main()
    assert run.rows == run.queries == run.published == []


def test_source_ast_failure_cannot_fall_back_to_an_old_main_proof(modules, monkeypatch, tmp_path):
    run = reserve_checker_run(modules, monkeypatch, tmp_path)

    def failed_proof():
        raise ValueError("Synthetic new solver AST mismatch")

    run.main.__globals__["optimization_reserve_source_proof"] = failed_proof
    with pytest.raises(ValueError, match="new solver AST mismatch"):
        run.main(dry=True)
    assert run.rows == run.queries == run.published == []


def test_new_checker_still_rejects_a_tampered_cached_ray_full_record(modules, monkeypatch, tmp_path):
    run = reserve_checker_run(modules, monkeypatch, tmp_path)
    run.attempts[1]["entry"]["rollout"] = dict(index=99)
    with pytest.raises(AssertionError):
        run.main()
    result = run.published[-1][1]
    assert result["complete"] is result["passed"] is False
    assert result["response_evaluations"] == 1 and result["query_count"] == 313
    assert run.ledger_calls == []


@pytest.mark.parametrize("defect", [None, "missing_source", "missing_test"])
def test_new_owner_requires_the_extended_native_qa_roster(modules, monkeypatch, tmp_path, defect):
    callback_owner = importlib.import_module("execute_tracking_v5_prefix_callback_backtracking")
    monkeypatch.setattr(callback_owner, "REPO", tmp_path)
    monkeypatch.setattr(callback_owner, "NATIVE_PYTHON", tmp_path / "python")
    prior_tests.test_native_qa_covers_the_complete_new_source_and_test_roster(modules, monkeypatch, tmp_path, defect)

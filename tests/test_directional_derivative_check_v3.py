"""Synthetic check-only ownership and frozen producer/new-QA provenance."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_directional_derivative_execution as prior
from test_accepted_point_sqp_execution import native_qa
from test_prefix_callback_backtracking_execution import write_json


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return SimpleNamespace(
        owner=importlib.import_module("execute_tracking_v5_directional_derivative_check_v3"),
        evidence=importlib.import_module("s2a.directional_derivative_check_v3_evidence"),
    )


def checker_receipt(count=77):
    report = prior.receipt("check", count)
    report.update(saved_stencil_check=dict(passed=True), historical_anchor_check=dict(passed=True))
    return report


def owner_run(modules, monkeypatch, tmp_path, *, defect=None):
    owner = modules.owner
    monkeypatch.setattr(owner, "REPO", tmp_path)
    monkeypatch.setattr(owner, "NATIVE_PYTHON", tmp_path / "native_python")
    plan = write_json(tmp_path / "v3plan.json", {})
    old_plan = write_json(tmp_path / "v2plan.json", {})
    monkeypatch.setattr(owner, "PLAN", Path(plan["path"]))
    monkeypatch.setattr(owner, "PRODUCER_PLAN", Path(old_plan["path"]))
    monkeypatch.setattr(owner, "strict_native_environment", lambda: None)
    for key, value in owner.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MUJOCO_GL", "forbidden")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "forbidden")

    def root(*, check=False, owner=False):
        return tmp_path / (("v3_check" if check else "v2_producer") + ("_owner" if owner else ""))

    monkeypatch.setattr(owner, "case_root", root)
    sources = {path: write_json(tmp_path / path, dict(frozen=path))["sha256"] for path in owner.PRODUCER_QA_SOURCES}
    log = write_json(tmp_path / "v2qa/pytest.log", {})
    old_qa = dict(source_sha256=sources, log_sha256=log["sha256"])
    old_qa_ref = write_json(tmp_path / "v2qa/result.json", old_qa)
    monkeypatch.setattr(owner, "PRODUCER_QA_REF", old_qa_ref)
    new_sources = dict(sources)
    new_sources.update(
        {
            path: write_json(tmp_path / path, dict(new_source=path))["sha256"]
            for path in set(owner.QA_SOURCES) - set(sources)
        }
    )
    new_log = write_json(tmp_path / "v3qa/pytest.log", {})
    new_qa = dict(source_sha256=new_sources, log_sha256=new_log["sha256"])
    if defect == "different_sources":
        new_qa["source_sha256"][next(iter(sources))] = "0" * 64
    qa_ref = write_json(tmp_path / "v3qa/result.json", new_qa)
    assert qa_ref != old_qa_ref  # Different QA is intentional, subject to the frozen subset gate.
    monkeypatch.setattr(owner, "verify_native_qa", lambda ref: owner.read(ref))
    monkeypatch.setattr(owner, "verify_v2_qa", lambda ref: owner.read(ref))
    monkeypatch.setattr(owner, "load_prerequisites", lambda: ({}, {}, {}, {}, {}))
    producer = prior.receipt("diagnose", 77)
    source_ref = dict(path=str(tmp_path / next(iter(sources))), sha256=next(iter(sources.values())))
    producer["source_refs_unchanged"] = [source_ref]
    producer_ref = write_json(root() / "verified.json", producer)
    registration = dict(
        experiment=owner.EXPERIMENT,
        stage="diagnose",
        plan_ref=old_plan,
        native_component_qa_ref=qa_ref if defect == "wrong_old_qa" else old_qa_ref,
        source_sha256={str(tmp_path / path): digest for path, digest in sources.items()},
    )
    predecessor = prior.owned_metadata(owner, root(owner=True), "diagnose", producer_ref, registration, defect=defect)
    if defect == "wrong_launch":
        predecessor["owned_command"]["launch"] = write_json(
            root(owner=True) / "launch.json",
            dict(
                automatic_retry=False,
                selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                command=owner.entry_command("check"),
                child=dict(pid=1234, start_ticks=5678),
            ),
        )
    predecessor_ref = write_json(root(owner=True) / "result.json", predecessor)
    monkeypatch.setattr(owner, "PRODUCER_OWNER_REF", predecessor_ref)
    monkeypatch.setattr(owner, "PRODUCER_REF", producer_ref)
    if defect == "changed_source":
        Path(source_ref["path"]).write_text("changed after producer closure")
    calls = []

    def child(command, environment, capture, *, timeout):
        calls.append(dict(command=command, environment=environment, timeout=timeout))
        if defect == "child_failure":
            raise ChildProcessError("synthetic checker exit1; no automatic retry")
        report = checker_receipt(76 if defect == "partial" else 77)
        if defect in ("stencil_failed", "historical_failed"):
            report["saved_stencil_check" if defect == "stencil_failed" else "historical_anchor_check"]["passed"] = False
        report_ref = write_json(root(check=True) / "verified.json", report)
        lifecycle = prior.owned_metadata(owner, capture, "check", report_ref, {})["owned_command"]
        if defect == "source_changed_during_child":
            Path(source_ref["path"]).write_text("changed during checker")
        return lifecycle

    monkeypatch.setattr(owner, "owned_command", child)
    return SimpleNamespace(
        run=lambda dry=False: owner.main(qa_ref, dry), calls=calls, root=root, qa_ref=qa_ref, old_qa_ref=old_qa_ref
    )


def test_check_only_owner_accepts_distinct_qa_with_identical68_source_subset(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path)
    run.run()
    assert len(run.calls) == 1
    call = run.calls[0]
    assert call["timeout"] == 1800
    assert call["command"] == modules.owner.entry_command("check")
    assert call["command"][1].endswith("/check_tracking_v5_directional_derivative_v3.py")
    assert modules.owner.entry_command("diagnose")[1].endswith("/probe_tracking_v5_directional_derivative_v2.py")
    assert all(call["environment"][key] == value for key, value in modules.owner.CPU_ENVIRONMENT.items())
    assert "MUJOCO_GL" not in call["environment"] and "PYOPENGL_PLATFORM" not in call["environment"]
    owner_ref = modules.owner.reference(run.root(check=True, owner=True) / "result.json")
    owner, report = modules.owner.closed_owner(owner_ref, {})
    assert owner["complete"] is report["complete"] is True
    registration = modules.owner.read(owner["registration_ref"])
    assert registration["stage"] == "check" and registration["automatic_retry"] is False
    assert registration["native_component_qa_ref"] == run.qa_ref
    assert registration["producer_native_component_qa_ref"] == run.old_qa_ref
    assert registration["producer_qa_subset_count"] == 68
    assert registration["expected_responses"] == 77 and registration["expected_unique_commands"] == 76


@pytest.mark.parametrize(
    "defect",
    [
        "different_sources",
        "wrong_old_qa",
        "no_wait",
        "not_reaped",
        "signal",
        "changed_source",
        "wrong_launch",
    ],
)
def test_original_producer_identity_lifecycle_and_frozen_subset_gate_precede_launch(
    modules, monkeypatch, tmp_path, defect
):
    run = owner_run(modules, monkeypatch, tmp_path, defect=defect)
    with pytest.raises(ValueError):
        run.run()
    assert run.calls == [] and not run.root(check=True, owner=True).exists()


@pytest.mark.parametrize(
    "defect", ["partial", "child_failure", "source_changed_during_child", "stencil_failed", "historical_failed"]
)
def test_no_partial_replay_missing_audit_or_automatic_retry(modules, monkeypatch, tmp_path, defect):
    run = owner_run(modules, monkeypatch, tmp_path, defect=defect)
    with pytest.raises((ValueError, ChildProcessError)):
        run.run()
    assert len(run.calls) == 1 and run.calls[0]["command"] == modules.owner.entry_command("check")
    result = json.loads((run.root(check=True, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and result["error"]


def test_dry_validates_all_metadata_without_launch_or_new_output(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path)
    run.run(dry=True)
    assert run.calls == [] and not run.root(check=True, owner=True).exists()


def test_failed_predecessor_closure_is_required_before_new_checker_launch(modules, monkeypatch, tmp_path):
    run = owner_run(modules, monkeypatch, tmp_path)

    def not_closed():
        raise ValueError("original failed checker requires actual wait/reap")

    monkeypatch.setattr(modules.owner, "load_prerequisites", not_closed)
    with pytest.raises(ValueError, match="actual wait/reap"):
        run.run()
    assert run.calls == []


@pytest.mark.parametrize("directory", ["owner", "result"])
def test_v3_output_is_exclusive(modules, monkeypatch, tmp_path, directory):
    run = owner_run(modules, monkeypatch, tmp_path)
    run.root(check=True, owner=directory == "owner").mkdir()
    with pytest.raises(ValueError, match="no overwrite or retry"):
        run.run()
    assert run.calls == []


def native_qa_pair(modules, monkeypatch, tmp_path):
    qa = native_qa(modules, monkeypatch, tmp_path)
    for name in (
        "execute_tracking_v5_directional_derivative_v2",
        "execute_tracking_v5_directional_derivative",
        "execute_tracking_v5_accepted_point_sqp",
    ):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "NATIVE_PYTHON", tmp_path / "native_python")
    old = copy.deepcopy(qa)
    old["source_sha256"] = {path: qa["source_sha256"][path] for path in modules.owner.PRODUCER_QA_SOURCES}
    old["command"] = [
        str(tmp_path / "native_python"),
        "-m",
        "pytest",
        *importlib.import_module("execute_tracking_v5_directional_derivative_v2").QA_TESTS,
    ]
    old_log = write_json(tmp_path / "old_qa/pytest.log", {})
    old["log_sha256"] = old_log["sha256"]
    old_ref = write_json(tmp_path / "old_qa/result.json", old)
    monkeypatch.setattr(modules.owner, "PRODUCER_QA_REF", old_ref)
    return qa, old


def test_native_qa_requires75_sources23_modules_and_original68_digest_subset(modules, monkeypatch, tmp_path):
    qa, old = native_qa_pair(modules, monkeypatch, tmp_path)
    owner = modules.owner
    assert len(owner.QA_SOURCES) == 75 and len(owner.QA_TESTS) == 23
    new_sources = set(owner.QA_SOURCES) - set(owner.PRODUCER_QA_SOURCES)
    old_tests = importlib.import_module("execute_tracking_v5_directional_derivative_v2").QA_TESTS
    new_tests = set(owner.QA_TESTS) - set(old_tests)
    assert len(new_sources) == 7 and len(new_tests) == 2
    target = tmp_path / "qa/result.json"
    assert owner.verify_native_qa(write_json(target, qa)) == qa
    assert owner.verify_producer_qa_subset(qa, {}) == old
    for source in new_sources:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][source]
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    for test in new_tests:
        changed = copy.deepcopy(qa)
        changed["command"].remove(test)
        with pytest.raises(ValueError):
            owner.verify_native_qa(write_json(target, changed))
    changed = copy.deepcopy(qa)
    changed["source_sha256"][owner.PRODUCER_QA_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="identical digest subset"):
        owner.verify_producer_qa_subset(changed, {})
    qa_ref = write_json(target, qa)
    (tmp_path / sorted(new_sources)[0]).write_text("new checker source changed after QA")
    with pytest.raises(ValueError):
        owner.verify_native_qa(qa_ref)
    assert len(importlib.import_module("execute_tracking_v5_directional_derivative_v2").QA_SOURCES) == 68


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified_responses", 76),
        ("repeated_responses_checked", 0),
        ("unique_command_count", 77),
        ("actual_consumed_reference_count", 1),
        ("control_qualified", True),
    ],
)
def test_original_receipt_guards_are_preserved(modules, field, value):
    report = checker_receipt()
    report[field] = value
    with pytest.raises(ValueError):
        modules.owner.validate_receipt(report)


def preserved_failure_fixture(modules, monkeypatch, tmp_path, defect):
    """Real small file hashes model the failed owner with no owned_command key."""
    evidence = modules.evidence
    monkeypatch.setattr(evidence, "REPO", tmp_path)
    monkeypatch.setattr(evidence, "NATIVE_PYTHON", tmp_path / "native_python")
    plan = write_json(tmp_path / "old_plan.json", {})
    monkeypatch.setattr(evidence, "PRODUCER_PLAN", Path(plan["path"]))
    producer = write_json(tmp_path / "closed_producer.json", {})
    producer_owner = write_json(tmp_path / "closed_producer_owner.json", {})
    monkeypatch.setattr(evidence, "PRODUCER_REF", producer)
    monkeypatch.setattr(evidence, "PRODUCER_OWNER_REF", producer_owner)
    sources = {path: write_json(tmp_path / path, dict(frozen=path))["sha256"] for path in evidence.PRODUCER_QA_SOURCES}
    log = write_json(tmp_path / "qa/pytest.log", {})
    qa = write_json(
        tmp_path / "qa/result.json",
        dict(
            complete=True,
            passed=True,
            source_unchanged=True,
            actual_wait=True,
            returncode=0,
            source_sha256=sources,
            log_sha256=log["sha256"],
        ),
    )
    monkeypatch.setattr(evidence, "PRODUCER_QA_REF", qa)
    common = dict(producer_ref=producer, producer_owner_ref=producer_owner, plan_ref=plan)
    registration = write_json(tmp_path / "check/registration.json", dict(common, source_refs=[], input_refs=[]))
    owner_registration = write_json(
        tmp_path / "owner/registration.json",
        dict(
            common,
            stage="check",
            timeout_seconds=1800,
            native_component_qa_ref=qa,
            native_component_qa_log_ref=log,
            source_sha256={str(tmp_path / path): digest for path, digest in sources.items()},
        ),
    )
    failed = dict(
        complete=False,
        passed=False,
        error="ValueError('Anchor inside old stage bounds')",
        registration_ref=registration,
        response_evaluations=77,
        verified_responses=77,
        unique_command_count=76,
        numerical_inner_steps=23100,
        query_count=24025,
        mj_forward_queries=24025,
        ledger_check=dict(passed=True),
        boundary_classification_disagreements=[],
        control_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
    )
    if defect == "partial":
        failed["verified_responses"] = 76
    elif defect == "premature_stencil":
        failed["saved_stencil_check"] = dict(passed=True)
    elif defect == "nonzero_physics":
        failed["physics_steps"] = 1
    failed_ref = write_json(tmp_path / "check/verified.json", failed)
    monkeypatch.setattr(evidence, "FAILED_CHECK_REF", failed_ref)
    failed_owner = dict(
        complete=False,
        passed=False,
        stage="check",
        registration_ref=owner_registration,
        error="ChildProcessError('025 command failed with1; no automatic retry')",
    )
    assert "owned_command" not in failed_owner
    monkeypatch.setattr(evidence, "FAILED_OWNER_REF", write_json(tmp_path / "owner/result.json", failed_owner))
    launch = write_json(
        tmp_path / "owner/capture/launch.json",
        dict(
            command=[
                str(evidence.NATIVE_PYTHON),
                str(tmp_path / "scripts/check_tracking_v5_directional_derivative_v2.py"),
            ],
            selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            automatic_retry=False,
            child=dict(pid=205307, start_ticks=616309119, ppid=201498),
        ),
    )
    exited = write_json(
        tmp_path / "owner/capture/exit.json",
        dict(
            actual_wait=defect != "no_wait",
            returncode=0 if defect == "wrong_exit" else 1,
            automatic_retry=False,
        ),
    )
    release = write_json(
        tmp_path / "owner/capture/ownership/release.json",
        dict(
            pid=205307,
            start_ticks=616309119,
            reaped=defect != "not_reaped",
            returncode=1,
            signal_sent=defect == "signal",
            original_error=failed_owner["error"],
        ),
    )
    monkeypatch.setattr(evidence, "FAILED_CAPTURE_REFS", (launch, exited, release))
    if defect == "source_changed":
        (tmp_path / next(iter(sources))).write_text("frozen source changed after the failed replay")
    return (failed_ref, launch, exited, release)


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "no_wait",
        "not_reaped",
        "signal",
        "wrong_exit",
        "partial",
        "source_changed",
        "premature_stencil",
        "nonzero_physics",
    ],
)
def test_preserved_failed_capture_is_closed_and_never_promoted_to_qualification(modules, monkeypatch, tmp_path, defect):
    expected = preserved_failure_fixture(modules, monkeypatch, tmp_path, defect)
    if defect is not None:
        with pytest.raises(ValueError):
            modules.evidence.verify_preserved_failure()
    else:
        refs = modules.evidence.verify_preserved_failure()
        assert all(refs[ref["path"]] == ref for ref in expected)
        failed = modules.evidence.read(expected[0])
        assert failed["complete"] is failed["passed"] is False
        assert "source_refs_unchanged" not in failed and "saved_stencil_check" not in failed

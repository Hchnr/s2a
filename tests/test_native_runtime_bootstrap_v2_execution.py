"""Repair CLI provenance and publication using synthetic files and callbacks only."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_native_runtime_bootstrap_execution as original_tests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_native_runtime_bootstrap_v2.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("_synthetic_native_bootstrap_v2", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.fixture
def admitted_v2(monkeypatch, tmp_path):
    """Reuse the original file admission fixture, adding a closed failed parent."""
    import s2a.native_runtime_bootstrap as old_core
    import s2a.native_runtime_bootstrap_v2 as core

    base = original_tests.execution.__wrapped__(monkeypatch, tmp_path)
    cli = load_cli()
    for name in ("REPO", "NATIVE_PYTHON"):
        monkeypatch.setattr(cli, name, getattr(base.cli, name))
    for name, relative in (
        ("CONFIG", "configs/native_runtime_bootstrap_v2_20261029.json"),
        ("PLAN", "plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md"),
        ("RESULT", "outputs/native_runtime_bootstrap_v2_20261029/result.json"),
    ):
        monkeypatch.setattr(cli, name, base.repo / relative)
    for relative in cli.NEW_SOURCES:
        base.write(base.repo / relative, f"synthetic repair source: {relative}".encode())
    base.cli = cli
    old_run = old_core.run_native_bootstrap
    admitted = original_tests.admitted.__wrapped__(base, monkeypatch)
    fake_run = old_core.run_native_bootstrap
    monkeypatch.setattr(old_core, "run_native_bootstrap", old_run)
    monkeypatch.setattr(core, "run_native_bootstrap", fake_run)

    parent_dir = base.repo / "outputs/failed_parent"
    old_config = copy.deepcopy(admitted.config)
    old_config["output_dir"] = str(base.repo / "outputs/native_runtime_bootstrap_20261029")
    config_ref = base.write(base.repo / "configs/native_runtime_bootstrap_20261029.json", old_config)
    prior_qa = copy.deepcopy(base.qa)
    prior_qa["source_sha256"] = {path: cli.reference(base.repo / path)["sha256"] for path in cli.PRIOR_SOURCE_PATHS}
    prior_qa_ref = base.write(parent_dir / "qa.json", prior_qa)
    admission_ref = base.write(parent_dir / "admission.json", dict(complete=True, passed=True, qa_ref=prior_qa_ref))
    counters = {name: dict(attempted=0, completed=0) for name in core.LIMITS}
    counters["configure"] = dict(attempted=1, completed=1)
    counters["model_construct"] = dict(attempted=1, completed=0)
    failed = dict(
        complete=False,
        passed=False,
        bootstrap_verified=False,
        runtime_loaded=True,
        source_unchanged=True,
        failure_stage="constructor",
        qa_ref=prior_qa_ref,
        config_ref=config_ref,
        bootstrap=dict(
            counters=counters,
            error=dict(message="non-absolute or traversing path: /synthetic/assets/scenes/../textures/tile.png"),
        ),
    )
    failed_ref = base.write(base.repo / "outputs/native_runtime_bootstrap_20261029/result.json", failed)
    owner = dict(
        complete=False,
        passed=False,
        automatic_retry=False,
        control_qualified=False,
        environment=dict(cli.CPU_ENVIRONMENT),
        qa_ref=prior_qa_ref,
        admission_ref=admission_ref,
        commands=[
            dict(
                actual_wait=True,
                reaped=True,
                returncode=1,
                signal_sent=False,
                automatic_retry=False,
                timeout_seconds=600,
                child_pid=12345,
                command=[
                    str(cli.NATIVE_PYTHON),
                    "scripts/verify_native_runtime_bootstrap.py",
                    "--qa-ref",
                    prior_qa_ref["path"],
                    "--qa-sha256",
                    prior_qa_ref["sha256"],
                ],
                log_ref=base.write(parent_dir / "owner.log", b"synthetic closed failure, no runtime launched"),
            )
        ],
    )
    parent_refs = dict(
        original_config_ref=config_ref,
        qa_ref=prior_qa_ref,
        admission_ref=admission_ref,
        result_ref=failed_ref,
        owner_ref=base.write(parent_dir / "owner.json", owner),
    )
    monkeypatch.setattr(cli, "PARENT_REFS", parent_refs)
    admitted.config["repair_parent"] = copy.deepcopy(parent_refs)
    base.write(cli.CONFIG, admitted.config)
    qa = copy.deepcopy(base.qa)
    qa.update(
        source_count=23,
        test_module_count=7,
        source_sha256={path: cli.reference(base.repo / path)["sha256"] for path in cli.SOURCE_PATHS},
        command=[str(cli.NATIVE_PYTHON), "-m", "pytest", "-q", *cli.QA_TESTS, f"--basetemp={base.repo}/outputs/qa_tmp"],
    )
    values = dict(vars(admitted))
    values.update(
        core=core,
        old_core=old_core,
        qa=qa,
        qa_ref=base.write(base.repo / "outputs/qa_v2/result.json", qa),
        parent_refs=parent_refs,
        failed=failed,
        owner=owner,
        prior_qa=prior_qa,
        old_result_bytes=Path(failed_ref["path"]).read_bytes(),
    )
    return SimpleNamespace(**values)


def test_finite_inverse_proof_reuses_frozen_cli_with_separate_globals():
    cli = load_cli()
    proof = cli.compile_cli_repair_proof()
    assert proof["passed"] and proof["inherited_cli_inverse_ast_equal"]
    assert proof["new_admission_wrappers_claimed_identical"] is False
    assert cli._inherited_main.__globals__ is vars(cli)
    assert len(cli.SOURCE_PATHS) == len(set(cli.SOURCE_PATHS)) == 23
    assert len(cli.QA_TESTS) == len(set(cli.QA_TESTS)) == 7
    original = importlib.util.spec_from_file_location("_unchanged_original_cli", original_tests.SCRIPT)
    old_cli = importlib.util.module_from_spec(original)
    original.loader.exec_module(old_cli)
    assert old_cli.CONFIG != cli.CONFIG and len(old_cli.SOURCE_PATHS) == 17
    assert old_cli.main.__globals__ is not cli._inherited_main.__globals__


@pytest.mark.parametrize("mutation", ["budget", "core_import", "receipt_binding"])
def test_inverse_proof_rejects_changes_outside_declared_repair(mutation):
    cli = load_cli()
    tree = cli.repaired_cli_ast()
    if mutation == "budget":
        next(node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == 600).value = 601
    elif mutation == "core_import":
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == cli.CORE_MODULE + "_v2"
        ).module = cli.CORE_MODULE
    else:
        call = next(iter(cli._receipt_calls(tree)))
        next(keyword for keyword in call.keywords if keyword.arg == "repair_parent").value = ast.Constant(True)
    with pytest.raises(ValueError, match="AST|four|metadata"):
        cli.compile_cli_repair_proof(tree)


def test_dry_admits_preserved_failure_and_never_dispatches_runtime(admitted_v2):
    a = admitted_v2
    result = a.cli.main(a.qa_ref, dry=True)
    assert result["passed"] and result["environment_initializations"] == 0
    assert result["repair_parent"]["original_source_count"] == 17
    assert result["repair_parent"]["returncode"] == 1
    assert result["cli_repair_proof"]["passed"] and result["core_repair_proof"]["passed"]
    assert a.calls == [] and not a.cli.RESULT.parent.exists()
    assert Path(a.parent_refs["result_ref"]["path"]).read_bytes() == a.old_result_bytes


def test_qa_requires_every_new_test_and_preserves_old_source_bytes(admitted_v2):
    a = admitted_v2
    assert a.cli.verify_native_qa(a.qa_ref) == a.qa
    changed = copy.deepcopy(a.qa)
    changed["command"].remove("tests/test_native_runtime_bootstrap_v2_execution.py")
    with pytest.raises(ValueError, match="test module"):
        a.cli.verify_native_qa(a.write(a.repo / "outputs/qa_v2/changed.json", changed))
    path = a.cli.PRIOR_SOURCE_PATHS[0]
    a.write(a.repo / path, b"changed original source")
    changed = copy.deepcopy(a.qa)
    changed["source_sha256"][path] = a.cli.reference(a.repo / path)["sha256"]
    with pytest.raises(ValueError, match="SHA mismatch"):
        a.cli.verify_native_qa(a.write(a.repo / "outputs/qa_v2/changed.json", changed))
    assert a.calls == []


@pytest.mark.parametrize("mutation", ["unreaped", "successful_parent", "compiled_model"])
def test_incomplete_or_wrong_parent_is_rejected_before_runtime(admitted_v2, mutation):
    a = admitted_v2
    if mutation == "compiled_model":
        a.failed["bootstrap"]["counters"]["model_construct"]["completed"] = 1
        key, value = "result_ref", a.failed
    else:
        a.owner["commands"][0]["reaped" if mutation == "unreaped" else "returncode"] = (
            False if mutation == "unreaped" else 0
        )
        key, value = "owner_ref", a.owner
    a.parent_refs[key] = a.write(a.parent_refs[key]["path"], value)
    a.config["repair_parent"] = copy.deepcopy(a.parent_refs)
    with pytest.raises(ValueError, match="failed and was reaped|no compiled model"):
        a.cli.inspect_inputs(a.config)
    assert a.calls == []


def test_configuration_repair_cannot_change_scope_or_redirect_output(admitted_v2):
    a = admitted_v2
    changed = copy.deepcopy(a.config)
    changed["timeout_seconds"] = 601
    with pytest.raises(ValueError, match="Only output directory"):
        a.cli.inspect_inputs(changed)
    changed = copy.deepcopy(a.config)
    changed["output_dir"] = str(a.repo / "outputs/alternate")
    with pytest.raises(ValueError, match="Fixed bootstrap result"):
        a.cli.inspect_inputs(changed)
    assert a.calls == []


def test_success_dispatches_only_v2_and_preserves_old_failure(admitted_v2, monkeypatch):
    a = admitted_v2

    def forbidden(*args, **kwargs):
        raise AssertionError("old bootstrap must never be dispatched")

    monkeypatch.setattr(a.old_core, "run_native_bootstrap", forbidden)
    result = a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "bootstrap"]
    assert result["complete"] is result["passed"] is result["source_unchanged"] is True
    assert result["control_qualified"] is result["historical_runtime_equivalence_claimed"] is False
    assert result["bootstrap"] == a.receipt
    assert result["repair_parent"]["result_ref"] == a.parent_refs["result_ref"]
    assert result["cli_repair_proof"]["passed"] and result["core_repair_proof"]["passed"]
    assert json.loads(a.cli.RESULT.read_text()) == result
    assert Path(a.parent_refs["result_ref"]["path"]).read_bytes() == a.old_result_bytes
    with pytest.raises(ValueError, match="exclusive"):
        a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "bootstrap"]


def test_failed_v2_keeps_partial_receipt_without_retry_or_overwriting_parent(admitted_v2, monkeypatch):
    a = admitted_v2
    partial = copy.deepcopy(a.receipt)
    partial.update(complete=False, passed=False, bootstrap_verified=False, failure_stage="hard_reset")
    partial["counters"]["model_construct"] = dict(attempted=2, completed=1)

    def fail(*args, **kwargs):
        a.calls.append("bootstrap")
        raise a.core.BootstrapFailure("synthetic new reset failure", partial)

    monkeypatch.setattr(a.core, "run_native_bootstrap", fail)
    with pytest.raises(a.core.BootstrapFailure, match="synthetic new reset failure"):
        a.cli.main(a.qa_ref)
    result = json.loads(a.cli.RESULT.read_text())
    assert result["bootstrap"] == partial and result["failure_stage"] == "hard_reset"
    assert result["complete"] is result["passed"] is result["bootstrap_verified"] is False
    assert result["runtime_loaded"] is result["source_unchanged"] is True
    assert result["repair_parent"]["passed"] and result["core_repair_proof"]["passed"]
    assert Path(a.parent_refs["result_ref"]["path"]).read_bytes() == a.old_result_bytes
    assert a.calls == ["cv2", "bootstrap"]

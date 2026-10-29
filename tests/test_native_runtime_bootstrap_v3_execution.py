"""Finite v3 CLI wiring with two preserved failures and no real runtime."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_native_runtime_bootstrap_v2_execution as v2_tests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_native_runtime_bootstrap_v3.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("_synthetic_native_bootstrap_v3", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.fixture
def admitted_v3(monkeypatch, tmp_path):
    import s2a.native_runtime_bootstrap_v3 as core

    previous = v2_tests.admitted_v2.__wrapped__(monkeypatch, tmp_path)
    cli = load_cli()
    for name in ("REPO", "NATIVE_PYTHON", "restore_cv2_environment", "verify_runtime_snapshot"):
        monkeypatch.setattr(cli, name, getattr(previous.cli, name))
    for name, relative in (
        ("CONFIG", "configs/native_runtime_bootstrap_v3_20261029.json"),
        ("PLAN", "plans/v0.0.1/native_runtime_bootstrap_v3_20261029.md"),
        ("RESULT", "outputs/native_runtime_bootstrap_v3_20261029/result.json"),
    ):
        monkeypatch.setattr(cli, name, previous.repo / relative)
    for relative in cli.NEW_SOURCES:
        previous.write(previous.repo / relative, f"synthetic v3 source: {relative}".encode())
    parent_dir = previous.repo / "outputs/failed_v2"
    admission_ref = previous.write(
        parent_dir / "admission.json", dict(complete=True, passed=True, qa_ref=previous.qa_ref)
    )
    counters = {name: dict(attempted=0, completed=0) for name in core.LIMITS}
    for name, count in dict(configure=1, model_construct=2, context=2, forward=9, step1=2).items():
        counters[name] = dict(attempted=count, completed=count)
    failed = dict(
        complete=False,
        passed=False,
        bootstrap_verified=False,
        runtime_loaded=True,
        source_unchanged=True,
        failure_stage="record_xml",
        qa_ref=previous.qa_ref,
        config_ref=cli.reference(previous.cli.CONFIG),
        bootstrap=dict(
            counters=counters,
            failure_stage="record_xml",
            error=dict(type="ValueError", message="invalid declared XML path"),
            cleanup_complete=True,
            cleanup_close_called=True,
            cleanup_outcome="closed",
            cleanup_target="outer",
            time_checks=[dict(operation=name, before=0.0, after=0.0) for name in ["forward"] * 9 + ["step1"] * 2],
        ),
    )
    failed_ref = previous.write(previous.cli.RESULT, failed)
    owner = dict(
        complete=False,
        passed=False,
        automatic_retry=False,
        control_qualified=False,
        environment=dict(cli.CPU_ENVIRONMENT),
        qa_ref=previous.qa_ref,
        admission_ref=admission_ref,
        commands=[
            dict(
                actual_wait=True,
                reaped=True,
                returncode=1,
                signal_sent=False,
                automatic_retry=False,
                timeout_seconds=600,
                child_pid=23456,
                command=[
                    str(cli.NATIVE_PYTHON),
                    "scripts/verify_native_runtime_bootstrap_v2.py",
                    "--qa-ref",
                    previous.qa_ref["path"],
                    "--qa-sha256",
                    previous.qa_ref["sha256"],
                ],
                log_ref=previous.write(parent_dir / "owner.log", b"synthetic v2 failure; no runtime loaded"),
            )
        ],
    )
    predecessor_refs = dict(
        original_config_ref=cli.reference(previous.cli.CONFIG),
        qa_ref=previous.qa_ref,
        admission_ref=admission_ref,
        result_ref=failed_ref,
        owner_ref=previous.write(parent_dir / "owner.json", owner),
    )
    monkeypatch.setattr(cli, "PARENT_REFS", copy.deepcopy(previous.parent_refs))
    monkeypatch.setattr(cli, "PREDECESSOR_REFS", predecessor_refs)
    config = copy.deepcopy(previous.config)
    config.update(output_dir=str(cli.RESULT.parent), repair_predecessor=copy.deepcopy(predecessor_refs))
    previous.write(cli.CONFIG, config)
    qa = copy.deepcopy(previous.qa)
    qa.update(
        source_count=29,
        test_module_count=9,
        source_sha256={path: cli.reference(previous.repo / path)["sha256"] for path in cli.SOURCE_PATHS},
        command=[
            str(cli.NATIVE_PYTHON),
            "-m",
            "pytest",
            "-q",
            *cli.QA_TESTS,
            f"--basetemp={previous.repo}/outputs/qa_v3_tmp",
        ],
    )

    def bootstrap(runtime_config, record, output, **kwargs):
        previous.calls.append("v3_bootstrap")
        assert runtime_config == previous.manifest["runtime_config"]
        assert record == previous.manifest["record"]
        assert output == cli.RESULT.parent / "work"
        assert kwargs == dict(allowed_asset_roots=config["allowed_asset_roots"])
        return copy.deepcopy(previous.receipt)

    monkeypatch.setattr(core, "run_native_bootstrap", bootstrap)
    return SimpleNamespace(
        cli=cli,
        core=core,
        previous=previous,
        repo=previous.repo,
        write=previous.write,
        calls=previous.calls,
        config=config,
        qa=qa,
        qa_ref=previous.write(previous.repo / "outputs/qa_v3/result.json", qa),
        predecessor_refs=predecessor_refs,
        failed=failed,
        owner=owner,
        old_v2_bytes=Path(failed_ref["path"]).read_bytes(),
    )


def test_inverse_proof_and_source_rosters_preserve_both_predecessors():
    cli = load_cli()
    proof = cli.compile_cli_repair_proof()
    assert proof["passed"] and proof["inherited_cli_inverse_ast_equal"]
    assert proof["predecessor_cli_proof"]["passed"]
    assert proof["new_admission_wrappers_claimed_identical"] is False
    assert len(cli.SOURCE_PATHS) == len(set(cli.SOURCE_PATHS)) == 29
    assert cli.SOURCE_PATHS[:23] == cli.frozen_cli.SOURCE_PATHS
    assert len(cli.QA_TESTS) == len(set(cli.QA_TESTS)) == 9
    assert cli.QA_TESTS[:7] == cli.frozen_cli.QA_TESTS
    assert cli._inherited_main.__globals__ is vars(cli)
    assert cli.frozen_cli._inherited_main.__globals__ is not vars(cli)


@pytest.mark.parametrize("mutation", ["budget", "old_source_path", "metadata"])
def test_inverse_proof_rejects_unregistered_ast_changes(mutation):
    cli = load_cli()
    tree = cli.repaired_cli_ast()
    if mutation == "budget":
        next(node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == 600).value = 601
    elif mutation == "old_source_path":
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "configs/native_runtime_bootstrap_20261029.json"
        ).value = "configs/native_runtime_bootstrap_v3_20261029.json"
    else:
        call = next(iter(cli.frozen_cli._receipt_calls(tree)))
        next(keyword for keyword in call.keywords if keyword.arg == "repair_predecessor").value = ast.Constant(True)
    with pytest.raises(ValueError, match="AST|metadata"):
        cli.compile_cli_repair_proof(tree)


def test_dry_checks_both_closed_failures_without_runtime_or_old_global_mutation(admitted_v3):
    a = admitted_v3
    prior_cli_globals = (a.cli.frozen_cli.REPO, a.cli.frozen_cli.PARENT_REFS, a.cli.frozen_cli.SOURCE_PATHS)
    result = a.cli.main(a.qa_ref, dry=True)
    assert result["environment_initializations"] == 0 and result["passed"]
    assert result["repair_parent"]["original_source_count"] == 17
    assert result["repair_predecessor"]["original_source_count"] == 23
    assert result["repair_predecessor"]["cleanup_complete"]
    assert result["cli_repair_proof"]["passed"] and result["core_repair_proof"]["passed"]
    assert a.calls == [] and not a.cli.RESULT.parent.exists()
    assert (a.cli.frozen_cli.REPO, a.cli.frozen_cli.PARENT_REFS, a.cli.frozen_cli.SOURCE_PATHS) == prior_cli_globals


def test_new_qa_cannot_rewrite_predecessor_source_even_with_updated_new_digest(admitted_v3):
    a = admitted_v3
    assert a.cli.verify_native_qa(a.qa_ref) == a.qa
    path = "src/s2a/native_runtime_bootstrap_v2.py"
    new_ref = a.write(a.repo / path, b"changed old v2 source")
    a.qa["source_sha256"][path] = new_ref["sha256"]
    with pytest.raises(ValueError, match="SHA mismatch"):
        a.cli.verify_native_qa(a.write(a.qa_ref["path"], a.qa))
    assert a.calls == []


@pytest.mark.parametrize("mutation", ["unreaped", "missing_cleanup", "different_stage", "extra_reset", "time_advanced"])
def test_predecessor_lifecycle_stage_and_counts_are_required_before_runtime(admitted_v3, mutation):
    a = admitted_v3
    if mutation == "unreaped":
        a.owner["commands"][0]["reaped"] = False
        key, value = "owner_ref", a.owner
    else:
        key, value = "result_ref", a.failed
        if mutation == "missing_cleanup":
            value["bootstrap"]["cleanup_complete"] = False
        elif mutation == "different_stage":
            value["failure_stage"] = "constructor"
        elif mutation == "extra_reset":
            value["bootstrap"]["counters"]["reset"] = dict(attempted=1, completed=1)
        else:
            value["bootstrap"]["time_checks"][0]["after"] = 0.002
    a.predecessor_refs[key] = a.write(a.predecessor_refs[key]["path"], value)
    a.config["repair_predecessor"] = copy.deepcopy(a.predecessor_refs)
    with pytest.raises(ValueError, match="record|reaped|time unchanged"):
        a.cli.inspect_inputs(a.config)
    assert a.calls == []


def test_scope_changes_are_rejected_before_any_runtime(admitted_v3):
    a = admitted_v3
    changed = copy.deepcopy(a.config)
    changed["limits"]["model_build"] += 1
    with pytest.raises(ValueError, match="Only output directory"):
        a.cli.inspect_inputs(changed)
    assert a.calls == []


def test_success_dispatches_only_v3_and_keeps_both_original_failures(admitted_v3, monkeypatch):
    a = admitted_v3

    def forbidden(*args, **kwargs):
        raise AssertionError("predecessor bootstrap dispatched")

    monkeypatch.setattr(a.previous.core, "run_native_bootstrap", forbidden)
    monkeypatch.setattr(a.previous.old_core, "run_native_bootstrap", forbidden)
    result = a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "v3_bootstrap"]
    assert result["complete"] is result["passed"] is result["source_unchanged"] is True
    assert result["control_qualified"] is result["historical_runtime_equivalence_claimed"] is False
    assert result["bootstrap"] == a.previous.receipt
    assert result["repair_parent"]["result_ref"] == a.previous.parent_refs["result_ref"]
    assert result["repair_predecessor"]["result_ref"] == a.predecessor_refs["result_ref"]
    assert Path(a.predecessor_refs["result_ref"]["path"]).read_bytes() == a.old_v2_bytes
    assert Path(a.previous.parent_refs["result_ref"]["path"]).read_bytes() == a.previous.old_result_bytes
    assert json.loads(a.cli.RESULT.read_text()) == result
    with pytest.raises(ValueError, match="exclusive"):
        a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "v3_bootstrap"]


def test_new_failure_is_published_once_with_actual_partial_counts(admitted_v3, monkeypatch):
    a = admitted_v3
    receipt = copy.deepcopy(a.previous.receipt)
    receipt.update(complete=False, passed=False, bootstrap_verified=False, failure_stage="initial_state")

    def fail(*args, **kwargs):
        a.calls.append("v3_bootstrap")
        raise a.core.BootstrapFailure("synthetic v3 state failure", receipt)

    monkeypatch.setattr(a.core, "run_native_bootstrap", fail)
    with pytest.raises(a.core.BootstrapFailure, match="synthetic v3 state failure"):
        a.cli.main(a.qa_ref)
    result = json.loads(a.cli.RESULT.read_text())
    assert result["complete"] is result["passed"] is False
    assert result["source_unchanged"] is True and result["bootstrap"] == receipt
    assert result["failure_stage"] == "initial_state"
    assert result["repair_parent"]["passed"] and result["repair_predecessor"]["passed"]
    assert a.calls == ["cv2", "v3_bootstrap"]

"""Synthetic asset-repair admission and ownership; no dynamics or real child."""

import importlib
import json
from pathlib import Path

import pytest

from s2a.prefix_trust_control_evidence import reference


@pytest.fixture
def repair():
    return importlib.import_module("s2a.prefix_trust_asset_repair")


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("execute_tracking_v5_prefix_trust_check_v2")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return reference(path)


def qa_receipt(execution, repair, monkeypatch, tmp_path, defect=None):
    monkeypatch.setattr(execution, "REPO", tmp_path)
    monkeypatch.setattr(execution, "NATIVE_PYTHON", tmp_path / "native-python")
    monkeypatch.setitem(execution.original_verify_native_qa.__globals__, "REPO", tmp_path)
    monkeypatch.setitem(execution.original_verify_native_qa.__globals__, "NATIVE_PYTHON", execution.NATIVE_PYTHON)
    sources = {}
    for relative in repair.QA_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen source\n")
        sources[relative] = reference(path)["sha256"]
    log = tmp_path / "pytest.log"
    log.write_text("synthetic six-file QA passed\n")
    qa = dict(
        complete=True,
        passed=True,
        source_unchanged=True,
        actual_wait=True,
        returncode=0,
        command=[str(execution.NATIVE_PYTHON), "-m", "pytest", *execution.QA_TESTS],
        runtime=dict(python="3.11.15", numpy="1.26.4", scipy="1.15.3"),
        environment=dict(execution.CPU_ENVIRONMENT),
        source_sha256=sources,
        log_sha256=reference(log)["sha256"],
    )
    old_source = next(path for path in repair.QA_SOURCES if path not in repair.REPAIR_SOURCES)
    old_test = next(path for path in execution.QA_TESTS if path not in repair.NEW_TESTS)
    if defect == "not_waited":
        qa["actual_wait"] = False
    elif defect == "failed_exit":
        qa["returncode"] = 1
    elif defect == "missing_old_test":
        qa["command"].remove(old_test)
    elif defect == "missing_new_test":
        qa["command"].remove(repair.NEW_TESTS[-1])
    elif defect == "missing_old_source":
        del sources[old_source]
    elif defect == "missing_new_source":
        del sources[repair.REPAIR_SOURCES[0]]
    elif defect == "changed_old_source":
        (tmp_path / old_source).write_text("changed after QA\n")
    elif defect == "changed_new_source":
        (tmp_path / repair.REPAIR_SOURCES[0]).write_text("changed after QA\n")
    elif defect == "gpu":
        qa["environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif defect == "thread_count":
        qa["environment"]["OMP_NUM_THREADS"] = "2"
    elif defect == "runtime":
        qa["runtime"]["numpy"] = "2.0.0"
    elif defect == "log_changed":
        log.write_text("changed after QA\n")
    return write_json(tmp_path / "qa.json", qa)


def test_owner_accepts_native_qa_covering_old_and_repair_sources(execution, repair, monkeypatch, tmp_path):
    assert len(execution.QA_TESTS) == 6 and len(repair.NEW_TESTS) == 3
    assert set(repair.NEW_TESTS) <= set(execution.QA_TESTS)
    assert execution.verify_native_qa(qa_receipt(execution, repair, monkeypatch, tmp_path))["passed"]


@pytest.mark.parametrize(
    "defect",
    [
        "not_waited",
        "failed_exit",
        "missing_old_test",
        "missing_new_test",
        "missing_old_source",
        "missing_new_source",
        "changed_old_source",
        "changed_new_source",
        "gpu",
        "thread_count",
        "runtime",
        "log_changed",
    ],
)
def test_owner_rejects_uncovered_changed_or_non_cpu_qa(execution, repair, monkeypatch, tmp_path, defect):
    with pytest.raises(ValueError):
        execution.verify_native_qa(qa_receipt(execution, repair, monkeypatch, tmp_path, defect))


def failed_execution_fixture(repair, monkeypatch, tmp_path, defect=None):
    """An actually closed, synthetic pre-query failure with real file hashes."""
    monkeypatch.setattr(repair, "REPO", tmp_path)
    monkeypatch.setattr(repair, "NATIVE_PYTHON", tmp_path / "native-python")
    capture = tmp_path / "failed/capture"
    monkeypatch.setattr(repair, "FAILED_CAPTURE", capture)
    source = tmp_path / "frozen.py"
    source.write_text("frozen source\n")
    reg = write_json(
        tmp_path / "failed/registration.json",
        dict(case_name="candidate", stage="check", source_sha256={str(source): reference(source)["sha256"]}),
    )
    producer_ref = dict(path=str(tmp_path / "candidate/verified.json"), sha256="synthetic-producer")
    monkeypatch.setattr(repair, "PRODUCER_REFS", {"candidate": producer_ref})
    check_reg = write_json(tmp_path / "failed/check-registration.json", dict(producer_ref=producer_ref))
    error = "ChildProcessError('Independent command failed with1; no automatic retry')"
    failed = dict(
        complete=False,
        passed=False,
        case_name="candidate",
        registration_ref=check_reg,
        error="ValueError('file not found wooden_cabinet_top_vis.msh')",
        boundary_classification_disagreements=[],
        **dict.fromkeys(
            [
                "response_evaluations",
                "verified_responses",
                "numerical_inner_steps",
                "mj_forward_queries",
                "query_count",
                "physics_steps",
                "world_model_calls",
                "fresh_cases",
            ],
            0,
        ),
    )
    owner = dict(complete=False, passed=False, case_name="candidate", stage="check", registration_ref=reg, error=error)
    launch = dict(
        command=[
            str(repair.NATIVE_PYTHON),
            str(tmp_path / "scripts/check_tracking_v5_prefix_trust_control.py"),
            "--case",
            "candidate",
        ],
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        automatic_retry=False,
        child=dict(pid=1234, start_ticks=98765),
    )
    exited = dict(actual_wait=True, returncode=1, automatic_retry=False)
    release = dict(reaped=True, returncode=1, signal_sent=False, pid=1234, start_ticks=98765, original_error=error)
    if defect in ("response_evaluations", "query_count", "mj_forward_queries", "numerical_inner_steps"):
        failed[defect] = 1
    elif defect == "wrong_failure":
        failed["error"] = "Unrelated failure"
    elif defect == "not_waited":
        exited["actual_wait"] = False
    elif defect == "not_reaped":
        release["reaped"] = False
    elif defect == "wrong_exit":
        exited["returncode"] = 0
    elif defect == "signalled":
        release["signal_sent"] = True
    elif defect == "retry":
        launch["automatic_retry"] = True
    elif defect == "wrong_child":
        release["start_ticks"] += 1
    elif defect == "wrong_case":
        launch["command"][-1] = "baseline"
    elif defect == "changed_source":
        source.write_text("modified frozen source\n")
    monkeypatch.setattr(repair, "FAILED_RESULT_REF", write_json(tmp_path / "failed/verified.json", failed))
    monkeypatch.setattr(repair, "FAILED_OWNER_REF", write_json(tmp_path / "failed/result.json", owner))
    lifecycle = {
        name: write_json(capture / name, value)
        for name, value in (("launch.json", launch), ("exit.json", exited), ("ownership/release.json", release))
    }
    monkeypatch.setattr(repair, "FAILED_LIFECYCLE_REFS", lifecycle)
    (capture / "command.log").write_text("synthetic pre-query load failure\n")


def test_failed_original_child_is_bound_without_reusing_any_response(repair, monkeypatch, tmp_path):
    failed_execution_fixture(repair, monkeypatch, tmp_path)
    refs = {}
    repair.verify_failed_execution(refs)
    expected = [repair.FAILED_RESULT_REF, repair.FAILED_OWNER_REF, *repair.FAILED_LIFECYCLE_REFS.values()]
    assert all(refs[ref["path"]] == ref for ref in expected)
    assert str(repair.FAILED_CAPTURE / "command.log") in refs


@pytest.mark.parametrize(
    "defect",
    [
        "response_evaluations",
        "query_count",
        "mj_forward_queries",
        "numerical_inner_steps",
        "wrong_failure",
        "not_waited",
        "not_reaped",
        "wrong_exit",
        "signalled",
        "retry",
        "wrong_child",
        "wrong_case",
        "changed_source",
    ],
)
def test_failed_execution_cannot_hide_work_or_incomplete_ownership(repair, monkeypatch, tmp_path, defect):
    failed_execution_fixture(repair, monkeypatch, tmp_path, defect)
    with pytest.raises(ValueError):
        repair.verify_failed_execution({})


def snapshot_fixture(repair, monkeypatch, tmp_path, defect=None):
    from s2a import xml_asset_snapshot

    monkeypatch.setattr(repair, "REPO", tmp_path)
    monkeypatch.setattr(repair, "SNAPSHOT_ROOT", tmp_path / "snapshot")
    source_refs = []
    for relative in repair.REPAIR_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic current repair source\n")
        source_refs.append(reference(path))
    snapshot = dict(
        original_xml_ref=repair.ORIGINAL_XML_REF,
        historical_manifest_ref=repair.HISTORICAL_MANIFEST_REF,
        file_reference_count=99,
        unique_asset_count=96,
        assets=[{} for _ in range(96)],
        byte_exact_inverse=True,
    )
    snapshot_ref = write_json(repair.SNAPSHOT_ROOT / "manifest.json", snapshot)
    verified = []

    def verify(ref):
        verified.append(ref)
        assert ref == snapshot_ref
        return snapshot

    monkeypatch.setattr(xml_asset_snapshot, "verify_asset_snapshot", verify)
    compiled = dict(
        complete=True,
        passed=True,
        manifest_ref=snapshot_ref,
        runtime=dict(mujoco="2.3.7", numpy="1.26.4", python="3.11.15"),
        environment=dict(repair.CPU_ENVIRONMENT),
        model_dimensions=dict(nq=9, nv=9),
        timestep=0.002,
        source_refs=source_refs,
        **dict.fromkeys(
            [
                "response_evaluations",
                "query_count",
                "numerical_inner_steps",
                "physics_steps",
                "world_model_calls",
                "fresh_cases",
            ],
            0,
        ),
    )
    if defect in ("response_evaluations", "query_count", "numerical_inner_steps", "physics_steps"):
        compiled[defect] = 1
    elif defect == "incomplete_compile":
        compiled["passed"] = False
    elif defect == "missing_asset":
        snapshot["assets"].pop()
    elif defect == "not_byte_exact":
        snapshot["byte_exact_inverse"] = False
    elif defect == "different_manifest":
        compiled["manifest_ref"] = dict(snapshot_ref, sha256="different")
    elif defect == "gpu":
        compiled["environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif defect == "changed_repair_source":
        Path(source_refs[0]["path"]).write_text("changed after compilation\n")
    write_json(repair.SNAPSHOT_ROOT / "compile.json", compiled)
    return verified


def test_snapshot_admission_requires_verified_closure_and_zero_query_compilation(repair, monkeypatch, tmp_path):
    verified = snapshot_fixture(repair, monkeypatch, tmp_path)
    value = repair.load_snapshot()
    assert len(verified) == 1 and value["snapshot_ref"] == verified[0]
    assert value["unique_asset_count"] == len(value["assets"]) == 96


@pytest.mark.parametrize(
    "defect",
    [
        "response_evaluations",
        "query_count",
        "numerical_inner_steps",
        "physics_steps",
        "incomplete_compile",
        "missing_asset",
        "not_byte_exact",
        "different_manifest",
        "gpu",
        "changed_repair_source",
    ],
)
def test_asset_compile_receipt_cannot_hide_work_or_a_different_snapshot(repair, monkeypatch, tmp_path, defect):
    snapshot_fixture(repair, monkeypatch, tmp_path, defect)
    with pytest.raises(ValueError):
        repair.load_snapshot()


def producers_fixture(repair, monkeypatch, tmp_path, defect=None):
    monkeypatch.setattr(repair, "REPO", tmp_path)
    monkeypatch.setattr(repair, "verify_failed_execution", lambda refs: None)
    source_sha256 = {}
    for relative in repair.QA_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n")
        source_sha256[relative] = reference(path)["sha256"]
    for name in ("HISTORICAL_MANIFEST_REF", "ORIGINAL_XML_REF"):
        monkeypatch.setattr(repair, name, write_json(tmp_path / f"{name}.json", {"synthetic": True}))
    qa_ref = write_json(tmp_path / "producer-qa.json", dict(source_sha256=source_sha256))
    owners, producers, owner_refs, producer_refs = {}, {}, {}, {}
    for name, count in (("baseline", 4000), ("candidate", 3148)):
        selected_qa = qa_ref
        if defect == "different_qa" and name == "candidate":
            selected_qa = write_json(tmp_path / "other-qa.json", dict(source_sha256=source_sha256))
        reg_ref = write_json(tmp_path / name / "registration.json", dict(native_component_qa_ref=selected_qa))
        producers[name] = dict(
            case_name=name,
            response_evaluations=count,
            prediction_calls=count,
            numerical_inner_steps=300 * count,
            query_count=1 + 312 * count,
            answer=dict(evaluated_candidates=[{} for _ in range(count)], witness_found=False),
            source_refs_unchanged=[],
        )
        if name == "candidate" and defect in ("query_count", "prediction_calls"):
            producers[name][defect] -= 1
        producer_refs[name] = write_json(tmp_path / name / "verified.json", producers[name])
        owners[name] = dict(result_ref=producer_refs[name], registration_ref=reg_ref)
        owner_refs[name] = write_json(tmp_path / name / "owner.json", owners[name])
    monkeypatch.setattr(repair, "PRODUCER_REFS", producer_refs)
    monkeypatch.setattr(repair, "PRODUCER_OWNER_REFS", owner_refs)
    closed = []

    def closed_owner(ref, refs):
        name = next(name for name, value in owner_refs.items() if value == ref)
        closed.append(name)
        repair.remember(ref, refs)
        repair.remember(owners[name]["registration_ref"], refs)
        return owners[name], producers[name]

    monkeypatch.setattr(repair, "closed_owner", closed_owner)
    if defect == "old_source_changed":
        (tmp_path / repair.SOURCES[0]).write_text("changed since original producer QA\n")
    return closed


def test_repair_requires_both_closed_producers_and_their_unchanged_original_qa(repair, monkeypatch, tmp_path):
    closed = producers_fixture(repair, monkeypatch, tmp_path)
    refs = repair.repair_prerequisites(require_snapshot=False)
    assert closed == ["baseline", "candidate"]
    assert all(refs[ref["path"]] == ref for ref in repair.PRODUCER_REFS.values())


@pytest.mark.parametrize("defect", ["different_qa", "old_source_changed", "query_count", "prediction_calls"])
def test_repair_cannot_relabel_or_change_original_production(repair, monkeypatch, tmp_path, defect):
    producers_fixture(repair, monkeypatch, tmp_path, defect)
    with pytest.raises(ValueError):
        repair.repair_prerequisites(require_snapshot=False)


def owner_fixture(execution, monkeypatch, tmp_path, *, case_name="candidate", defect=None):
    monkeypatch.setattr(execution, "REPO", tmp_path)
    monkeypatch.setattr(execution, "NATIVE_PYTHON", tmp_path / "native-python")
    for key, value in execution.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.setattr(execution, "strict_native_environment", lambda: None)

    def roots(name, owner=False):
        return tmp_path / f"{name}_{'owner' if owner else 'check'}"

    monkeypatch.setattr(execution, "repair_root", roots)
    plan = tmp_path / "plan.md"
    plan.write_text("synthetic fixed v2 checker plan\n")
    monkeypatch.setattr(execution, "PLAN", plan)
    log = tmp_path / "pytest.log"
    log.write_text("synthetic QA\n")
    qa = dict(log_sha256=reference(log)["sha256"])
    qa_ref = write_json(tmp_path / "qa.json", qa)
    calls = []
    monkeypatch.setattr(execution, "verify_native_qa", lambda ref: qa)
    monkeypatch.setattr(execution, "load_asset_repair_prerequisites", lambda: ({}, {}, {}, {}))
    snapshot = dict(
        snapshot_ref=write_json(tmp_path / "manifest.json", {"synthetic": True}),
        compile_ref=write_json(tmp_path / "compile.json", {"synthetic": True}),
    )
    monkeypatch.setattr(execution, "load_snapshot", lambda: snapshot)

    def owned(command, environment, capture, *, timeout):
        calls.append(dict(command=command, environment=environment, capture=capture, timeout=timeout))
        if defect == "child_failed":
            raise ChildProcessError("synthetic child failed; no automatic retry")
        count = 4000 if case_name == "baseline" else 3148
        report = dict(
            complete=True,
            passed=True,
            case_name=case_name,
            response_evaluations=count,
            verified_responses=count,
            numerical_inner_steps=count * 300,
            query_count=count * 312 + 1,
            mj_forward_queries=count * 312 + 1,
            boundary_classification_disagreements=[],
            control_qualified=False,
        )
        if defect == "partial_trace":
            report["verified_responses"] -= 1
        elif defect == "query_count":
            report["query_count"] -= 1
        elif defect == "classification":
            report["boundary_classification_disagreements"] = ["synthetic disagreement"]
        elif defect == "control_claim":
            report["control_qualified"] = True
        write_json(roots(case_name) / "verified.json", report)
        return dict(synthetic_owned_receipt=True)

    monkeypatch.setattr(execution, "owned_command", owned)
    return qa_ref, roots, calls


def test_owner_dry_checks_admission_without_launch_or_response(execution, monkeypatch, tmp_path, capsys):
    qa_ref, roots, calls = owner_fixture(execution, monkeypatch, tmp_path)
    execution.main("candidate", qa_ref, dry=True)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["new_responses"] == 0 and receipt["expected_responses"] == 3148
    assert calls == [] and not roots("candidate", owner=True).exists()


@pytest.mark.parametrize("case_name,count", [("baseline", 4000), ("candidate", 3148)])
def test_v2_owner_delegates_one_complete_replay_with_fixed_timeout(execution, monkeypatch, tmp_path, case_name, count):
    qa_ref, roots, calls = owner_fixture(execution, monkeypatch, tmp_path, case_name=case_name)
    execution.main(case_name, qa_ref)
    assert len(calls) == 1
    assert calls[0]["command"] == [
        str(execution.NATIVE_PYTHON),
        str(tmp_path / "scripts/check_tracking_v5_prefix_trust_control_v2.py"),
        "--case",
        case_name,
    ]
    assert calls[0]["timeout"] == 5400
    assert all(calls[0]["environment"][key] == value for key, value in execution.CPU_ENVIRONMENT.items())
    assert not {"MUJOCO_GL", "PYOPENGL_PLATFORM"} & set(calls[0]["environment"])
    registration = json.loads((roots(case_name, owner=True) / "registration.json").read_text())
    assert registration["expected_responses"] == count
    assert registration["expected_inner_steps"] == count * 300
    assert registration["expected_queries"] == count * 312 + 1
    assert registration["preserved_failed_work"] == dict(response_evaluations=0, query_count=0, numerical_inner_steps=0)
    assert registration["automatic_retry"] is False
    result = json.loads((roots(case_name, owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is True


@pytest.mark.parametrize("defect", ["child_failed", "partial_trace", "query_count", "classification", "control_claim"])
def test_owner_preserves_failed_or_incomplete_checks_without_retry(execution, monkeypatch, tmp_path, defect):
    qa_ref, roots, calls = owner_fixture(execution, monkeypatch, tmp_path, defect=defect)
    with pytest.raises((ValueError, ChildProcessError)):
        execution.main("candidate", qa_ref)
    assert len(calls) == 1
    result = json.loads((roots("candidate", owner=True) / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and result["error"]


@pytest.mark.parametrize("owner", [True, False])
def test_owner_rejects_existing_output_without_launch(execution, monkeypatch, tmp_path, owner):
    qa_ref, roots, calls = owner_fixture(execution, monkeypatch, tmp_path)
    roots("candidate", owner=owner).mkdir()
    with pytest.raises(ValueError):
        execution.main("candidate", qa_ref)
    assert calls == []

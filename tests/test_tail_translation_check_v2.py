"""Typed metadata repair tests; saved archives are never sent to a response model."""

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_prefix_callback_backtracking_execution import write_json

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return SimpleNamespace(
        checker=importlib.import_module("check_tracking_v5_tail_translation_reserve_v2"),
        owner=importlib.import_module("execute_tracking_v5_tail_translation_check_v2"),
        evidence=importlib.import_module("s2a.tail_translation_check_v2_evidence"),
        audit=importlib.import_module("tail_translation_reserve_audit"),
    )


@pytest.mark.parametrize(
    "actual,saved",
    [
        (np.zeros((25, 7), dtype=bool), [[False] * 7 for _ in range(25)]),
        (np.array([True, False]), [True, False]),
        (np.array([1, 26], dtype=np.int32), np.array([1, 26], dtype=np.uint64)),
        (np.array([2**63 + 1], dtype=np.uint64), [2**63 + 1]),
        (np.array([], dtype=bool), np.array([], dtype=bool)),
    ],
)
def test_label_arrays_are_exact_without_boolean_or_integer_subtraction(modules, monkeypatch, actual, saved):
    def forbidden(*args, **kwargs):
        pytest.fail("Exact labels must not reach floating-point subtraction")

    monkeypatch.setattr(modules.checker, "numerical_compare", forbidden)
    assert modules.checker.semantic_compare("labels", actual, saved, 0.0) == 0.0


@pytest.mark.parametrize(
    "actual,saved",
    [
        ([[True, False]], [[True, True]]),
        ([[True, False]], [True, False]),
        ([True, False], [1, 0]),
        ([True, False], [1.0, 0.0]),
        ([1, 0], [True, False]),
        ([1, 0], [1.0, 0.0]),
        ([1.0, 0.0], [1, 0]),
        ([1, 2], [1, 3]),
        ([1, 0], [True, 0]),
        (np.array([2**63 - 1], dtype=np.int64), np.array([2**63], dtype=np.uint64)),
    ],
)
def test_label_mismatch_never_receives_a_tolerance(modules, actual, saved):
    with pytest.raises(ValueError, match="Exact labels labels"):
        modules.checker.semantic_compare("labels", actual, saved, 1e20)


def test_floating_comparison_is_the_original_function_and_boundary(modules, monkeypatch):
    old = modules.checker.numerical_compare
    calls = []

    def observed(*args):
        calls.append(args)
        return old(*args)

    monkeypatch.setattr(modules.checker, "numerical_compare", observed)
    tolerance = modules.checker.NUMERIC_TOLERANCE
    assert modules.checker.semantic_compare("float", [0.0], [tolerance]) == tolerance
    assert len(calls) == 1 and calls[0] == ("float", [0.0], [tolerance], tolerance)
    for expected in (np.nextafter(tolerance, np.inf), np.nan, np.inf):
        with pytest.raises(ValueError):
            modules.checker.semantic_compare("float", [0.0], [expected])
    with pytest.raises(ValueError):
        modules.checker.semantic_compare("float", [[0.0]], [0.0])
    assert modules.checker.semantic_compare("torque", [0.0], [1e-8], 1e-8) == 1e-8
    with pytest.raises(ValueError):
        modules.checker.semantic_compare("torque", [0.0], [np.nextafter(1e-8, np.inf)], 1e-8)


@pytest.fixture(scope="module")
def saved_archives():
    """The two preregistered saved JSON archives, read once with exact SHA binding."""
    from s2a.tail_translation_check_v2_evidence import PRIOR_PRODUCER_REF, PRODUCER_REF

    def load(ref):
        raw = Path(ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        return json.loads(raw)

    return load(PRODUCER_REF), load(PRIOR_PRODUCER_REF)


def test_actual_saved_bool_failure_and_complete_ledger_repair_without_responses(modules, saved_archives, monkeypatch):
    producer, prior = saved_archives
    answer, prior_answer = producer["answer"], prior["answer"]
    historical = prior_answer["evaluated_candidates"][0]
    flags = historical["rollout"]["periods"][0]["clipped"]
    assert np.asarray(flags).dtype.kind == "b"
    with pytest.raises(TypeError, match="boolean subtract"):
        modules.checker.numerical_compare("prior historical anchor.rollout.periods[0].clipped", flags, flags)

    def no_response(*args, **kwargs):
        pytest.fail("Saved metadata regression must not create an engine or replay a response")

    monkeypatch.setattr(modules.checker, "case_engine", no_response)
    monkeypatch.setattr(modules.checker, "verify_response_trace", no_response)
    seen = []

    def compare(name, actual, saved, tolerance=modules.checker.NUMERIC_TOLERANCE):
        seen.append((name, np.asarray(actual).dtype.kind))
        return modules.checker.semantic_compare(name, actual, saved, tolerance)

    result = modules.audit.audit_tail_translation_ledger(
        answer, historical_anchor=historical, prior_answer=prior_answer, compare=compare
    )
    assert result["passed"] is True and result["response_evaluations"] == 14
    assert result["unique_command_count"] == 13
    assert result["full_responses_independently_replayed_here"] is False
    assert result["control_qualified"] is False
    for field in ("selected_response_index", "improvement_found", "sufficient_reserve_restored"):
        assert result[field] == answer[field]
    assert result["witness_indices"] == producer["witness_indices"]
    assert any(name.endswith("periods[0].clipped") and kind == "b" for name, kind in seen)


def test_actual_historical_boolean_bit_change_is_rejected_before_ledger_acceptance(modules, saved_archives):
    producer, prior = saved_archives
    historical = copy.deepcopy(prior["answer"]["evaluated_candidates"][0])
    flags = historical["rollout"]["periods"][0]["clipped"]
    flags[0][0] = not flags[0][0]
    with pytest.raises(ValueError, match=r"periods\[0\].clipped"):
        modules.audit.audit_tail_translation_ledger(
            producer["answer"],
            historical_anchor=historical,
            prior_answer=prior["answer"],
            compare=modules.checker.semantic_compare,
        )


def test_checker_inverse_ast_has_only_registered_metadata_changes():
    old = ast.parse((REPO / "scripts/check_tracking_v5_tail_translation_reserve.py").read_text())
    new = ast.parse((REPO / "scripts/check_tracking_v5_tail_translation_reserve_v2.py").read_text())
    new.body = [
        node for node in new.body if not (isinstance(node, ast.FunctionDef) and node.name == "semantic_compare")
    ]
    additions = {"FAILED_CHECK_REF", "FAILED_OWNER_REF", "REPAIR_PLAN"}
    removed_keywords = []
    for node in ast.walk(new):
        if isinstance(node, ast.ImportFrom):
            if node.module == "s2a.tail_translation_check_v2_evidence":
                node.module = "s2a.tail_translation_reserve_evidence"
                node.names = [name for name in node.names if name.name not in additions]
            elif node.module == "execute_tracking_v5_tail_translation_check_v2":
                node.module = "execute_tracking_v5_tail_translation_reserve"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "semantic_compare":
                node.func.id = "numerical_compare"
            for keyword in node.keywords.copy():
                if keyword.arg in {"repair_plan_ref", "failed_checker_ref", "failed_owner_ref"}:
                    removed_keywords.append(keyword.arg)
                    node.keywords.remove(keyword)
    assert sorted(removed_keywords) == ["failed_checker_ref", "failed_owner_ref", "repair_plan_ref"]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def qa_fixture(modules, monkeypatch, tmp_path):
    """Exercise only the new layer over a previously validated old QA layer."""
    owner = modules.owner
    monkeypatch.setattr(owner, "REPO", tmp_path)
    sources = {path: write_json(tmp_path / path, {"source": path})["sha256"] for path in owner.QA_SOURCES}
    old = dict(source_sha256={path: sources[path] for path in owner.PRIOR_QA_SOURCES})
    old_ref = write_json(tmp_path / "old_qa.json", old)
    monkeypatch.setattr(owner, "PRIOR_QA_REF", old_ref)
    monkeypatch.setattr(owner, "verify_prior_qa", owner.read)
    qa = dict(
        source_sha256=sources,
        command=["native", "-m", "pytest", "-q", *owner.QA_TESTS, "--basetemp=synthetic"],
    )
    return qa, old


def test_repair_qa_retains_all95_old_digests_and_requires100_sources30_modules(modules, monkeypatch, tmp_path):
    qa, _ = qa_fixture(modules, monkeypatch, tmp_path)
    owner = modules.owner
    assert len(owner.PRIOR_QA_SOURCES) == 95 and len(owner.QA_SOURCES) == 100 and len(owner.QA_TESTS) == 30
    assert owner.verify_native_qa(write_json(tmp_path / "qa.json", qa)) == qa
    for path in modules.evidence.SOURCES:
        changed = copy.deepcopy(qa)
        del changed["source_sha256"][path]
        with pytest.raises(ValueError, match="Complete100-source"):
            owner.verify_native_qa(write_json(tmp_path / "missing.json", changed))
    changed = copy.deepcopy(qa)
    changed["command"].remove(owner.QA_TESTS[-1])
    with pytest.raises(ValueError, match="Complete100-source"):
        owner.verify_native_qa(write_json(tmp_path / "missing_test.json", changed))


def test_changed_old_digest_cannot_be_laundered_by_new_qa(modules, monkeypatch, tmp_path):
    qa, _ = qa_fixture(modules, monkeypatch, tmp_path)
    path = modules.owner.PRIOR_QA_SOURCES[0]
    qa["source_sha256"][path] = write_json(tmp_path / path, {"changed": True})["sha256"]
    with pytest.raises(ValueError, match="unchanged QA subset"):
        modules.owner.verify_native_qa(write_json(tmp_path / "changed.json", qa))


def test_new_source_change_after_qa_is_rejected(modules, monkeypatch, tmp_path):
    qa, _ = qa_fixture(modules, monkeypatch, tmp_path)
    write_json(tmp_path / modules.evidence.SOURCES[0], {"changed": True})
    with pytest.raises(ValueError, match="current repair sources"):
        modules.owner.verify_native_qa(write_json(tmp_path / "changed.json", qa))


def test_check_only_paths_preserve_original_producer_and_new_checker(modules):
    owner, evidence = modules.owner, modules.evidence
    old = importlib.import_module("execute_tracking_v5_tail_translation_reserve")
    assert owner.entry_command("diagnose") == old.entry_command("diagnose")
    assert owner.entry_command("check") == [
        str(owner.NATIVE_PYTHON),
        str(REPO / "scripts/check_tracking_v5_tail_translation_reserve_v2.py"),
    ]
    assert evidence.case_root() == evidence.prior.case_root()
    assert evidence.case_root(owner=True) == evidence.prior.case_root(owner=True)
    assert evidence.case_root(check=True) != evidence.prior.case_root(check=True)
    with pytest.raises(ValueError):
        owner.entry_command("replay_partial")


def checker_registration(owner):
    return dict(
        producer_ref=owner.PRODUCER_REF,
        producer_owner_ref=owner.PRODUCER_OWNER_REF,
        plan_ref=owner.reference(owner.PLAN),
        repair_plan_ref=owner.reference(owner.REPAIR_PLAN),
        failed_checker_ref=owner.FAILED_CHECK_REF,
        failed_owner_ref=owner.FAILED_OWNER_REF,
    )


def test_checker_registration_keeps_both_distinct_plans_and_original_failed_run(modules):
    owner = modules.owner
    registration = checker_registration(owner)
    assert registration["plan_ref"] != registration["repair_plan_ref"]
    owner.validate_checker_registration(registration)


@pytest.mark.parametrize(
    "field",
    ["producer_ref", "producer_owner_ref", "plan_ref", "repair_plan_ref", "failed_checker_ref", "failed_owner_ref"],
)
def test_checker_registration_rejects_each_wrong_binding(modules, field):
    registration = copy.deepcopy(checker_registration(modules.owner))
    registration[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="both plans and preserved failure"):
        modules.owner.validate_checker_registration(registration)

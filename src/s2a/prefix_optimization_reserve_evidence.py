"""Admit a fixed optimization reserve only after complete callback-ray replay."""

from s2a.prefix_callback_backtracking_evidence import (
    BACKTRACKING_REF as BACKTRACKING_REF,
)
from s2a.prefix_callback_backtracking_evidence import (
    CPU_ENVIRONMENT as CPU_ENVIRONMENT,
)
from s2a.prefix_callback_backtracking_evidence import (
    NATIVE_PYTHON as NATIVE_PYTHON,
)
from s2a.prefix_callback_backtracking_evidence import (
    PREFIX_CHECK_REF as PREFIX_CHECK_REF,
)
from s2a.prefix_callback_backtracking_evidence import (
    QA_SOURCES as CALLBACK_QA_SOURCES,
)
from s2a.prefix_callback_backtracking_evidence import (
    QA_TESTS as CALLBACK_QA_TESTS,
)
from s2a.prefix_callback_backtracking_evidence import (
    REPO,
    closed_owner,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.prefix_callback_backtracking_evidence import (
    TRUST_REGISTRATION_REFS as TRUST_REGISTRATION_REFS,
)
from s2a.prefix_callback_backtracking_evidence import (
    case_root as callback_root,
)
from s2a.prefix_callback_backtracking_evidence import (
    checked as checked,
)
from s2a.prefix_callback_backtracking_evidence import (
    load_prerequisites as load_callback_prerequisites,
)
from s2a.prefix_callback_backtracking_evidence import (
    model_registration as model_registration,
)

PLAN = REPO / "plans/v0.0.1/prefix_optimization_reserve_20261029.md"
SOURCES = (
    "src/s2a/prefix_optimization_reserve_plan.py",
    "src/s2a/prefix_optimization_reserve_evidence.py",
    "scripts/prefix_optimization_reserve_audit.py",
    "scripts/probe_tracking_v5_prefix_optimization_reserve.py",
    "scripts/check_tracking_v5_prefix_optimization_reserve.py",
    "scripts/execute_tracking_v5_prefix_optimization_reserve.py",
    "plans/v0.0.1/prefix_optimization_reserve_20261029.md",
    "tests/test_prefix_optimization_reserve_plan.py",
    "tests/test_prefix_optimization_reserve_audit.py",
    "tests/test_prefix_optimization_reserve_execution.py",
)
QA_SOURCES = (*CALLBACK_QA_SOURCES, *SOURCES)
QA_TESTS = (*CALLBACK_QA_TESTS, *SOURCES[-3:])
PREFIX_OPTIMIZATION_MARGIN = 1e-4
CALLBACK_PRODUCER_REF = dict(
    path=str(callback_root() / "verified.json"),
    sha256="81051b71f0822b56e52492f87ce5032623212453d228f4e5952514bef036ebbe",
)
CALLBACK_OWNER_REF = dict(
    path=str(callback_root(owner=True) / "result.json"),
    sha256="a233f5c141bf52f1bc0cf26f8c6e0426cf6ff2d4f1e3bf5689c0c3ac9defa427",
)
CALLBACK_CHECK_REG_REF = dict(
    path=str(callback_root(check=True, owner=True) / "registration.json"),
    sha256="3bc501a4855135fdca55a7d4106489692dadf0c06be7eb9d8a2cbaa4826db5b4",
)
CALLBACK_QA_REF = dict(
    path=str(REPO / "outputs/prefix_callback_backtracking_native_qa_20261029_vv9yu8uw/result.json"),
    sha256="05e3ec58822c9a3a612ab30a3212451c4cceffc0e6edf3cdecf317d1dd6a3add",
)


def case_root(case_name="candidate", *, check=False, owner=False):
    require(case_name == "candidate", "One registered fixed-reserve candidate")
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_prefix_optimization_reserve{suffix}_20261029"


def closed_callback_replay(refs):
    """Bind actual terminal replay and all full ray/anchor coverage, never a partial log."""
    producer_owner, producer = closed_owner(remember(CALLBACK_OWNER_REF, refs), refs)
    require(producer_owner["result_ref"] == CALLBACK_PRODUCER_REF, "Exact prior callback producer")
    remember(CALLBACK_PRODUCER_REF, refs)
    owner_ref = remember(reference(callback_root(check=True, owner=True) / "result.json"), refs)
    owner, report = closed_owner(owner_ref, refs)
    result_ref = remember(reference(callback_root(check=True) / "verified.json"), refs)
    registration = read(remember(CALLBACK_CHECK_REG_REF, refs))
    report_reg = read(remember(report["registration_ref"], refs))
    producer_reg = read(producer_owner["registration_ref"])
    require(
        owner["registration_ref"] == CALLBACK_CHECK_REG_REF
        and owner["result_ref"] == result_ref
        and owner["case_name"] == report["case_name"] == registration["case_name"] == "candidate"
        and owner["stage"] == registration["stage"] == "check"
        and registration["producer_owner_ref"] == report_reg["producer_owner_ref"] == CALLBACK_OWNER_REF
        and registration["producer_ref"] == report_reg["producer_ref"] == CALLBACK_PRODUCER_REF
        and registration["native_component_qa_ref"] == producer_reg["native_component_qa_ref"] == CALLBACK_QA_REF
        and read(owner["owned_command"]["launch"])["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_prefix_callback_backtracking.py")]
        and report["response_evaluations"] == report["verified_responses"] == 1012
        and report["numerical_inner_steps"] == 303600
        and report["query_count"] == report["mj_forward_queries"] == 315745
        and report["boundary_classification_disagreements"] == []
        and report["witness_found"] is report["control_qualified"] is False
        and report["complete_ray_rollouts_checked"] == 75
        and report["accepted_anchor_audits_checked"] == 13
        and all(report[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "Complete original callback replay, all full rays/anchors and successful real wait/reap required",
    )
    ledger = report["ledger_check"]
    require(
        ledger["total_backtracking_attempts"] == 75
        and ledger["total_backtracking_new_responses"] == 60
        and len(ledger["accepted_anchor_indices"]) == 13
        and ledger["witness_indices"] == []
        and ledger["termination_reason"] == "trust_region_stalled",
        "Exact independently checked finite callback search",
    )
    initial = producer["answer"]["initial"]
    require(initial["prefix_length"] == 9, "Same incomplete prefix9 common warm")
    headroom = min(1.0 - float(value) for row in initial["normalized_squared_errors"][:9] for value in row)
    require(headroom >= PREFIX_OPTIMIZATION_MARGIN, "Common warm already has the fixed optimization reserve")
    for ref in producer["source_refs_unchanged"] + report["source_refs_unchanged"]:
        remember(ref, refs)
    finish(refs)
    return dict(owner_ref=owner_ref, result_ref=result_ref, minimum_initial_prefix_headroom=headroom)


def load_prerequisites():
    original, registration, backtracking, refs = load_callback_prerequisites()
    closed_callback_replay(refs)
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    finish(refs)
    return original, registration, backtracking, refs

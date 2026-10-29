"""Admit one full H12 restart from the completely verified tail restoration."""

import json

from s2a.tail_translation_check_v2_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.tail_translation_check_v2_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.tail_translation_check_v2_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.tail_translation_check_v2_evidence import QA_SOURCES as PRIOR_QA_SOURCES
from s2a.tail_translation_check_v2_evidence import QA_TESTS as PRIOR_QA_TESTS
from s2a.tail_translation_check_v2_evidence import REPO, closed_owner, finish, read, reference, remember, require
from s2a.tail_translation_check_v2_evidence import load_prerequisites as load_tail_prerequisites
from s2a.tail_translation_check_v2_evidence import model_registration as model_registration
from s2a.tail_translation_reserve_evidence import checked as checked

PREFIX_OPTIMIZATION_MARGIN = 1e-4
PLAN = REPO / "plans/v0.0.1/restored_anchor_sqp_20261029.md"
SOURCES = (
    "src/s2a/restored_anchor_sqp_plan.py",
    "src/s2a/restored_anchor_sqp_evidence.py",
    "scripts/restored_anchor_sqp_ledger_audit.py",
    "scripts/probe_tracking_v5_restored_anchor_sqp.py",
    "scripts/check_tracking_v5_restored_anchor_sqp.py",
    "scripts/execute_tracking_v5_restored_anchor_sqp.py",
    "plans/v0.0.1/restored_anchor_sqp_20261029.md",
    "tests/test_restored_anchor_sqp_plan.py",
    "tests/test_restored_anchor_sqp_ledger_audit.py",
    "tests/test_restored_anchor_sqp_execution.py",
)
QA_SOURCES = (*PRIOR_QA_SOURCES, *SOURCES)
QA_TESTS = (*PRIOR_QA_TESTS, *SOURCES[-3:])
PRIOR_QA_REF = dict(
    path=str(REPO / "outputs/tail_translation_check_v2_native_qa_20261029_y5rxr7gl/result.json"),
    sha256="fe36dfd12c29f6a55fd73d789cc91132e3fb726069100cf3a729743e421590d8",
)
TAIL_PRODUCER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_tail_translation_reserve_20261029/verified.json"),
    sha256="0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70",
)
TAIL_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_tail_translation_reserve_execution_20261029/result.json"),
    sha256="f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb",
)
TAIL_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_tail_translation_reserve_check_v2_20261029/verified.json"),
    sha256="142b0d9ad478b30c1e707c054cec48bd29326b0b24b4e07e64b6ea8cb37e6bde",
)
TAIL_CHECK_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_tail_translation_reserve_check_v2_execution_20261029/result.json"),
    sha256="3fa4fb454a1d03584d2216982acec4eae1f9d06e635e38f4b8c23c6ba3df6843",
)
TAIL_CLOSURE_REF = dict(
    path=str(REPO / "outputs/tail_translation_reserve_v2_final_closure_20261029.json"),
    sha256="52783b3ef40e8cf2e208277a8d841c2a53553551c971f68f2e33d7e64dc958e8",
)
TAIL_ANALYSIS_REF = dict(
    path=str(REPO / "outputs/tail_translation_reserve_v2_analysis_20261029/result.json"),
    sha256="00cb2a2d740047d4839c417a3d80e807ac92cb15d879fd0f6d3fb74d93e9135c",
)
TAIL_INITIAL_REF = dict(TAIL_PRODUCER_REF, locator="answer.evaluated_candidates[5]")


def case_root(case_name="candidate", *, check=False, owner=False):
    require(case_name == "candidate", "One registered restored-anchor H12 search")
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_restored_anchor_sqp{suffix}_20261029"


def validate_tail_archive(producer, checker, closure, analysis):
    """Validate the admitted scientific outcome; hashes are bound by the caller."""
    require(
        all(row["complete"] is row["passed"] is True for row in (producer, checker, closure, analysis))
        and closure["sources_unchanged"] is analysis["sources_unchanged"] is analysis["source_closure_passed"] is True
        and closure["checker_revision"] == "check_v2"
        and closure["producer_result_ref"] == analysis["producer_ref"] == TAIL_PRODUCER_REF
        and closure["checker_result_ref"] == analysis["checker_ref"] == TAIL_CHECK_REF
        and closure["producer_owner_ref"] == TAIL_OWNER_REF
        and closure["checker_owner_ref"] == TAIL_CHECK_OWNER_REF
        and analysis["closure_ref"] == TAIL_CLOSURE_REF
        and closure["checker_qa_ref"] == PRIOR_QA_REF
        and closure["producer_qa_source_count"] == 95
        and closure["checker_qa_source_count"] == closure["unchanged_producer_source_subset_count"] + 5 == 100
        and closure["checker_qa_test_module_count"] == 30
        and closure["source_count"] == 2485,
        "The original producer and isolated v2 replay have a complete source-bound closure",
    )
    for row in (producer, checker, closure):
        require(
            row["response_evaluations"] == 14
            and row["numerical_inner_steps"] == 4200
            and row["query_count"] == 4369
            and row["witness_indices"] == []
            and row["control_qualified"] is False,
            "All fourteen prior responses, no complete witness or control qualification",
        )
    require(
        checker["verified_responses"] == 14
        and checker["boundary_classification_disagreements"] == closure["boundary_classification_disagreements"] == []
        and all(
            checker[key]["passed"] is True
            for key in ("grid_check", "prior_residual_check", "ledger_check", "independent_restoration_check")
        )
        and closure["total_executed_response_calls"] == analysis["total_executed_response_calls"] == 42
        and closure["total_executed_numerical_inner_steps"] == 12600
        and closure["total_executed_queries"] == 13107
        and closure["preserved_failed_replay"]["returncode"] == 1,
        "Complete numerical and metadata replay with the actual failed-v1 cost retained",
    )
    answer = producer["answer"]
    for row in (
        answer,
        checker["ledger_check"],
        checker["independent_restoration_check"],
        closure,
        analysis["analysis"],
    ):
        require(
            row["improvement_found"] is row["sufficient_reserve_restored"] is True
            and type(row["selected_response_index"]) is int
            and row["selected_response_index"] == 5,
            "The exact closed candidate5 restored sufficient empirical reserve; no fallback selection",
        )
    require(
        len(answer["evaluated_candidates"]) == len(answer["request_ledger"]) == 14
        and answer["causal_checks_all_passed"] is answer["repeatability"]["exact"] is True
        and answer["historical_anchor_comparison"]["comparison"]["exact"] is True
        and answer["selected"]["prefix_length"] == 9
        and answer["selected"]["feasible"] is False
        and json.dumps(answer["selected"], sort_keys=True, allow_nan=False)
        == json.dumps(answer["evaluated_candidates"][5], sort_keys=True, allow_nan=False),
        "Complete selected record is the exact saved request5, with prior prefix but no full witness",
    )
    return answer["selected"]


def load_prerequisites():
    source, registration, _, _, refs = load_tail_prerequisites()
    owner, producer = closed_owner(remember(TAIL_OWNER_REF, refs), refs)
    check_owner, checker = closed_owner(remember(TAIL_CHECK_OWNER_REF, refs), refs)
    require(
        owner["result_ref"] == TAIL_PRODUCER_REF and check_owner["result_ref"] == TAIL_CHECK_REF, "Exact prior owners"
    )
    closure, analysis = (read(remember(ref, refs)) for ref in (TAIL_CLOSURE_REF, TAIL_ANALYSIS_REF))
    validate_tail_archive(producer, checker, closure, analysis)
    for stage, old_owner in (
        ("probe_tracking_v5_tail_translation_reserve", owner),
        ("check_tracking_v5_tail_translation_reserve_v2", check_owner),
    ):
        require(
            read(old_owner["owned_command"]["launch"])["command"]
            == [str(NATIVE_PYTHON), str(REPO / f"scripts/{stage}.py")],
            "Exact already closed producer and repaired replay entries",
        )
    for record in (closure, analysis):
        declared = record["source_refs_unchanged"]
        require(
            len(declared) == len({ref["path"] for ref in declared}) == record["source_count"]
            and {ref["path"]: ref["sha256"] for ref in declared} == record["input_sha256"],
            "Complete prior source map, without dropped or conflicting paths",
        )
        for ref in declared:
            remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return source, registration, producer, refs

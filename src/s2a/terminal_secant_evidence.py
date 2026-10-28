"""Bind finite terminal-row extrapolation to the closed original diagnostic."""

from s2a.directional_derivative_check_v3_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.directional_derivative_check_v3_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.directional_derivative_check_v3_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.directional_derivative_check_v3_evidence import PRODUCER_OWNER_REF as DIAGNOSTIC_OWNER_REF
from s2a.directional_derivative_check_v3_evidence import PRODUCER_REF as DIAGNOSTIC_REF
from s2a.directional_derivative_check_v3_evidence import QA_SOURCES as V3_QA_SOURCES
from s2a.directional_derivative_check_v3_evidence import QA_TESTS as V3_QA_TESTS
from s2a.directional_derivative_check_v3_evidence import REPO, finish, read, reference, remember, require
from s2a.directional_derivative_check_v3_evidence import checked as checked
from s2a.directional_derivative_check_v3_evidence import closed_owner as closed_owner
from s2a.directional_derivative_check_v3_evidence import load_prerequisites as load_v3_prerequisites
from s2a.directional_derivative_check_v3_evidence import model_registration as model_registration
from s2a.directional_derivative_v2_evidence import admit_anchor as admit_anchor

PLAN = REPO / "plans/v0.0.1/terminal_secant_probe_20261029.md"
SOURCES = (
    "src/s2a/terminal_secant_probe.py",
    "src/s2a/terminal_secant_evidence.py",
    "scripts/terminal_secant_audit.py",
    "scripts/probe_tracking_v5_terminal_secant_probe.py",
    "scripts/check_tracking_v5_terminal_secant_probe.py",
    "scripts/execute_tracking_v5_terminal_secant_probe.py",
    "plans/v0.0.1/terminal_secant_probe_20261029.md",
    "tests/test_terminal_secant_probe.py",
    "tests/test_terminal_secant_audit.py",
    "tests/test_terminal_secant_execution.py",
)
QA_SOURCES = (*V3_QA_SOURCES, *SOURCES)
QA_TESTS = (*V3_QA_TESTS, *SOURCES[-3:])
V3_QA_REF = dict(
    path=str(REPO / "outputs/directional_derivative_check_v3_native_qa_20261029_30du0rxo/result.json"),
    sha256="73356f8cc6d75037ab09daa592ab5f21ab28218084dcb442717087fc65332da2",
)
DIAGNOSTIC_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_directional_derivative_v3_check_20261029/verified.json"),
    sha256="74a7e5b8b0ee04b93eb8aa4c90a178418f377c55db1a6e606eddd9d327a2b0cf",
)
DIAGNOSTIC_CHECK_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_directional_derivative_v3_check_execution_20261029/result.json"),
    sha256="e57c508a5f48804a4763e40b53f69965151108e1afe28bd09f4b52c347e6e439",
)
DIAGNOSTIC_CLOSURE_REF = dict(
    path=str(REPO / "outputs/directional_derivative_v3_final_closure_20261029.json"),
    sha256="28b5d3b4a7e107ea1ed5e752927b142904a0a5f7bdf1d5bb9dba64ac2a3ca108",
)
DIAGNOSTIC_ANALYSIS_REF = dict(
    path=str(REPO / "outputs/directional_derivative_v3_analysis_20261029/result.json"),
    sha256="717c5d4758ada20cdaeddf6ec7a120ff42a065fa19693333666afcb84524d4f0",
)


def case_root(*, check=False, owner=False):
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_terminal_secant_probe{suffix}_20261029"


def load_prerequisites():
    original, registration, _, _, refs = load_v3_prerequisites()
    diagnostic_owner, diagnostic = closed_owner(remember(DIAGNOSTIC_OWNER_REF, refs), refs)
    check_owner, check = closed_owner(remember(DIAGNOSTIC_CHECK_OWNER_REF, refs), refs)
    closure = read(remember(DIAGNOSTIC_CLOSURE_REF, refs))
    analysis = read(remember(DIAGNOSTIC_ANALYSIS_REF, refs))
    require(
        diagnostic_owner["result_ref"] == DIAGNOSTIC_REF
        and check_owner["result_ref"] == DIAGNOSTIC_CHECK_REF
        and closure["complete"] is closure["passed"] is closure["sources_unchanged"] is True
        and closure["producer_owner_ref"] == DIAGNOSTIC_OWNER_REF
        and closure["producer_result_ref"] == DIAGNOSTIC_REF
        and closure["checker_owner_ref"] == DIAGNOSTIC_CHECK_OWNER_REF
        and closure["checker_result_ref"] == DIAGNOSTIC_CHECK_REF
        and closure["checker_qa_ref"] == V3_QA_REF
        and closure["qa_source_count"] == 75
        and closure["qa_test_module_count"] == 23
        and diagnostic["response_evaluations"] == check["verified_responses"] == 77
        and check["unique_command_count"] == 76
        and check["numerical_inner_steps"] == 23100
        and check["query_count"] == 24025
        and check["boundary_classification_disagreements"] == []
        and check["saved_stencil_check"]["passed"] is check["historical_anchor_check"]["passed"] is True
        and diagnostic["witness_found"] is check["witness_found"] is False,
        "The original complete77-call diagnostic, independent stencil and historical anchor have closed",
    )
    require(
        analysis["complete"] is analysis["passed"] is analysis["sources_unchanged"] is True
        and analysis["source_closure_passed"] is True
        and analysis["producer_ref"] == DIAGNOSTIC_REF
        and analysis["owner_ref"] == DIAGNOSTIC_OWNER_REF
        and analysis["closure_ref"] == DIAGNOSTIC_CLOSURE_REF
        and analysis["analysis"]["counts"]["directional_pairs"] == 39
        and analysis["control_qualified"] is False
        and all(
            analysis[key] == 0
            for key in ("new_responses", "new_queries", "physics_steps", "world_model_calls", "fresh_cases")
        ),
        "The full saved-grid analysis is source-bound, without new response or control qualification",
    )
    for stage, owner in (("probe", diagnostic_owner), ("check", check_owner)):
        script = (
            "probe_tracking_v5_directional_derivative_v2.py"
            if stage == "probe"
            else "check_tracking_v5_directional_derivative_v3.py"
        )
        require(
            read(owner["owned_command"]["launch"])["command"] == [str(NATIVE_PYTHON), str(REPO / "scripts" / script)],
            "Exact already completed original diagnostic commands",
        )
    for ref in [*closure["source_refs_unchanged"], *analysis["source_refs_unchanged"]]:
        remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return original, registration, diagnostic, closure, refs

"""Bind tail-only restoration to the fully closed finite secant experiment."""

from s2a.terminal_secant_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.terminal_secant_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.terminal_secant_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.terminal_secant_evidence import QA_SOURCES as PRIOR_QA_SOURCES
from s2a.terminal_secant_evidence import QA_TESTS as PRIOR_QA_TESTS
from s2a.terminal_secant_evidence import REPO, finish, read, reference, remember, require
from s2a.terminal_secant_evidence import admit_anchor as admit_anchor
from s2a.terminal_secant_evidence import checked as checked
from s2a.terminal_secant_evidence import closed_owner as closed_owner
from s2a.terminal_secant_evidence import load_prerequisites as load_secant_prerequisites
from s2a.terminal_secant_evidence import model_registration as model_registration

PLAN = REPO / "plans/v0.0.1/tail_translation_reserve_20261029.md"
SOURCES = (
    "src/s2a/tail_translation_reserve.py",
    "src/s2a/tail_translation_reserve_evidence.py",
    "scripts/tail_translation_reserve_audit.py",
    "scripts/probe_tracking_v5_tail_translation_reserve.py",
    "scripts/check_tracking_v5_tail_translation_reserve.py",
    "scripts/execute_tracking_v5_tail_translation_reserve.py",
    "plans/v0.0.1/tail_translation_reserve_20261029.md",
    "tests/test_tail_translation_reserve.py",
    "tests/test_tail_translation_reserve_audit.py",
    "tests/test_tail_translation_execution.py",
)
QA_SOURCES = (*PRIOR_QA_SOURCES, *SOURCES)
QA_TESTS = (*PRIOR_QA_TESTS, *SOURCES[-3:])
PRIOR_QA_REF = dict(
    path=str(REPO / "outputs/terminal_secant_probe_native_qa_20261029_2wj0oaca/result.json"),
    sha256="28db556f5af66b66ad915cd2575ed89cd8e830c5d830b9b6564e534cc1c39019",
)
PRIOR_PRODUCER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_terminal_secant_probe_20261029/verified.json"),
    sha256="39aff2ebda4d272b1b03e0cfe60e5bf26a69de10c4870f55afa4280052b98c34",
)
PRIOR_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_terminal_secant_probe_execution_20261029/result.json"),
    sha256="ffd7763b90f18fd16cbde76bcce5827e102fb948bd118d4439b4788d44ffb380",
)
PRIOR_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_terminal_secant_probe_check_20261029/verified.json"),
    sha256="06214022d72f13423c8829344630ed632e4f9a1b6a40125e0aeb2a2e222464ba",
)
PRIOR_CHECK_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_terminal_secant_probe_check_execution_20261029/result.json"),
    sha256="cdf6e32104efc5f8344c51c5176eee87a3ebe3a80444dcfdff2938c88d1305a3",
)
PRIOR_CLOSURE_REF = dict(
    path=str(REPO / "outputs/terminal_secant_probe_final_closure_20261029.json"),
    sha256="29e3b90d5f0919044abba53102411816f6b78a648a3509a5602a58f64436d779",
)
PRIOR_ANALYSIS_REF = dict(
    path=str(REPO / "outputs/terminal_secant_probe_analysis_20261029/result.json"),
    sha256="44d9c06da3824ab17755750ac195ce60152d15e79fd536d17492d9b33a1268e0",
)


def case_root(*, check=False, owner=False):
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_tail_translation_reserve{suffix}_20261029"


def load_prerequisites():
    original, registration, _, _, refs = load_secant_prerequisites()
    owner, producer = closed_owner(remember(PRIOR_OWNER_REF, refs), refs)
    check_owner, checker = closed_owner(remember(PRIOR_CHECK_OWNER_REF, refs), refs)
    closure = read(remember(PRIOR_CLOSURE_REF, refs))
    analysis = read(remember(PRIOR_ANALYSIS_REF, refs))
    require(
        owner["result_ref"] == PRIOR_PRODUCER_REF
        and check_owner["result_ref"] == PRIOR_CHECK_REF
        and closure["complete"] is closure["passed"] is closure["sources_unchanged"] is True
        and closure["producer_result_ref"] == PRIOR_PRODUCER_REF
        and closure["producer_owner_ref"] == PRIOR_OWNER_REF
        and closure["checker_result_ref"] == PRIOR_CHECK_REF
        and closure["checker_owner_ref"] == PRIOR_CHECK_OWNER_REF
        and closure["qa_ref"] == PRIOR_QA_REF
        and closure["qa_source_count"] == 85
        and closure["qa_test_module_count"] == 26
        and producer["response_evaluations"] == checker["verified_responses"] == 17
        and checker["unique_command_count"] == 16
        and checker["numerical_inner_steps"] == 5100
        and checker["query_count"] == 5305
        and checker["boundary_classification_disagreements"] == []
        and all(checker[key]["passed"] is True for key in ("geometry_check", "fit_check", "ledger_check"))
        and producer["witness_found"] is checker["witness_found"] is False
        and producer["answer"]["extrapolation_passed"] is producer["answer"]["safety_accuracy_passed"] is False,
        "The complete prior17-call secant experiment and its scientific rejection are closed",
    )
    require(
        analysis["complete"] is analysis["passed"] is analysis["sources_unchanged"] is True
        and analysis["source_closure_passed"] is True
        and analysis["producer_ref"] == PRIOR_PRODUCER_REF
        and analysis["checker_ref"] == PRIOR_CHECK_REF
        and analysis["closure_ref"] == PRIOR_CLOSURE_REF
        and analysis["analysis"]["counts"]["validation_points"] == 15
        and analysis["control_qualified"] is False
        and all(
            analysis[key] == 0
            for key in ("new_responses", "new_queries", "physics_steps", "world_model_calls", "fresh_cases")
        ),
        "All15 prior points are source-bound and analyzed without new control activity",
    )
    for stage, old_owner in (("probe", owner), ("check", check_owner)):
        require(
            read(old_owner["owned_command"]["launch"])["command"]
            == [str(NATIVE_PYTHON), str(REPO / f"scripts/{stage}_tracking_v5_terminal_secant_probe.py")],
            "Exact already completed prior experiment entries",
        )
    for ref in [*closure["source_refs_unchanged"], *analysis["source_refs_unchanged"]]:
        remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return original, registration, producer, closure, refs

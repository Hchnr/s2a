"""Admit accepted-point SQP after the exact complete reserve replay."""

from s2a.prefix_optimization_reserve_evidence import (
    BACKTRACKING_REF as BACKTRACKING_REF,
)
from s2a.prefix_optimization_reserve_evidence import (
    CPU_ENVIRONMENT as CPU_ENVIRONMENT,
)
from s2a.prefix_optimization_reserve_evidence import (
    NATIVE_PYTHON as NATIVE_PYTHON,
)
from s2a.prefix_optimization_reserve_evidence import (
    PREFIX_CHECK_REF as PREFIX_CHECK_REF,
)
from s2a.prefix_optimization_reserve_evidence import (
    PREFIX_OPTIMIZATION_MARGIN as PREFIX_OPTIMIZATION_MARGIN,
)
from s2a.prefix_optimization_reserve_evidence import (
    QA_SOURCES as RESERVE_QA_SOURCES,
)
from s2a.prefix_optimization_reserve_evidence import (
    QA_TESTS as RESERVE_QA_TESTS,
)
from s2a.prefix_optimization_reserve_evidence import (
    REPO,
    closed_owner,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.prefix_optimization_reserve_evidence import (
    checked as checked,
)
from s2a.prefix_optimization_reserve_evidence import (
    load_prerequisites as load_reserve_prerequisites,
)
from s2a.prefix_optimization_reserve_evidence import (
    model_registration as model_registration,
)

PLAN = REPO / "plans/v0.0.1/accepted_point_sqp_20261029.md"
SOURCES = (
    "src/s2a/accepted_point_sqp_core.py",
    "src/s2a/accepted_point_sqp_plan.py",
    "src/s2a/accepted_point_sqp_evidence.py",
    "scripts/accepted_point_sqp_math_audit.py",
    "scripts/accepted_point_sqp_ledger_audit.py",
    "scripts/probe_tracking_v5_accepted_point_sqp.py",
    "scripts/check_tracking_v5_accepted_point_sqp.py",
    "scripts/execute_tracking_v5_accepted_point_sqp.py",
    "plans/v0.0.1/accepted_point_sqp_20261029.md",
    "tests/test_accepted_point_sqp_core.py",
    "tests/test_accepted_point_sqp_plan.py",
    "tests/test_accepted_point_sqp_math_audit.py",
    "tests/test_accepted_point_sqp_ledger_audit.py",
    "tests/test_accepted_point_sqp_execution.py",
)
QA_SOURCES = (*RESERVE_QA_SOURCES, *SOURCES)
QA_TESTS = (*RESERVE_QA_TESTS, *SOURCES[-5:])
RESERVE_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_optimization_reserve_execution_20261029/result.json"),
    sha256="b14ff55ef82ebb6b5a28f20f2f76af7244e774bad15b35fafd9a1c50a384c581",
)
RESERVE_PRODUCER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_optimization_reserve_20261029/verified.json"),
    sha256="321a096afd1f0a571cc70f51e3ba3dba6ec3da2973eb83981665b754bc1bcf4a",
)
RESERVE_CHECK_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_optimization_reserve_check_execution_20261029/result.json"),
    sha256="f34e0eaf207c54afb1a679c9c3396b3047793211008b468a0d95bda440a43277",
)
RESERVE_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_optimization_reserve_check_20261029/verified.json"),
    sha256="81f60ce7d9c8ceeb455f2309472c0f8a5854448b158c1bd82a4f028ef2054876",
)
RESERVE_CLOSURE_REF = dict(
    path=str(REPO / "outputs/prefix_optimization_reserve_final_closure_20261029.json"),
    sha256="05e728b1d49919ee3b549ac6d83e66d42b8fd091605cb512d2e19cd36f624ea4",
)
RESERVE_QA_REF = dict(
    path=str(REPO / "outputs/prefix_optimization_reserve_native_qa_20261029_h56dgw22/result.json"),
    sha256="a79441b12430d2988f8de30cc1b973563ee444f29220cf267c60da4e142e1042",
)


def case_root(case_name="candidate", *, check=False, owner=False):
    require(case_name == "candidate", "One registered accepted-point SQP candidate")
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_accepted_point_sqp{suffix}_20261029"


def closed_reserve_replay(refs):
    producer_owner, producer = closed_owner(remember(RESERVE_OWNER_REF, refs), refs)
    owner, report = closed_owner(remember(RESERVE_CHECK_OWNER_REF, refs), refs)
    require(producer_owner["result_ref"] == RESERVE_PRODUCER_REF, "Exact reserve producer")
    require(owner["result_ref"] == RESERVE_CHECK_REF, "Exact complete reserve replay")
    registration = read(remember(owner["registration_ref"], refs))
    producer_registration = read(remember(producer_owner["registration_ref"], refs))
    report_registration = read(remember(report["registration_ref"], refs))
    closure = read(remember(RESERVE_CLOSURE_REF, refs))
    require(
        owner["stage"] == registration["stage"] == "check"
        and producer_owner["stage"] == producer_registration["stage"] == "diagnose"
        and registration["producer_owner_ref"] == report_registration["producer_owner_ref"] == RESERVE_OWNER_REF
        and registration["producer_ref"] == report_registration["producer_ref"] == RESERVE_PRODUCER_REF
        and registration["native_component_qa_ref"]
        == producer_registration["native_component_qa_ref"]
        == RESERVE_QA_REF
        and read(owner["owned_command"]["launch"])["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_prefix_optimization_reserve.py")],
        "Exact closed producer/replay and identical frozen native QA",
    )
    require(
        producer["response_evaluations"] == report["response_evaluations"] == report["verified_responses"] == 4000
        and report["numerical_inner_steps"] == 1200000
        and report["query_count"] == report["mj_forward_queries"] == 1248001
        and report["complete_ray_rollouts_checked"] == 177
        and report["accepted_anchor_audits_checked"] == 54
        and report["boundary_classification_disagreements"] == []
        and report["witness_found"] is report["control_qualified"] is False
        and all(report[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "Complete exact finite reserve replay without a full witness or control qualification",
    )
    require(
        closure["complete"] is closure["passed"] is closure["sources_unchanged"] is True
        and closure["producer_owner_ref"] == RESERVE_OWNER_REF
        and closure["checker_owner_ref"] == RESERVE_CHECK_OWNER_REF
        and closure["checker_result_ref"] == RESERVE_CHECK_REF
        and closure["qa_ref"] == RESERVE_QA_REF
        and closure["verified_responses"] == 4000
        and closure["producer_new_responses_by_role"]
        == dict(warm=1, gradient_probe=3822, objective=53, backtracking=124),
        "Exact terminal source closure and saved response accounting",
    )
    for item in producer["source_refs_unchanged"] + report["source_refs_unchanged"] + closure["source_refs_unchanged"]:
        remember(item, refs)
    finish(refs)
    return dict(producer_ref=RESERVE_PRODUCER_REF, checker_ref=RESERVE_CHECK_REF, closure_ref=RESERVE_CLOSURE_REF)


def load_prerequisites():
    original, registration, backtracking, refs = load_reserve_prerequisites()
    closed_reserve_replay(refs)
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    finish(refs)
    return original, registration, backtracking, refs

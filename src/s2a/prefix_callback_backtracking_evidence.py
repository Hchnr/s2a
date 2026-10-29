"""Admission for one finite ray-accepted search after both full H12 replays."""

from s2a.prefix_trust_asset_repair import (
    ALL_QA_TESTS,
    ORIGINAL_XML_REF,
    PRODUCER_REFS,
    load_asset_repair_prerequisites,
    load_snapshot,
    repair_root,
)
from s2a.prefix_trust_asset_repair import (
    QA_SOURCES as REPAIR_QA_SOURCES,
)
from s2a.prefix_trust_control_evidence import (
    BACKTRACKING_REF as BACKTRACKING_REF,
)
from s2a.prefix_trust_control_evidence import (
    CPU_ENVIRONMENT as CPU_ENVIRONMENT,
)
from s2a.prefix_trust_control_evidence import (
    NATIVE_PYTHON as NATIVE_PYTHON,
)
from s2a.prefix_trust_control_evidence import (
    PREFIX_CHECK_REF as PREFIX_CHECK_REF,
)
from s2a.prefix_trust_control_evidence import (
    REPO,
    closed_owner,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.prefix_trust_control_evidence import (
    checked as checked,
)

PLAN = REPO / "plans/v0.0.1/prefix_callback_backtracking_20261029.md"
SOURCES = (
    "src/s2a/prefix_callback_backtracking_plan.py",
    "src/s2a/prefix_callback_backtracking_evidence.py",
    "scripts/prefix_callback_backtracking_audit.py",
    "scripts/probe_tracking_v5_prefix_callback_backtracking.py",
    "scripts/check_tracking_v5_prefix_callback_backtracking.py",
    "scripts/execute_tracking_v5_prefix_callback_backtracking.py",
    "plans/v0.0.1/prefix_callback_backtracking_20261029.md",
    "tests/test_prefix_callback_backtracking_plan.py",
    "tests/test_prefix_callback_backtracking_audit.py",
    "tests/test_prefix_callback_backtracking_execution.py",
)
QA_SOURCES = (*REPAIR_QA_SOURCES, *SOURCES)
QA_TESTS = (*ALL_QA_TESTS, *SOURCES[-3:])
TRUST_REGISTRATION_REFS = {
    case: dict(path=str(repair_root(case, owner=True) / "registration.json"), sha256=digest)
    for case, digest in {
        "baseline": "3e1f9073aba9522f837b42824d819446be3f2cf0a914793b5cf282abe2f98a77",
        "candidate": "389a0d2ca37f8748b104172052275272c90026cb159e9745910049e900bac107",
    }.items()
}
DIAGNOSIS_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_trust_saved_linearization_execution_20261029/result.json"),
    sha256="fcc29ced5b904a12a5d610512e35bc4ae5a456b86b178316a8a2dadf6a55b369",
)
DIAGNOSIS_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_trust_saved_linearization_20261029/result.json"),
    sha256="0fe281c9c67dfbbc0441586ce1f680baa24539c08a22438f63d0947307f97394",
)
CROSSCHECK_REF = dict(
    path=str(REPO / "outputs/prefix_trust_linearization_crosscheck_20261029/result.json"),
    sha256="08b6ff1f0ffdf061789389402b9918093ad85998cb832a4b5fbd7a1afcb8cfd0",
)
CROSSCHECK_EXIT_REF = dict(
    path=str(REPO / "outputs/prefix_trust_linearization_crosscheck_20261029/exit.json"),
    sha256="acc64e3622fa83ddac246ac7159396c3404934d803eef08765ee7e7d2e770305",
)


def case_root(case_name="candidate", *, check=False, owner=False):
    require(case_name == "candidate", "One registered callback-backtracking candidate")
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_prefix_callback_backtracking{suffix}_20261029"


def closed_trust_replays(refs):
    """Read actual completed results; a live or partial replay cannot admit search."""
    shared = []
    results = {}
    snapshot = load_snapshot()
    for case, count in (("baseline", 4000), ("candidate", 3148)):
        owner_ref = remember(reference(repair_root(case, owner=True) / "result.json"), refs)
        owner, report = closed_owner(owner_ref, refs)
        result_ref = remember(reference(repair_root(case) / "verified.json"), refs)
        registration = read(remember(TRUST_REGISTRATION_REFS[case], refs))
        report_reg = read(remember(report["registration_ref"], refs))
        launch = read(owner["owned_command"]["launch"])
        require(
            owner["registration_ref"] == TRUST_REGISTRATION_REFS[case]
            and owner["result_ref"] == result_ref
            and owner["case_name"] == report["case_name"] == registration["case_name"] == case
            and owner["stage"] == registration["stage"] == "check_v2"
            and report_reg["producer_ref"] == registration["producer_ref"] == PRODUCER_REFS[case]
            and launch["command"]
            == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_prefix_trust_control_v2.py"), "--case", case]
            and report["response_evaluations"] == report["verified_responses"] == count
            and report["numerical_inner_steps"] == count * 300
            and report["query_count"] == report["mj_forward_queries"] == 1 + count * 312
            and report["boundary_classification_disagreements"] == []
            and report["witness_found"] is report["control_qualified"] is False
            and all(report[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
            "Both original complete strict H12 replays and actual successful wait/reap are required",
        )
        proof = report["asset_loading_proof"]
        require(
            registration["snapshot_ref"] == proof["snapshot_ref"] == snapshot["snapshot_ref"]
            and registration["compile_ref"] == proof["compile_ref"] == snapshot["compile_ref"]
            and proof["original_xml_ref"] == ORIGINAL_XML_REF
            and proof["migrated_xml_ref"] == snapshot["migrated_xml_ref"]
            and proof["complete_model_preserved"] is True
            and proof["model_body_or_geom_pruning"] is False
            and proof["asset_count"] == 96,
            "Both independent replays used the same exact complete asset closure",
        )
        for ref in report["source_refs_unchanged"]:
            remember(ref, refs)
        shared.append((registration["native_component_qa_ref"], registration["source_sha256"]))
        results[case] = dict(owner_ref=owner_ref, result_ref=result_ref, registration_ref=TRUST_REGISTRATION_REFS[case])
    require(shared[0] == shared[1], "Both replays share the same frozen repair QA/source closure")
    finish(refs)
    return results


def model_registration(original_registration, refs):
    """Replace only the complete XML locator; retain the original registration."""
    snapshot = load_snapshot()
    require(original_registration["xml_ref"] == ORIGINAL_XML_REF, "Same original causal full model")
    for key in ("snapshot_ref", "compile_ref", "original_xml_ref", "historical_manifest_ref", "migrated_xml_ref"):
        require(refs.get(snapshot[key]["path"]) == snapshot[key], "Full asset proof already admitted")
    proof = {
        key: snapshot[key]
        for key in ("snapshot_ref", "compile_ref", "original_xml_ref", "historical_manifest_ref", "migrated_xml_ref")
    }
    proof.update(asset_count=96, complete_model_preserved=True, model_body_or_geom_pruning=False)
    return dict(original_registration, xml_ref=dict(snapshot["migrated_xml_ref"])), proof


def load_prerequisites():
    original, registration, backtracking, refs = load_asset_repair_prerequisites()
    closed_trust_replays(refs)
    owner, diagnosis = closed_owner(remember(DIAGNOSIS_OWNER_REF, refs), refs)
    require(owner["result_ref"] == DIAGNOSIS_REF, "Exact closed saved-linearization diagnosis")
    remember(DIAGNOSIS_REF, refs)
    require(
        diagnosis["input_ref"] == PRODUCER_REFS["candidate"]
        and diagnosis["data_gaps"] == []
        and diagnosis["summary"]["nonzero_step_count"] == 36
        and diagnosis["summary"]["zero_step_count"] == 9
        and diagnosis["independent_numerical_qualified"] is diagnosis["control_qualified"] is False,
        "Complete saved arithmetic diagnosis, without substituting it for dynamics qualification",
    )
    crosscheck = read(remember(CROSSCHECK_REF, refs))
    exited = read(remember(CROSSCHECK_EXIT_REF, refs))
    require(
        crosscheck["complete"] is crosscheck["passed"] is True
        and crosscheck["groups_checked"] == crosscheck["callbacks_checked"] == 45
        and crosscheck["maximum_numeric_difference"] == crosscheck["maximum_jacobian_difference"] == 0
        and crosscheck["input_and_source_refs"] == crosscheck["source_refs_unchanged"]
        and exited["actual_wait"] is exited["reaped"] is True
        and exited["returncode"] == 0
        and exited["signal_sent"] is exited["automatic_retry"] is False,
        "Closed independent arithmetic crosscheck with unchanged provenance",
    )
    for ref in crosscheck["source_refs_unchanged"]:
        remember(ref, refs)
    remember(exited["stdout_ref"], refs)
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    finish(refs)
    return original, registration, backtracking, refs

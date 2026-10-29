"""Admit one finite derivative diagnostic after the complete SQP replay closes."""

import numpy as np

from s2a.accepted_point_sqp_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.accepted_point_sqp_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.accepted_point_sqp_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.accepted_point_sqp_evidence import QA_SOURCES as SQP_QA_SOURCES
from s2a.accepted_point_sqp_evidence import QA_TESTS as SQP_QA_TESTS
from s2a.accepted_point_sqp_evidence import (
    REPO,
    closed_owner,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.accepted_point_sqp_evidence import case_root as sqp_root
from s2a.accepted_point_sqp_evidence import checked as checked
from s2a.accepted_point_sqp_evidence import load_prerequisites as load_sqp_prerequisites
from s2a.accepted_point_sqp_evidence import model_registration as model_registration

PLAN = REPO / "plans/v0.0.1/directional_derivative_consistency_20261029.md"
SOURCES = (
    "src/s2a/directional_derivative_probe.py",
    "src/s2a/directional_derivative_evidence.py",
    "scripts/directional_derivative_audit.py",
    "scripts/probe_tracking_v5_directional_derivative.py",
    "scripts/check_tracking_v5_directional_derivative.py",
    "scripts/execute_tracking_v5_directional_derivative.py",
    "plans/v0.0.1/directional_derivative_consistency_20261029.md",
    "tests/test_directional_derivative_probe.py",
    "tests/test_directional_derivative_audit.py",
    "tests/test_directional_derivative_execution.py",
)
QA_SOURCES = (*SQP_QA_SOURCES, *SOURCES)
QA_TESTS = (*SQP_QA_TESTS, *SOURCES[-3:])
SQP_OWNER_REF = dict(
    path=str(sqp_root(owner=True) / "result.json"),
    sha256="07301c0ac1fc413dba8f677a5bfbd123af707ed73589dd5e9a932fd807203da6",
)
SQP_PRODUCER_REF = dict(
    path=str(sqp_root() / "verified.json"),
    sha256="66b5b2f89905b7e8be17d92926ffe96015d2433c5f9ce9dc02860870b3d739da",
)
SQP_QA_REF = dict(
    path=str(REPO / "outputs/accepted_point_sqp_native_qa_20261029_ubofewed/result.json"),
    sha256="15fcabb5e5a3f09df41cf0fc7dfce469c011bf54fdf6ff944bde874ee6466143",
)
SQP_CHECK_OWNER_REF = dict(
    path=str(sqp_root(check=True, owner=True) / "result.json"),
    sha256="54844de29f4f153af3a090424ef9a67d047e9e1152351b4b6c79fcf4d90d96ab",
)
SQP_CHECK_REF = dict(
    path=str(sqp_root(check=True) / "verified.json"),
    sha256="01cc69cac324ef4018a3313c634cc41a3dd56e09918c8e1d99b75af5adaf5021",
)
SQP_CLOSURE_REF = dict(
    path=str(REPO / "outputs/accepted_point_sqp_final_closure_20261029.json"),
    sha256="d42f7feb220c98967fd84c5f1f88995af8514afe27a7e90565568dc466ef976b",
)
TERMINAL_DIAGNOSTIC_REF = dict(
    path=str(REPO / "outputs/accepted_point_sqp_terminal_linearization_20261029/result.json"),
    sha256="951eb83b022938d1d908170c8bd8da32bca7a7687ad7b60cc6219e16963db3b9",
)


def case_root(*, check=False, owner=False):
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_directional_derivative{suffix}_20261029"


def closed_sqp_replay(refs):
    producer_owner, producer = closed_owner(remember(SQP_OWNER_REF, refs), refs)
    checker_owner_ref = remember(SQP_CHECK_OWNER_REF, refs)
    checker_owner, checker = closed_owner(checker_owner_ref, refs)
    checker_ref = remember(SQP_CHECK_REF, refs)
    producer_reg = read(remember(producer_owner["registration_ref"], refs))
    checker_reg = read(remember(checker_owner["registration_ref"], refs))
    checker_child_reg = read(remember(checker["registration_ref"], refs))
    require(
        producer_owner["result_ref"] == SQP_PRODUCER_REF
        and checker_owner["result_ref"] == checker_ref
        and producer_owner["stage"] == producer_reg["stage"] == "diagnose"
        and checker_owner["stage"] == checker_reg["stage"] == "check"
        and producer_reg["native_component_qa_ref"] == checker_reg["native_component_qa_ref"] == SQP_QA_REF
        and checker_reg["producer_owner_ref"] == checker_child_reg["producer_owner_ref"] == SQP_OWNER_REF
        and checker_reg["producer_ref"] == checker_child_reg["producer_ref"] == SQP_PRODUCER_REF,
        "Exact normally closed SQP producer and complete replay with the same frozen native QA",
    )
    for stage, owner in (("probe", producer_owner), ("check", checker_owner)):
        require(
            read(owner["owned_command"]["launch"])["command"]
            == [str(NATIVE_PYTHON), str(REPO / f"scripts/{stage}_tracking_v5_accepted_point_sqp.py")],
            "Original registered SQP commands",
        )
    ledger = checker["ledger_check"]
    require(
        producer["response_evaluations"] == checker["response_evaluations"] == checker["verified_responses"] == 1919
        and producer["numerical_inner_steps"] == checker["numerical_inner_steps"] == 575700
        and producer["query_count"] == checker["query_count"] == checker["mj_forward_queries"] == 598729
        and checker["complete_ray_rollouts_checked"] == 118
        and checker["accepted_anchor_audits_checked"] == 25
        and len(ledger["qp_certificate_checks"]) == 27
        and len(ledger["curvature_checks"]) == 24
        and ledger["total_qp_iterations"] == 466
        and all(item["passed"] for item in ledger["qp_certificate_checks"] + ledger["curvature_checks"])
        and checker["boundary_classification_disagreements"] == []
        and checker["witness_found"] is checker["control_qualified"] is False
        and all(checker[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "All SQP responses, rays, anchors and algebra fully verified without control qualification",
    )
    closure_ref = remember(SQP_CLOSURE_REF, refs)
    closure = read(closure_ref)
    require(
        closure["complete"] is closure["passed"] is closure["sources_unchanged"] is True
        and closure["producer_owner_ref"] == SQP_OWNER_REF
        and closure["producer_result_ref"] == SQP_PRODUCER_REF
        and closure["checker_owner_ref"] == checker_owner_ref
        and closure["checker_result_ref"] == checker_ref
        and closure["qa_ref"] == SQP_QA_REF
        and closure["verified_responses"] == 1919
        and closure["qp_certificate_checks"] == 27
        and closure["curvature_checks"] == 24
        and closure["witness_found"] is closure["control_qualified"] is False,
        "Final complete SQP source and lifecycle closure",
    )
    diagnostic = read(remember(TERMINAL_DIAGNOSTIC_REF, refs))
    require(
        diagnostic["complete"] is diagnostic["passed"] is diagnostic["source_closure_passed"] is True
        and diagnostic["producer_ref"] == SQP_PRODUCER_REF
        and diagnostic["owner_ref"] == SQP_OWNER_REF
        and diagnostic["diagnosis"]["ray_count"] == 27,
        "Exact saved terminal linearization diagnosis",
    )
    remember(closure["driver_ref"], refs)
    remember(diagnostic["driver_ref"], refs)
    for ref in producer["source_refs_unchanged"] + checker["source_refs_unchanged"] + closure["source_refs_unchanged"]:
        remember(ref, refs)
    for ref in read(remember(diagnostic["source_refs_ref"], refs)):
        remember(ref, refs)
    finish(refs)
    return producer, dict(checker_ref=checker_ref, checker_owner_ref=checker_owner_ref, closure_ref=closure_ref)


def load_prerequisites():
    original, registration, _, refs = load_sqp_prerequisites()
    producer, closure = closed_sqp_replay(refs)
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    finish(refs)
    return original, registration, producer, closure, refs


def saved_stencil(answer):
    """Bind the fixed archived commands and reconstruct the original precision FD."""
    trace, requests = answer["evaluated_candidates"], answer["request_ledger"]
    anchor = np.asarray(trace[1819]["commands"], dtype=float)
    require(anchor.shape == (12, 6) and np.isfinite(anchor).all(), "Fixed complete anchor1819")
    directions, probe_indices, deltas = [], None, None
    for index in (24, 25, 26):
        phase = answer["phases"][index]
        linear = phase["linearization"]
        require(
            phase["stage_index"] == index and linear["complete"] is True and linear["base_response_index"] == 1819,
            "Each fixed stage differentiates the same complete anchor",
        )
        displacement = np.asarray(phase["qp_direction"]["commands"], dtype=float) - anchor
        require(
            displacement.shape == (12, 6)
            and np.isfinite(displacement).all()
            and displacement.tobytes() == np.asarray(phase["qp_direction"]["displacement"], dtype=float).tobytes()
            and np.max(np.abs(displacement)) > 0,
            "Exact nonzero saved QP displacement",
        )
        directions.append(displacement)
        rows = requests[phase["requests_before"] : phase["generation_requests_after"]]
        require(
            len(rows) == 74
            and [row["role"] for row in rows[:2]] == ["stage_anchor", "gradient_base"]
            and all(row["response_index"] == 1819 and row["outcome"] == "cache_hit" for row in rows[:2]),
            "One complete fixed anchor plus72 axis stencil",
        )
        indices, nominal = [], []
        for axis, row in enumerate(rows[2:]):
            delta, response_index = row["nominal_delta"], row["response_index"]
            require(
                row["role"] == "gradient_probe"
                and row["axis"] == axis
                and row["stage_index"] == index
                and abs(delta) == 1e-4
                and type(response_index) is int
                and 0 <= response_index < len(trace),
                "Fixed ordered original one-sided FD probe",
            )
            expected = anchor.ravel().copy()
            expected[axis] += delta
            require(
                expected.tobytes().hex() == row["command_key"]
                and expected.tobytes() == np.asarray(trace[response_index]["commands"], dtype=float).tobytes(),
                "Original FD command bytes and response association",
            )
            indices.append(response_index)
            nominal.append(delta)
        require(
            indices == linear["probe_response_indices"] and nominal == linear["nominal_deltas"],
            "Saved linearization agrees with actual request routing",
        )
        if probe_indices is not None:
            require(indices == probe_indices and nominal == deltas, "Three stages share the same saved stencil")
        probe_indices, deltas = indices, nominal

    def squared(entry):
        errors = np.asarray(entry["tcp_errors"], dtype=float)
        require(errors.shape == (12, 6) and np.isfinite(errors).all(), "Finite archived TCP errors")
        return np.column_stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
        ).ravel()

    base_safety, base_q = np.asarray(trace[1819]["safety_slack"], dtype=float), squared(trace[1819])
    Js = np.column_stack(
        [(np.asarray(trace[i]["safety_slack"]) - base_safety) / h for i, h in zip(probe_indices, deltas, strict=True)]
    )
    Jq = np.column_stack([(squared(trace[i]) - base_q) / h for i, h in zip(probe_indices, deltas, strict=True)])
    require(np.isfinite(Js).all() and np.isfinite(Jq).all(), "Finite full original-precision Jacobians")
    return anchor.copy(), np.asarray(directions), Js, Jq


def compare_saved_stencil(answer, Js, Jq):
    """Retain full predictions and residuals, including explicit rejected requests."""
    base = answer["evaluated_candidates"][0]
    fields = ("safety_slack", "normalized_squared_errors")
    unit_predictions = []
    for stage, unit in zip(answer["source_stages"], answer["unit_directions"], strict=True):
        unit = np.asarray(unit).ravel()
        unit_predictions.append(
            dict(source_stage=stage, safety_slack=Js @ unit, normalized_squared_errors=(Jq @ unit).reshape(12, 2))
        )
    request_predictions = []
    for request in answer["request_ledger"]:
        delta = np.asarray(request["actual_displacement"]).ravel()
        predicted = dict(safety_slack=Js @ delta, normalized_squared_errors=(Jq @ delta).reshape(12, 2))
        observed, residual = None, None
        if request["response_index"] is not None:
            entry = answer["evaluated_candidates"][request["response_index"]]
            observed = {field: np.asarray(entry[field]) - np.asarray(base[field]) for field in fields}
            residual = {field: observed[field] - predicted[field] for field in fields}
        request_predictions.append(
            dict(
                request_index=request["request_index"],
                predicted_change=predicted,
                observed_change=observed,
                residual=residual,
            )
        )
    return dict(
        anchor_response_index=1819,
        source_stages=[24, 25, 26],
        jacobian_safety_slack=Js,
        jacobian_normalized_squared_errors=Jq,
        unit_predictions=unit_predictions,
        request_predictions=request_predictions,
    )

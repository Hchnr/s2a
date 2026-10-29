"""Metadata-only provenance for one registered saved-direction experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/tracking_v5_saved_direction_backtracking_20261028"
CHECK_ROOT = REPO / "outputs/tracking_v5_saved_direction_backtracking_check_20261028"
PLAN = REPO / "plans/v0.0.1/saved_direction_backtracking_20261028.md"
NATIVE_PYTHON = REPO / "outputs/libero_env311/bin/python"
DIAGNOSTIC_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_differences_20261028/result.json"),
    sha256="f918aaf2709fd3bce09ef90d32e92a5f7c255bb753e502c9caea3fb6d954e5db",
)
CASE_STEMS = dict(
    prefix="prefix_preserving_trajectory_control",
    early="paired_reference_phase_early",
    late="paired_reference_phase_late",
)
CHECK_OWNER_SHAS = dict(
    prefix="a494d9b06ae45f7aaea6a587029d0d4c77a30f5b83a09d2fe07f0082a728ed74",
    early="1dae567731040e13d825bc025d4c9fdceafefd04416ba8b7ad448c4b2bf357b5",
    late="393eb2ee31e4f2d385f8375d1b30c945e4b24e324fa74dc2bb2efa5975471613",
)
CPU_ENVIRONMENT = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
)
QA_TESTS = (
    "tests/test_saved_direction_backtracking.py",
    "tests/test_saved_direction_backtracking_execution.py",
    "tests/test_saved_direction_backtracking_independent_check.py",
)
SOURCES = (
    "src/s2a/saved_direction_backtracking.py",
    "src/s2a/saved_direction_backtracking_evidence.py",
    "scripts/probe_tracking_v5_saved_direction_backtracking.py",
    "scripts/check_tracking_v5_saved_direction_backtracking.py",
    "scripts/execute_tracking_v5_saved_direction_backtracking.py",
    "plans/v0.0.1/saved_direction_backtracking_20261028.md",
    *QA_TESTS,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact path/SHA reference required")
    path = Path(ref["path"])
    require(path.is_absolute() and reference(path) == ref, "Evidence path/SHA mismatch")
    return path


def read(ref):
    path = checked(ref)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Evidence changed while reading")
    return json.loads(payload)


def remember(ref, refs):
    require(refs.get(ref["path"], ref) == ref, "One immutable digest per input/source")
    if ref["path"] not in refs:
        checked(ref)
        refs[ref["path"]] = ref
    return ref


def finish(refs):
    for ref in refs.values():
        checked(ref)


def closed_owner(ref, refs):
    """Bind a complete CPU result to real child wait/reap and unchanged sources."""
    owner = read(remember(ref, refs))
    require(owner["complete"] is owner["passed"] is True, "Prerequisite owner must have completed successfully")
    for item in [owner["registration_ref"], owner["result_ref"], *owner["owned_command"].values()]:
        remember(item, refs)
    launch = read(owner["owned_command"]["launch"])
    exited = read(owner["owned_command"]["exit"])
    release = read(owner["owned_command"]["release"])
    require(
        launch["automatic_retry"] is exited["automatic_retry"] is False
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and exited["actual_wait"] is True
        and exited["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"],
        "Matching CPU launch and actual successful wait/reap without signals or retry",
    )
    result = read(owner["result_ref"])
    require(result["complete"] is result["passed"] is True, "Complete prerequisite result")
    for path, digest in read(owner["registration_ref"])["source_sha256"].items():
        remember(dict(path=path, sha256=digest), refs)
    return owner, result


def load_prerequisites():
    """Read fixed old evidence only; no dynamics, simulator or new output."""
    refs = {}
    diagnostic = read(remember(DIAGNOSTIC_REF, refs))
    require(diagnostic["complete"] is diagnostic["passed"] is True, "Completed saved-difference diagnosis")
    require(
        diagnostic["source_refs_before"] == diagnostic["source_refs_after"]
        and diagnostic["input_refs_before"] == diagnostic["input_refs_after"]
        and diagnostic["control_qualified"] is diagnostic["retiming_qualified"] is False
        and all(
            diagnostic[k] == 0
            for k in (
                "new_forward_responses",
                "new_dynamics_queries",
                "physics_steps",
                "world_model_calls",
                "fresh_cases",
            )
        ),
        "Read-only diagnosis with unchanged provenance and no control qualification",
    )
    diagnosis_reg = read(remember(diagnostic["registration_ref"], refs))
    for item in [*diagnostic["input_refs_after"], *diagnostic["source_refs_after"], diagnosis_reg["qa_ref"]]:
        remember(item, refs)
    producers, registrations = {}, {}
    for name, stem in CASE_STEMS.items():
        horizon = 12 if name == "prefix" else 13
        check_owner_ref = dict(
            path=str(REPO / f"outputs/tracking_v5_{stem}_check_execution_20261028/result.json"),
            sha256=CHECK_OWNER_SHAS[name],
        )
        check_owner, verification = closed_owner(check_owner_ref, refs)
        check_reg = read(remember(verification["registration_ref"], refs))
        old_owner, producer = closed_owner(check_reg["producer_owner_ref"], refs)
        producer_ref = diagnosis_reg["producer_refs"][name]
        require(
            old_owner["result_ref"] == check_reg["producer_ref"] == producer_ref
            and verification["trace_count"] == 4000
            and verification["numerical_inner_steps"] == 4000 * horizon * 25
            and verification["mj_forward_queries"] == 4000 * horizon * 26 + 1
            and verification["predicted_control_periods"] == 4000 * horizon
            and verification["boundary_classification_disagreements"] == []
            and verification["witness_found"] is verification["control_qualified"] is False,
            "Exact fully reconstructed registered trace without a witness",
        )
        owner_reg = read(check_owner["registration_ref"])
        require(owner_reg["stage"] == "check", "Full independent checker owner")
        for item in verification["source_refs_unchanged"]:
            remember(item, refs)
        registrations[name] = read(remember(producer["registration_ref"], refs))
        producers[name] = producer
        require(registrations[name]["endpoint"] == 368 and registrations[name]["horizon"] == horizon, "Same actual368")
    common = (
        "current_q",
        "current_velocity",
        "fixed_gripper_state",
        "previous_command",
        "previous_site",
        "previous_physical_delta",
        "latched_rotation_goal",
        "initial_joint",
        "kp",
        "kd",
        "joint_limits",
        "torque_limits",
        "site_from_tcp",
        "past_inner_error_rad",
        "terminal_normals",
    )
    for name in ("early", "late"):
        require(
            all(np.array_equal(producers[name][key], producers["prefix"][key]) for key in common), "Same causal inputs"
        )
        for key in ("chunks_ref", "xml_ref", "control_ref", "states_ref", "original_rows_ref", "tool_ref"):
            require(registrations[name][key] == registrations["prefix"][key], "Same registered causal source")
        require(producers[name]["case_contract"]["original_node_endpoint_indices"][9] == 10, "Original cursor9 mapping")
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    for relative in (
        "src/s2a/native_environment.py",
        "src/s2a/preview_result_publication.py",
        "src/s2a/artifacts.py",
        "scripts/run_wam_readout.py",
        "src/s2a/worker_handshake.py",
        "src/s2a/paired_reference_phase.py",
    ):
        remember(reference(REPO / relative), refs)
    return diagnostic, producers, registrations, refs

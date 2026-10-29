"""Bind one new check-only replay to the original producer and failed checker."""

import copy
from pathlib import Path

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, require

CONFIG = REPO / "configs/tracking/trajectory_search/state_vector_dual_recovery_fresh_diagnostic_v2_20261029.json"
OUTPUT = REPO / "outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_v2_20261029"
OWNER = REPO / "outputs/state_vector_dual_recovery_fresh_diagnostic_v2_owner_20261029"
PRIOR_REFS = {
    "producer_ref": (
        "outputs/tracking_v5_state_vector_dual_recovery_20261029/verified.json",
        "436fbe20f20cf6678aa329c05e928845b563f76cb0a976c11395e16481aba551",
    ),
    "producer_registration_ref": (
        "outputs/tracking_v5_state_vector_dual_recovery_20261029/registration.json",
        "039029faf7b34d34ba3c7f679da0a5a638f6af6c32d1a3738f279917d752e797",
    ),
    "failed_checker_ref": (
        "outputs/tracking_v5_state_vector_dual_recovery_check_20261029/verified.json",
        "6036f0cd4a711ef44b556b489cf946252d2d30e8dafb8cb232cfef0c4d4fcb85",
    ),
    "failed_owner_ref": (
        "outputs/state_vector_dual_recovery_checker_recovery_v2_20261029/result.json",
        "cc55b1626b3ef7c3667850eb9d80f86fe538e82c030da06e21b4da382dcb60e1",
    ),
    "initial_failed_owner_ref": (
        "outputs/state_vector_dual_recovery_checker_20261029/result.json",
        "2ec7846a7b13adcc25163af6060c69c74598eab2f279d990d548e7c2d646709f",
    ),
    "initial_diagnostic_config_ref": (
        "configs/tracking/trajectory_search/state_vector_dual_recovery_fresh_diagnostic_20261029.json",
        "3c04681e45944d2ac0f47e3aa2bd502c52195e7f4e8c3341b7b202286d3912d7",
    ),
    "initial_diagnostic_checker_ref": (
        "outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_20261029/verified.json",
        "c9d714e9e9b9a78cda7f6a229eecea41f3ea2a035728399e89f3b8c4049c1cd5",
    ),
    "initial_diagnostic_owner_ref": (
        "outputs/state_vector_dual_recovery_fresh_diagnostic_owner_20261029/result.json",
        "27613d5c82404fceb059cb9f6a0018a0a1475d309067f6bf826573d6fb86697e",
    ),
}
CURRENT_SOURCES = (
    "scripts/tracking/trajectory_search/check_tracking_v5_state_vector_fresh_diagnostic.py",
    "scripts/tracking/trajectory_search/execute_tracking_v5_state_vector_fresh_diagnostic.py",
    "scripts/tracking/trajectory_search/state_vector_fresh_diagnostic_admission.py",
    "scripts/verification/trajectory_search/native_replay_inputs.py",
    "tests/research/trajectory_search/test_fresh_decision_diagnostics.py",
    "tests/research/trajectory_search/test_native_replay_inputs.py",
    "tests/research/trajectory_search/test_fresh_diagnostic_admission.py",
)
QA_TARGETS = (
    "tests/research/trajectory_search/test_state_vector_gn_search_execution.py",
    "tests/research/trajectory_search/test_state_vector_dual_recovery_execution.py",
    "tests/research/trajectory_search/test_state_vector_dual_recovery_audit.py",
    *CURRENT_SOURCES[-3:],
)


def validate_spec(spec):
    """Reject changing an old failure, output, budget or numerical convention."""
    expected = dict(
        schema_version="s2a.state-vector-fresh-diagnostic.v2",
        seed=42,
        output_dir=str(OUTPUT.relative_to(REPO)),
        expected_responses=122,
        expected_unique_commands=121,
        expected_inner_steps=36600,
        expected_queries=38065,
        numeric_tolerance=1e-10,
        torque_tolerance=1e-8,
        timeout_seconds=1800,
        automatic_retry=False,
        tool_transform_policy="original_independent_solve",
        source_paths=list(CURRENT_SOURCES),
    )
    require(isinstance(spec, dict) and set(spec) == set(expected) | set(PRIOR_REFS) | {"plan_ref"}, "Exact replay keys")
    for key, value in expected.items():
        require(type(spec[key]) is type(value) and spec[key] == value, "Fixed diagnostic contract: " + key)
    for key, (relative, digest) in PRIOR_REFS.items():
        require(
            spec[key] == dict(path=str(REPO / relative), sha256=digest), "Original failed/producer identity: " + key
        )
    plan = spec["plan_ref"]
    require(
        isinstance(plan, dict)
        and set(plan) == {"path", "sha256"}
        and plan["path"]
        == str(
            REPO / "plans/v0.0.1/tracking/trajectory_search/state_vector_dual_recovery_fresh_diagnostic_v2_20261029.md"
        ),
        "Explicit new diagnostic plan",
    )
    return copy.deepcopy(spec)


def admit_replay(context, path):
    require(Path(path) == CONFIG, "One fixed diagnostic configuration")
    config_ref = context.bind_current_source(path)
    spec = validate_spec(context.read_current_json(config_ref))
    context.checked_current(spec["plan_ref"])
    for key in PRIOR_REFS:
        context.checked_current(spec[key])
    old = context.read_current_json(spec["failed_checker_ref"])
    require(
        old["complete"] is old["passed"] is False
        and old["response_evaluations"] == old["verified_responses"] == 122
        and old["numerical_inner_steps"] == 36600
        and old["query_count"] == 38065
        and old["qp_solves"] == old["nnls_fit_attempts"] == 0,
        "Original fully replayed checker failure remains a failure",
    )
    first = context.read_current_json(spec["initial_diagnostic_checker_ref"])
    require(
        first["complete"] is first["passed"] is False
        and first["response_evaluations"] == first["verified_responses"] == first["query_count"] == 0
        and first["source_validation_after"] is True,
        "First diagnostic failed before numerical responses; preserve that boundary",
    )
    for relative in CURRENT_SOURCES:
        context.bind_current_source(REPO / relative)
    context.finish()
    return dict(config_ref=config_ref, config=spec)

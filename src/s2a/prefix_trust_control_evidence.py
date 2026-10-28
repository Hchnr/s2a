"""Read-only admission for the registered matched-warm H12 control comparison."""

import numpy as np

from s2a.saved_direction_backtracking_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.saved_direction_backtracking_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.saved_direction_backtracking_evidence import REPO as REPO
from s2a.saved_direction_backtracking_evidence import checked as checked
from s2a.saved_direction_backtracking_evidence import closed_owner as closed_owner
from s2a.saved_direction_backtracking_evidence import finish as finish
from s2a.saved_direction_backtracking_evidence import load_prerequisites as original_prerequisites
from s2a.saved_direction_backtracking_evidence import read as read
from s2a.saved_direction_backtracking_evidence import reference as reference
from s2a.saved_direction_backtracking_evidence import remember as remember
from s2a.saved_direction_backtracking_evidence import require as require

PLAN = REPO / "plans/v0.0.1/prefix_trust_region_control_20261028.md"
BACKTRACKING_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_direction_backtracking_20261028/verified.json"),
    sha256="0957f81930cd65d08761220c76000374e1b7c9826872145319e311a42e26028e",
)
BACKTRACKING_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_direction_backtracking_check_v2_20261028/verified.json"),
    sha256="f41948e32a766d2ad3b29528f247527c6286ec37230f8f4fc575f59ac7d58136",
)
BACKTRACKING_CHECK_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_direction_backtracking_check_v2_execution_20261028/result.json"),
    sha256="c19227f75375c94fae8d20319e3cee68f9856674e4fb73ab6e6acfae7f3d6600",
)
PREFIX_CHECK_REF = dict(
    path=str(REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_check_20261028/verified.json"),
    sha256="dde839ae104d17fd469409ec539b780ebb829d0e5dca9052c755fe0fefe6e831",
)
QA_TESTS = (
    "tests/test_prefix_trust_region_plan.py",
    "tests/test_prefix_trust_region_independent_check.py",
    "tests/test_prefix_trust_control_execution.py",
)
SOURCES = (
    "src/s2a/prefix_trust_region_plan.py",
    "src/s2a/prefix_trust_control_evidence.py",
    "scripts/probe_tracking_v5_prefix_trust_control.py",
    "scripts/check_tracking_v5_prefix_trust_control.py",
    "scripts/execute_tracking_v5_prefix_trust_control.py",
    "plans/v0.0.1/prefix_trust_region_control_20261028.md",
    *QA_TESTS,
)


def case_root(case_name, *, check=False, owner=False):
    require(case_name in ("baseline", "candidate"), "One registered comparison arm")
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_prefix_trust_{case_name}{suffix}_20261028"


def load_prerequisites():
    """Bind complete independent backtracking, original causality and one exact warm."""
    _, originals, registrations, refs = original_prerequisites()
    owner, verification = closed_owner(remember(BACKTRACKING_CHECK_OWNER_REF, refs), refs)
    require(owner["result_ref"] == BACKTRACKING_CHECK_REF, "Exact complete repaired independent audit")
    remember(BACKTRACKING_CHECK_REF, refs)
    require(
        verification["response_evaluations"] == 96
        and verification["numerical_inner_steps"] == 29700
        and verification["query_count"] == 30891
        and verification["boundary_classification_disagreements"] == []
        and verification["control_qualified"] is False,
        "All96 responses independently verified without relaxed classification or physical qualification",
    )
    check_reg = read(remember(verification["registration_ref"], refs))
    producer_owner, backtracking = closed_owner(check_reg["producer_owner_ref"], refs)
    require(
        producer_owner["result_ref"] == check_reg["producer_ref"] == BACKTRACKING_REF,
        "Exact closed saved-direction producer",
    )
    for ref in [*verification["source_refs_unchanged"], *backtracking["source_refs_unchanged"]]:
        remember(ref, refs)
    old = originals["prefix"]
    selected = backtracking["cases"]["prefix"]["entries"][55]
    request, metrics = selected["request"], selected["classification"]
    require(
        request["case_name"] == "prefix"
        and request["base_index"] == 3246
        and request["trial_index"] == 3319
        and request["alpha"] == 0.0625
        and request["horizon"] == 12
        and request["target_index"] == 9
        and metrics["prefix_length"] == 9
        and metrics["joint_feasible"] is metrics["buffer_satisfied"] is True
        and metrics["feasible_descent"] is True
        and metrics["feasible"] is False,
        "One registered safe H12 warm; it does not already solve cursor9",
    )
    trace = old["answer"]["evaluated_candidates"]
    base, trial = np.asarray(trace[3246]["commands"]), np.asarray(trace[3319]["commands"])
    warm = np.asarray(request["commands"], dtype=float)
    require(warm.tobytes() == (base + 0.0625 * (trial - base)).tobytes(), "Exact registered warm command bytes")
    checked(PREFIX_CHECK_REF)
    remember(PREFIX_CHECK_REF, refs)
    for relative in SOURCES:
        remember(reference(REPO / relative), refs)
    for relative in (
        "src/s2a/prefix_preserving_trajectory_plan.py",
        "scripts/check_tracking_v5_saved_direction_backtracking.py",
        "scripts/check_tracking_v5_saved_direction_backtracking_v2.py",
        "scripts/check_tracking_v5_prefix_preserving_trajectory_control.py",
        "src/s2a/native_goal_trajectory.py",
        "src/s2a/mujoco_robot_dynamics.py",
        "scripts/run_wam_readout.py",
    ):
        remember(reference(REPO / relative), refs)
    finish(refs)
    return old, registrations["prefix"], backtracking, refs

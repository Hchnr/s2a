"""Compare the experimental branchpoint with the original successful history."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_correct_reference_same_state_arm as trial
from scripts.simulation.runtime import run_reset_correct_reference as original

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/correct_reference_same_state_arm_20261116/branchpoint_history_audit.json"
PRODUCER = {
    "path": str(ROOT / "outputs/correct_reference_same_state_arm_20261116/producer/result.json"),
    "sha256": "b87c7738629a8e728ca20ae4e2600f1ebd4233233eaf2c9f44ad38248536f89d",
}
CHECKER = {
    "path": str(ROOT / "outputs/correct_reference_same_state_arm_20261116/checker/result.json"),
    "sha256": "7cecd6e3e9697584d98f49b60de4fc1744e9defeaeb815f38809a9729ea72f8b",
}
COHORT_DRY = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_dry_20261116.json"),
    "sha256": "b1fa5eadf94aa475386d0fc1762da32b2df0152d4da8c5e75ee44b5099874cd5",
}
COHORT_AUDIT = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_20261116/audit.json"),
    "sha256": "dbe7d296758cc7cb9c9f66168ed140b13ed2967dbb84e5879044c3e849ec704a",
}
DIVERGENCE = {
    "path": str(ROOT / "outputs/correct_reference_failure_diagnosis_20261116/result.json"),
    "sha256": "fb37341ecfae367a5e8b7c3ace7c6b64b1d82541311675b857f159767945d897",
}
FIELDS = (
    "controller_joint_pos",
    "controller_joint_vel",
    "controller_goal_pos",
    "controller_goal_ori",
    "gripper_qpos",
    "opening",
    "gripper_current_action",
)


def compare_history(baseline, current, old_positions, new_positions, *, dt):
    """Report measured branchpoint differences and one-period mean velocities."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("A positive finite endpoint interval is required")
    differences = {}
    for name in FIELDS:
        a, b = np.asarray(baseline[name], dtype=float), np.asarray(current[name], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Finite matching branchpoint fields required: {name}")
        differences[name] = float(np.max(np.abs(a - b)))
    old, new = np.asarray(old_positions, dtype=float), np.asarray(new_positions, dtype=float)
    if old.shape != (2, 3) or new.shape != (2, 3) or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("Two finite object positions from each history are required")
    return dict(
        field_maximum_absolute_differences=differences,
        object_position_difference_m=float(np.linalg.norm(new[-1] - old[-1])),
        previous_period_mean_velocity_difference_m_s=float(np.linalg.norm((new[-1] - new[0] - old[-1] + old[0]) / dt)),
        endpoint_interval_s=float(dt),
    )


def main():
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    producer, checker, dry, cohort, divergence = (
        prefix.read(ref) for ref in (PRODUCER, CHECKER, COHORT_DRY, COHORT_AUDIT, DIVERGENCE)
    )
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True
        and cohort["complete"] is cohort["passed"] is dry["complete"] is dry["passed"] is True
        and divergence["complete"] is divergence["passed"] is True,
        "Closed source-bound physical and original cohort receipts are required",
    )
    produced, checked = ([prefix.read(ref) for ref in group["branch_refs"]] for group in (producer, checker))
    current = produced[0]["runtime"]["continuation"]["branchpoint"]
    for item in (produced[1], *checked):
        trial.compare_branchpoints(current, item["runtime"]["continuation"]["branchpoint"])
    identity, entry = dry["inputs"]["cases"][3], cohort["cases"][3]
    diagnostic = divergence["cases"][3]
    prefix.require(
        identity["spec"] == entry["case"] == diagnostic["case"]
        and entry["replayed"] is True
        and diagnostic["producer_ref"] == entry["producer_ref"],
        "Original case 03 and its independently mapped object state are required",
    )
    candidate = prefix.read(entry["producer_ref"])
    candidate_periods = original.periods(candidate)
    rows = prefix.read(identity["refs"]["rows"])
    step = 40
    sample = current["sample"]
    np.testing.assert_array_equal(sample["sim_states"], candidate_periods[step - 1]["actual_state"])
    with np.load(prefix.checked(identity["refs"]["arrays"]), allow_pickle=False) as archive:
        baseline = {name: np.asarray(archive[name][step]).copy() for name in FIELDS if name != "gripper_current_action"}
        offsets = np.asarray(archive["gripper_current_action_offsets"][step : step + 2])
        baseline["gripper_current_action"] = np.asarray(
            archive["gripper_current_action_flat"][offsets[0] : offsets[1]]
        ).copy()
        dt = float(archive["sim_time"][step] - archive["sim_time"][step - 1])
    candidate_dt = float(
        candidate_periods[step - 1]["endpoint_receipt"]["sim_time"]
        - candidate_periods[step - 2]["endpoint_receipt"]["sim_time"]
    )
    np.testing.assert_allclose([dt, candidate_dt], [0.05, 0.05], atol=1e-12, rtol=0)
    target = "akita_black_bowl_1"
    old_positions = [rows[index]["objects"][target]["position"] for index in (step - 2, step - 1)]
    columns = diagnostic["metrics"]["state_position_mapping"]["columns"]
    new_positions = [np.asarray(candidate_periods[index]["actual_state"])[columns] for index in (step - 2, step - 1)]
    metrics = compare_history(baseline, sample, old_positions, new_positions, dt=dt)
    prefix.require(snapshot(ROOT)["sources"] == before, "Branchpoint audit sources changed")
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                source_unchanged=True,
                exploratory_after_outcome=True,
                metrics=metrics,
                producer_ref=PRODUCER,
                checker_ref=CHECKER,
                cohort_dry_ref=COHORT_DRY,
                cohort_audit_ref=COHORT_AUDIT,
                divergence_ref=DIVERGENCE,
                source_ref=prefix.reference(Path(__file__).resolve()),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

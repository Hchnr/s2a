"""Audit the opening-lead physical run and replay against the unmodified H40."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as original
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check

ROOT = prefix.REPO
PRODUCER = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_20261112/producer/result.json"),
    "sha256": "b0284288c78f39e445fb94983733e22de74ed6d1a6c3f15a0db015a09a1734db",
}
CHECKER = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_20261112/checker/result.json"),
    "sha256": "25e540795861a87a86bff50a37ec2a81d9677182b06983d542a83e3e9b943441",
}
BASELINE = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
OUTPUT = ROOT / "outputs/bowl_h40_opening_lead_audit_20261112/result.json"


def compare_openings(baseline, candidate, model_openings):
    """Compare actual openings to the same original model targets and the led targets."""
    model = np.asarray(model_openings, dtype=float)
    prefix.require(
        model.shape == (40,)
        and np.isfinite(model).all()
        and np.all((model >= 0) & (model <= 1))
        and len(baseline) == len(candidate) == 40,
        "A full, finite H40 and both 40-period traces are required",
    )
    points = []
    for index, (old, new) in enumerate(zip(baseline, candidate, strict=True)):
        old_score, new_score = old["completed_feedback"]["score"], new["completed_feedback"]["score"]
        reference, shifted = float(model[index]), float(model[min(index + 4, 39)])
        prefix.require(
            old["step"] == new["step"] == 8 + index
            and abs(old_score["opening_reference"] - reference) <= 1e-10
            and abs(new_score["opening_reference"] - shifted) <= 1e-10,
            "Both recorded references must match the independently indexed original H40",
        )
        old_actual, actual = float(old_score["opening"]), float(new_score["opening"])
        prefix.require(
            np.isfinite([old_actual, actual]).all()
            and 0 <= old_actual <= 1
            and 0 <= actual <= 1
            and abs(old_score["opening_error"] - (old_actual - reference)) <= 1e-10
            and abs(new_score["opening_error"] - (actual - shifted)) <= 1e-10,
            "Measured openings and both stored own-reference errors must be valid",
        )
        points.append(
            dict(
                step=8 + index,
                model_index=index,
                lead_source_index=min(index + 4, 39),
                original_reference=reference,
                lead_reference=shifted,
                baseline_actual=old_actual,
                candidate_actual=actual,
                baseline_original_error=old_actual - reference,
                candidate_original_error=actual - reference,
                candidate_lead_error=actual - shifted,
                baseline_grasp=old["target_grasp"],
                candidate_grasp=new["target_grasp"],
                baseline_success=old["task_success"],
                candidate_success=new["task_success"],
            )
        )

    def stats(error):
        array = np.asarray([row[error] for row in points])
        return dict(rmse=float(np.sqrt(np.mean(array**2))), mae=float(np.mean(np.abs(array))))

    summary = dict(
        baseline_original=stats("baseline_original_error"),
        candidate_original=stats("candidate_original_error"),
        candidate_lead=stats("candidate_lead_error"),
        baseline_minimum_actual=min(row["baseline_actual"] for row in points),
        candidate_minimum_actual=min(row["candidate_actual"] for row in points),
        baseline_final_actual=points[-1]["baseline_actual"],
        candidate_final_actual=points[-1]["candidate_actual"],
        baseline_grasp_periods=sum(row["baseline_grasp"] for row in points),
        candidate_grasp_periods=sum(row["candidate_grasp"] for row in points),
        baseline_success_periods=sum(row["baseline_success"] for row in points),
        candidate_success_periods=sum(row["candidate_success"] for row in points),
    )
    return points, summary


def main():
    prefix.environment()
    producer, checker, baseline = prefix.read(PRODUCER), prefix.read(CHECKER), prefix.read(BASELINE)
    for result in (producer, checker):
        workers = prefix.read(result["workers_ref"])["workers"]
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["model_initializations"] == result["model_requests"] == result["fresh_cases"] == 0
            and result["automatic_retry"] is False
            and len(workers) == 1
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True
            and workers[0]["returncode"] == 0
            and workers[0]["signal_sent"] is False
            and result["counts"]["control_steps"] == 48
            and result["counts"]["native_steps"] == 1200
            and result["counts"]["models"] == result["counts"]["contexts"] == 3
            and result["counts"]["stop"] is None,
            "Both CPU phases, exact costs and reaped workers are required",
        )
    registration, checked_registration = (prefix.read(result["registration_ref"]) for result in (producer, checker))
    prefix.require(
        registration["inputs"] == checked_registration["inputs"]
        and registration["source_refs"] == checked_registration["source_refs"]
        and checked_registration["producer_branch_ref"] == producer["branch_ref"]
        and registration["inputs"]["refs"]["fixed_h40_audit"] == BASELINE,
        "Producer, checker and baseline must use the same registered inputs",
    )
    envelope = prefix.read(registration["inputs"]["refs"]["envelope"])
    action = np.asarray(envelope["action"], dtype=float)
    prefix.require(action.shape == (40, 128) and np.isfinite(action).all(), "The saved physical H40 is required")
    model_openings = np.clip(action[:, 9], 0, 1)
    branch, replay, old = (
        prefix.read(ref) for ref in (producer["branch_ref"], checker["branch_ref"], baseline["producer_branch_ref"])
    )
    first, candidate = original.flatten_fixed(branch)
    old_first, old_periods = original.flatten_fixed(old)
    continuation = replay["runtime"]["continuation"]
    replay_first = continuation["first_window"]["periods"]
    replay_periods = [row for window in continuation["replays"] for row in window["periods"]]
    prefix.require(replay["candidate_modules_unloaded"] is True, "The checker must not select candidate commands")
    common_prefix = comparison.compare_periods(first, old_first)
    first_diff = comparison.compare_periods(first, replay_first)
    replay_diff = comparison.compare_periods(candidate, replay_periods)
    capture_check.compare_capture(
        branch["runtime"]["fixed_reference_capture"], replay["runtime"]["checker_fixed_capture"]
    )
    capture_check.compare_capture(
        branch["runtime"]["fixed_reference_capture"], old["runtime"]["fixed_reference_capture"]
    )
    segments = branch["runtime"]["continuation"]["fixed"]["segments"]
    prefix.require(
        [s["opening_lead_steps"] for s in segments] == [4] * 5
        and [s["opening_terminal_hold_endpoints"] for s in segments] == [4, 12, 20, 28, 36],
        "All future hold slots must be explicitly accounted for",
    )
    points, summary = compare_openings(old_periods, candidate, model_openings)
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        baseline_ref=BASELINE,
        producer_branch_ref=producer["branch_ref"],
        checker_branch_ref=checker["branch_ref"],
        points=points,
        summary=summary,
        producer_counts=producer["counts"],
        baseline_counts=baseline["producer_counts"],
        common_first_window_max_absolute_difference=common_prefix,
        replay_first_window_max_absolute_difference=first_diff,
        replay_candidate_max_absolute_difference=replay_diff,
        new_producer_periods=48,
        new_checker_periods=48,
        cumulative_periods=baseline["cumulative_periods"] + 96,
        cumulative_inner_steps=baseline["cumulative_inner_steps"] + 2400,
        cumulative_sim_models=baseline["cumulative_sim_models"] + 6,
        cumulative_contexts=baseline["cumulative_contexts"] + 6,
        cumulative_model_initializations=baseline["cumulative_model_initializations"],
        cumulative_requests_attempted=baseline["cumulative_requests_attempted"],
        cumulative_requests_completed=baseline["cumulative_requests_completed"],
        cumulative_denoises=baseline["cumulative_denoises"],
        audit_models=0,
        audit_physics_steps=0,
        audit_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()

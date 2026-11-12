"""Compare fixed-H40 producer, independent replay and prior reobservation evidence."""

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as physical_audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_20261112/producer/result.json"),
    "sha256": "8df1eb68ee69310fb2c31d96c8884121cd79c2c1e6d89bc3d18e2181698570a7",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_20261112/checker/result.json"),
    "sha256": "f18a252703e0107f9b520483ee3a1e6da01bb5b8953e07b40c8e289351db3508",
}
PRIOR = {
    "path": str(ROOT / "outputs/reset_multitask_live_recovery_audit_20261111/result.json"),
    "sha256": "218582f9851373f02a3dbd22d392cb48783507dc241d4835fc6783fc8bf4e2a3",
}


def flatten_fixed(branch):
    continuation = branch["runtime"]["continuation"]
    first = continuation["first_window"]["periods"]
    fixed = [period for segment in continuation["fixed"]["segments"] for period in segment["continuation"]["periods"]]
    return first, fixed


def main():
    prefix.environment()
    producer, checker, prior = prefix.read(PRODUCER), prefix.read(CHECKER), prefix.read(PRIOR)
    for result in (producer, checker):
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["model_initializations"] == result["model_requests"] == 0
            and result["fresh_cases"] == 0
            and result["automatic_retry"] is False,
            "Both CPU phases must close with unchanged sources and no new model calls",
        )
    branch, replay = prefix.read(producer["branch_ref"]), prefix.read(checker["branch_ref"])
    old = prefix.read(prior["cases"][0]["producer_ref"])
    first, fixed = flatten_fixed(branch)
    replay_first = replay["runtime"]["continuation"]["first_window"]["periods"]
    replay_fixed = [period for row in replay["runtime"]["continuation"]["replays"] for period in row["periods"]]
    old_run = old["runtime"]["continuation"]
    old_first = old_run["first_window"]["periods"]
    old_live = [period for segment in old_run["stream"]["segments"] for period in segment["continuation"]["periods"]]
    prefix.require(
        branch["case"]["case_id"] == replay["case"]["case_id"] == old["case"]["case_id"] == "task0_init0_s42_raw"
        and replay["candidate_modules_unloaded"] is True
        and len(first) == len(replay_first) == len(old_first) == 8
        and len(fixed) == len(replay_fixed) == len(old_live) == 40
        and producer["counts"]["last_generated_endpoint_index"] == 39
        and producer["counts"]["stop"] is None,
        "The three same-case runs and completed fixed H40 are required",
    )
    replay_first_diff = physical_audit.compare_periods(first, replay_first)
    replay_fixed_diff = physical_audit.compare_periods(fixed, replay_fixed)
    common_first_diff = physical_audit.compare_periods(first, old_first)
    common_live_diff = physical_audit.compare_periods(fixed[:8], old_live[:8])
    capture_check.compare_capture(
        branch["runtime"]["fixed_reference_capture"], replay["runtime"]["checker_fixed_capture"]
    )
    capture_check.compare_capture(branch["runtime"]["fixed_reference_capture"], old["runtime"]["streaming_captures"][0])

    def openings(periods):
        scores = [period["completed_feedback"]["score"] for period in periods]
        return dict(
            minimum_reference=float(min(row["opening_reference"] for row in scores)),
            minimum_measured=float(min(row["opening"] for row in scores)),
            final_reference=float(scores[-1]["opening_reference"]),
            final_measured=float(scores[-1]["opening"]),
            target_grasp_periods=sum(row["target_grasp"] for row in periods),
            task_success_periods=sum(row["task_success"] for row in periods),
        )

    fixed_openings, old_openings = openings(fixed), openings(old_live)
    prefix.require(
        np.isclose(fixed_openings["final_reference"], 0.7062058448791504, atol=1e-10, rtol=0)
        and fixed_openings["target_grasp_periods"] == old_openings["target_grasp_periods"] == 0
        and fixed_openings["task_success_periods"] == old_openings["task_success_periods"] == 0,
        "The actual terminal reference and observed task labels must agree with the saved evidence",
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        prior_audit_ref=PRIOR,
        producer_branch_ref=producer["branch_ref"],
        checker_branch_ref=checker["branch_ref"],
        old_reobservation_branch_ref=prior["cases"][0]["producer_ref"],
        common_first_window_max_absolute_difference=common_first_diff,
        common_first_reference_max_absolute_difference=common_live_diff,
        independent_first_window_max_absolute_difference=replay_first_diff,
        independent_fixed_reference_max_absolute_difference=replay_fixed_diff,
        producer_counts=producer["counts"],
        checker_counts=checker["counts"],
        fixed_reference_openings=fixed_openings,
        old_reobservation_openings=old_openings,
        prior_cumulative_periods=prior["cumulative_periods"],
        new_producer_periods=48,
        new_checker_periods=48,
        cumulative_periods=prior["cumulative_periods"] + 96,
        cumulative_inner_steps=prior["cumulative_inner_steps"] + 2400,
        cumulative_sim_models=prior["cumulative_sim_models"] + 6,
        cumulative_contexts=prior["cumulative_contexts"] + 6,
        cumulative_model_initializations=prior["cumulative_model_initializations"],
        cumulative_requests_attempted=prior["cumulative_requests_attempted"],
        cumulative_requests_completed=prior["cumulative_requests_completed"],
        cumulative_denoises=prior["cumulative_denoises"],
        audit_model_requests=0,
        audit_physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()

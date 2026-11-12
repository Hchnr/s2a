"""Compare amplified-closing commands with independent replay and original H40."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as original
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check

ROOT = prefix.REPO
PRODUCER = {
    "path": str(ROOT / "outputs/bowl_h40_closing_intent_20261112/producer/result.json"),
    "sha256": "017f9fe29a4c4de07f1c65f74e864438535359115bf44e81c221fffde4e93979",
}
CHECKER = {
    "path": str(ROOT / "outputs/bowl_h40_closing_intent_20261112/checker/result.json"),
    "sha256": "077cb4418c17b7c60a6319b9b4cd0d4546fab4491a5754b5410f5a0a679931f1",
}
BASELINE = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
PRIOR = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_audit_20261112/result.json"),
    "sha256": "005cb8a5e91c8625f2bb7befee71cd9978d0d1fb1276b75629769ec4d67b8577",
}
OUTPUT = ROOT / "outputs/bowl_h40_closing_intent_audit_20261112/result.json"


def compare_openings(baseline, candidate, model_openings):
    """Keep original and transformed references separate at each completed period."""
    model = np.asarray(model_openings, dtype=float)
    prefix.require(
        model.shape == (40,)
        and np.isfinite(model).all()
        and np.all((model >= 0) & (model <= 1))
        and len(baseline) == 40
        and 0 <= len(candidate) <= 40,
        "A full generated H40 and bounded actual continuation are required",
    )
    points = []
    for index, new in enumerate(candidate):
        old = baseline[index]
        old_score, new_score = old["completed_feedback"]["score"], new["completed_feedback"]["score"]
        reference = float(model[index])
        transformed = float(np.clip(3 * model[min(index + 8, 39)] - 2, 0, 1))
        prefix.require(
            old["step"] == new["step"] == 8 + index
            and abs(old_score["opening_reference"] - reference) <= 1e-10
            and abs(new_score["opening_reference"] - transformed) <= 1e-10,
            "Original and modified references must match the independent index calculation",
        )
        actual, old_actual = float(new_score["opening"]), float(old_score["opening"])
        prefix.require(
            np.isfinite([actual, old_actual]).all()
            and 0 <= actual <= 1
            and 0 <= old_actual <= 1
            and abs(new_score["opening_error"] - (actual - transformed)) <= 1e-10,
            "Measured opening and recorded modified-reference error must match",
        )
        points.append(
            dict(
                step=8 + index,
                model_index=index,
                transformed_source_index=min(index + 8, 39),
                original_reference=reference,
                modified_reference=transformed,
                baseline_actual=old_actual,
                candidate_actual=actual,
                baseline_original_error=old_actual - reference,
                candidate_original_error=actual - reference,
                candidate_modified_error=actual - transformed,
            )
        )

    def statistics(key):
        if not points:
            return None
        values = np.asarray([row[key] for row in points])
        return dict(rmse=float(np.sqrt(np.mean(values**2))), mae=float(np.mean(np.abs(values))))

    return points, dict(
        baseline_original=statistics("baseline_original_error"),
        candidate_original=statistics("candidate_original_error"),
        candidate_modified=statistics("candidate_modified_error"),
    )


def endpoint_checks(periods):
    """Record pose and finger-joint endpoint limits without claiming inner-finger coverage."""
    failures, joints = [], []
    for row in periods:
        error = np.asarray(row["tcp_error"], dtype=float)
        q = np.asarray(row["completed_feedback"]["score"]["qpos"], dtype=float)
        prefix.require(
            error.shape == (6,) and q.shape == (2,) and np.isfinite([*error, *q]).all(),
            "Finite TCP error and two finger positions are required",
        )
        if not row["pose_caps_satisfied"]:
            failures.append(
                dict(
                    endpoint_step=row["step"] + 1,
                    position_error_m=float(np.linalg.norm(error[:3])),
                    rotation_error_rad=float(np.linalg.norm(error[3:])),
                )
            )
        if not (0 <= q[0] <= 0.04 and -0.04 <= q[1] <= 0):
            joints.append(dict(endpoint_step=row["step"] + 1, qpos=q))
    return dict(
        pose_cap_failure_endpoints=failures,
        finger_nominal_range_outside_endpoints=joints,
        finger_inner_samples_recorded=False,
    )


def main():
    prefix.environment()
    producer, checker, baseline, prior = (prefix.read(ref) for ref in (PRODUCER, CHECKER, BASELINE, PRIOR))
    for result in (producer, checker):
        workers = prefix.read(result["workers_ref"])["workers"]
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["model_initializations"] == result["model_requests"] == result["fresh_cases"] == 0
            and result["automatic_retry"] is False
            and len(workers) == 1
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True
            and workers[0]["returncode"] == 0
            and workers[0]["signal_sent"] is False,
            "Both CPU phases and their reaped workers are required",
        )
    registration, checked_registration = (prefix.read(item["registration_ref"]) for item in (producer, checker))
    prefix.require(
        registration["inputs"] == checked_registration["inputs"]
        and registration["source_refs"] == checked_registration["source_refs"]
        and checked_registration["producer_branch_ref"] == producer["branch_ref"]
        and registration["inputs"]["refs"]["fixed_h40_audit"] == BASELINE
        and registration["inputs"]["refs"]["opening_lead_audit"] == PRIOR,
        "Producer, checker and both baseline audits must share the registered inputs",
    )
    envelope = prefix.read(registration["inputs"]["refs"]["envelope"])
    action = np.asarray(envelope["action"], dtype=float)
    prefix.require(action.shape == (40, 128) and np.isfinite(action).all(), "The recorded physical H40 is required")
    model_openings = np.clip(action[:, 9], 0, 1)
    branch, replay, old = (
        prefix.read(ref) for ref in (producer["branch_ref"], checker["branch_ref"], baseline["producer_branch_ref"])
    )
    first, candidate = original.flatten_fixed(branch)
    old_first, old_periods = original.flatten_fixed(old)
    continuation = replay["runtime"]["continuation"]
    replay_first = continuation["first_window"]["periods"]
    replay_periods = [row for window in continuation["replays"] for row in window["periods"]]
    prefix.require(
        len(first) == len(old_first) == len(replay_first) == 8
        and len(candidate) == len(replay_periods)
        and len(candidate) <= 40
        and replay["candidate_modules_unloaded"] is True
        and producer["counts"]["control_steps"] == checker["counts"]["control_steps"] == 8 + len(candidate)
        and producer["counts"]["native_steps"] == checker["counts"]["native_steps"] == 25 * (8 + len(candidate))
        and continuation["reported_producer_stop"] == producer["counts"]["stop"]
        and continuation["stop_reason_independently_verified"] is False,
        "The checker must replay precisely the completed actions and preserve the reported stop",
    )
    common_first = comparison.compare_periods(first, old_first)
    replay_first_diff = comparison.compare_periods(first, replay_first)
    replay_diff = comparison.compare_periods(candidate, replay_periods)
    if candidate:
        capture_check.compare_capture(
            branch["runtime"]["fixed_reference_capture"], replay["runtime"]["checker_fixed_capture"]
        )
        capture_check.compare_capture(
            branch["runtime"]["fixed_reference_capture"], old["runtime"]["fixed_reference_capture"]
        )
    segments = branch["runtime"]["continuation"]["fixed"]["segments"]
    prefix.require(
        all(
            s["opening_lead_steps"] == 8
            and s["opening_gain"] == 3.0
            and s["opening_terminal_hold_endpoints"] == min(40, 8 + s["reference_offset"])
            for s in segments
        ),
        "The transformed reference and held slots must remain explicit",
    )
    points, opening_summary = compare_openings(old_periods, candidate, model_openings)
    all_periods = first + candidate
    counts = producer["counts"]
    checked = endpoint_checks(all_periods)
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        baseline_ref=BASELINE,
        prior_ref=PRIOR,
        producer_branch_ref=producer["branch_ref"],
        checker_branch_ref=checker["branch_ref"],
        points=points,
        opening_summary=opening_summary,
        endpoint_checks=checked,
        producer_counts=counts,
        checker_counts=checker["counts"],
        target_grasp_endpoints=[row["step"] + 1 for row in all_periods if row["target_grasp"]],
        task_success_endpoints=[row["step"] + 1 for row in all_periods if row["task_success"]],
        common_first_window_max_absolute_difference=common_first,
        replay_first_window_max_absolute_difference=replay_first_diff,
        replay_candidate_max_absolute_difference=replay_diff,
        new_producer_periods=counts["control_steps"],
        new_checker_periods=checker["counts"]["control_steps"],
        cumulative_periods=prior["cumulative_periods"] + 2 * counts["control_steps"],
        cumulative_inner_steps=prior["cumulative_inner_steps"] + 2 * counts["native_steps"],
        cumulative_sim_models=prior["cumulative_sim_models"] + counts["models"] + checker["counts"]["models"],
        cumulative_contexts=prior["cumulative_contexts"] + counts["contexts"] + checker["counts"]["contexts"],
        cumulative_model_initializations=prior["cumulative_model_initializations"],
        cumulative_requests_attempted=prior["cumulative_requests_attempted"],
        cumulative_requests_completed=prior["cumulative_requests_completed"],
        cumulative_denoises=prior["cumulative_denoises"],
        audit_models=0,
        audit_physics_steps=0,
        audit_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()

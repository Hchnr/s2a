"""Reconstruct completed-period refresh signals from six recorded trajectories."""

import collections
import math
from pathlib import Path

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json"),
    "sha256": "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
}
PLAN = ROOT / "plans/v0.0.1/tracking/references/refresh_margin_preference_20261115.md"
OUTPUT = ROOT / "outputs/refresh_margin_events_20261115/result.json"
POSITION_LIMIT_M, ROTATION_LIMIT_RAD, TOLERANCE = 0.0095, 0.047, 1e-10


def check_signal(period, signal):
    """Reconstruct the registered trigger without importing the candidate policy."""
    step = signal["current_step"]
    prefix.require(
        type(step) is int
        and period["step"] == step - 1
        and period["endpoint_receipt"]["endpoint_step"] == step
        and len(period["actual_inner"]) == 25,
        "A signal must follow its own fully completed physical period",
    )
    history = period["completed_feedback"]["policy_history"]["past_inner_error"]
    verified = period["decision"]["quality"]["verification"]
    prefix.require(
        history["last_completed_step"] == step - 1
        and history["future_guarantee"] is False
        and verified["passed"] is verified["first_pose_caps_satisfied"] is True,
        "The recorded past and audited first pose must be causal",
    )
    if verified["hard_pose_prefix"] != 2:
        prefix.require(signal["replan"] is True and signal["reason"] == "no_audited_second_pose", "Missing second pose")
        return dict(reason="no_audited_second_pose", near_position=False, near_rotation=False)
    prefix.require(
        verified["prefix_pose_caps_satisfied"] is verified["prefix_pose_caps_enforced"] is True,
        "The two-pose witness must be enforced and passed",
    )
    errors = verified["tcp_errors"]
    prefix.require(len(errors) == 8 and all(len(row) == 6 for row in errors), "Eight six-dimensional errors required")
    second = errors[1]
    prefix.require(all(math.isfinite(value) for row in errors for value in row), "Finite recorded errors required")
    position = math.sqrt(math.fsum(float(x) ** 2 for x in second[:3]))
    rotation = math.sqrt(math.fsum(float(x) ** 2 for x in second[3:]))
    prefix.require(
        abs(signal["predicted_next_position_error_m"] - position) <= 1e-12
        and abs(signal["predicted_next_rotation_error_rad"] - rotation) <= 1e-12
        and signal["position_margin_limit_m"] == POSITION_LIMIT_M
        and signal["rotation_margin_limit_rad"] == ROTATION_LIMIT_RAD,
        "Recorded trigger magnitudes and fixed limits must match the witness",
    )
    replan = position >= POSITION_LIMIT_M - TOLERANCE or rotation >= ROTATION_LIMIT_RAD - TOLERANCE
    reason = "quality_margin_consumed" if replan else "quality_margin_available"
    prefix.require(signal["replan"] is replan and signal["reason"] == reason, "Recorded refresh decision differs")
    return dict(
        reason=reason,
        near_position=abs(position - POSITION_LIMIT_M) <= TOLERANCE,
        near_rotation=abs(rotation - ROTATION_LIMIT_RAD) <= TOLERANCE,
    )


def main():
    prefix.environment()
    review = prefix.read(SOURCE)
    prefix.require(review["complete"] is True, "The completed six-case physical review is required")
    cases, refs = [], [SOURCE, prefix.reference(PLAN), prefix.reference(Path(__file__).resolve())]
    for case in review["cases"]:
        producer = prefix.read(case["producer_ref"])
        refs.append(case["producer_ref"])
        segments = producer["runtime"]["continuation"]["adaptive"]["segments"]
        reasons, near = collections.Counter(), collections.Counter()
        for segment in segments:
            periods = segment["continuation"]["periods"]
            for signal in segment["signals"]:
                offset = signal["current_step"] - segment["start_step"] - 1
                prefix.require(0 <= offset < len(periods), "The signal must bind a period in its own segment")
                row = check_signal(periods[offset], signal)
                reasons[row["reason"]] += 1
                if row["reason"] == "quality_margin_consumed":
                    near["position"] += row["near_position"]
                    near["rotation"] += row["near_rotation"]
                    near["either"] += row["near_position"] or row["near_rotation"]
                    near["both"] += row["near_position"] and row["near_rotation"]
        cases.append(dict(case_index=case["case_index"], signal_counts=dict(reasons), near_boundary=dict(near)))
    prefix.require([row["case_index"] for row in cases] == [0, 2, 4, 6, 8, 10], "All six original cases required")
    totals = collections.Counter()
    near = collections.Counter()
    for row in cases:
        totals.update(row["signal_counts"])
        near.update(row["near_boundary"])
    for ref in refs:
        prefix.checked(ref)
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                retrospective=True,
                cases=cases,
                signal_counts=dict(totals),
                consumed_near_boundary=dict(near),
                tolerance=TOLERANCE,
                source_refs=refs,
                new_model_requests=0,
                new_physics_steps=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

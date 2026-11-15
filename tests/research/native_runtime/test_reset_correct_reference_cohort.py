"""Cohort scoring preserves failed cases and enforces the registered gates."""

import pytest

from scripts.simulation.runtime.run_reset_correct_reference_cohort import checker_eligible, qualifies


def valid_summary():
    return dict(
        full_trajectory=True,
        completed_periods=182,
        expected_periods=182,
        stable_last_40=True,
        position_rmse_m=0.001,
        rotation_rmse_rad=0.01,
        minimum_inner_margin_rad=0.1,
        inner_below_buffer_samples=0,
        inner_outside_samples=0,
    )


CONFIG = dict(position_cap_m=0.01, rotation_cap_rad=0.05, joint_buffer_rad=0.05)


@pytest.mark.parametrize(
    "failure", ["missing_replay", "short", "unstable", "position", "rotation", "buffer", "outside"]
)
def test_cohort_qualification_requires_every_gate(failure):
    summary = valid_summary()
    assert qualifies(summary, True, CONFIG)
    replayed = failure != "missing_replay"
    if failure == "short":
        summary.update(completed_periods=181, full_trajectory=False)
    elif failure == "unstable":
        summary["stable_last_40"] = False
    elif failure == "position":
        summary["position_rmse_m"] = 0.011
    elif failure == "rotation":
        summary["rotation_rmse_rad"] = 0.051
    elif failure == "buffer":
        summary["inner_below_buffer_samples"] = 1
    elif failure == "outside":
        summary["inner_outside_samples"] = 1
    assert not qualifies(summary, replayed, CONFIG)


def test_checker_replays_closed_peers_and_preserves_failed_cases():
    cases = [dict(spec=dict(index=index)) for index in range(4)]
    producer = dict(cases={"0": dict(status="closed"), "1": dict(status="failed"), "3": dict(status="closed")})
    assert checker_eligible(producer, cases, reused=3) == [0]

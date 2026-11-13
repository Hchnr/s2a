"""Keep causal histories separated across the evaluated trajectories."""

import numpy as np

from scripts.tracking.response_models import multitask_causal_tcp_innovation as audit


def test_each_trajectory_starts_without_another_trajectorys_innovation():
    def period(step, actual):
        return {
            "step": step,
            "decision": {"quality": {"first_tcp_error": [0.0] * 6, "verification": {"passed": True}}},
            "tcp_error": [actual, 0, 0, 0, 0, 0],
            "pose_caps_satisfied": True,
        }

    groups = [
        dict(protocol="adaptive", case_index=0, case_id="first", periods=[period(0, 0.004), period(1, 0.004)]),
        dict(protocol="adaptive", case_index=2, case_id="second", periods=[period(0, 0.0), period(1, 0.0)]),
    ]
    cases, summaries = audit.evaluate_groups(groups)
    assert [row["periods_with_history"] for row in cases] == [1, 1]
    assert summaries["adaptive"]["raw"]["periods"] == 2
    assert all(case["rows"][0]["previous_innovation"] is None for case in cases)
    np.testing.assert_array_equal(cases[1]["rows"][1]["previous_innovation"]["predicted_error"], np.zeros(6))


def test_recorded_roster_has_all_independently_replayed_periods():
    _config, groups, _consumed = audit.load_trajectories()
    assert [(group["protocol"], group["case_index"]) for group in groups] == [
        (protocol, index) for protocol in ("adaptive", "fixed") for index in (0, 2, 4, 6, 8, 10)
    ]
    assert sum(len(group["periods"]) for group in groups if group["protocol"] == "adaptive") == 1025
    assert sum(len(group["periods"]) for group in groups if group["protocol"] == "fixed") == 270

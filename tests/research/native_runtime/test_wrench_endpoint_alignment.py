"""Ensure wrench inputs precede the action and bind to its measured outcome."""

import pytest

from scripts.tracking.response_models.analyze_wrench_endpoint_replay import aligned_periods


def test_alignment_uses_start_wrench_and_checks_end_state():
    start = {
        "endpoint_step": 80,
        "sim_time": 1.0,
        "sim_state_sha256": "start",
        "wrench": {"local": [3, 4, 0, 0, 0, 0]},
        "offline_gripper_contacts": [],
    }
    end = {
        "endpoint_step": 81,
        "sim_time": 1.05,
        "sim_state_sha256": "end",
        "wrench": {"local": [99, 0, 0, 0, 0, 0]},
        "offline_gripper_contacts": [{"geoms": ["a", "b"]}],
    }
    period = {
        "step": 80,
        "endpoint_receipt": {"endpoint_step": 81, "sim_time": 1.05, "sim_state_sha256": "end"},
        "decision": {"quality": {"first_tcp_error": [0, 0, 0, 0, 0, 0]}},
        "tcp_error": [0.003, 0.004, 0, 0, 0, 0],
    }
    branch = {"runtime": {"continuation": {"segments": [{"continuation": {"periods": [period]}}]}}}
    trace = {"rows": [start, end]}
    aligned = aligned_periods(trace, branch)[0]
    assert aligned["force_norm_n"] == 5.0
    assert aligned["position_innovation_m"] == 0.005
    assert aligned["offline_gripper_contact"] is False
    end["sim_state_sha256"] = "different"
    with pytest.raises(ValueError, match="following endpoint"):
        aligned_periods(trace, branch)

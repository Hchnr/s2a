"""A second least-squares formulation checks each saved forecast and endpoint."""

import copy

import pytest

from scripts.tracking.response_models import audit_action_conditioned_tcp_cohort as audit
from scripts.tracking.response_models import evaluate_action_conditioned_tcp as scoring


def test_independent_audit_accepts_causal_scores_and_rejects_changed_measurements():
    periods = [
        dict(
            step=i,
            command=[i * 0.1] * 6 + [0.0],
            decision={"quality": {"first_tcp_error": [0.0] * 6, "verification": {"passed": True}}},
            tcp_error=[i * 0.001, 0, 0, 0, 0, 0],
            pose_caps_satisfied=True,
        )
        for i in range(4)
    ]
    spec = dict(key="example", expected_periods=4)
    rows = scoring.evaluate(periods)
    result = dict(complete=True, passed=True, identity=spec, rows=rows, summary=scoring.summarize(rows))
    assert audit.verify_case(spec, periods, result)["scored_periods"] == 3
    corrupted = copy.deepcopy(result)
    corrupted["rows"][2]["actual_error"][0] = 0.004
    with pytest.raises(ValueError, match="measured endpoint"):
        audit.verify_case(spec, periods, corrupted)
    corrupted = copy.deepcopy(result)
    corrupted["rows"][2]["action_conditioned"]["predicted_error"][0] += 0.001
    with pytest.raises(ValueError, match="Independent forecast"):
        audit.verify_case(spec, periods, corrupted)

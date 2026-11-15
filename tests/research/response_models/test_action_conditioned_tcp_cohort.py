"""The retrospective cohort retains its audited roster and actual prefixes."""

import pytest

from scripts.tracking.response_models import run_action_conditioned_tcp_cohort as cohort


def test_registered_roster_retains_early_stop_and_independent_wine_case():
    _config, roster, consumed = cohort.load_manifest()
    assert len(roster) == 31
    assert len(consumed) == 7
    assert {
        protocol: sum(item["expected_periods"] for item in roster if item["protocol"] == protocol)
        for protocol in cohort.PROTOCOLS
    } == cohort.EXPECTED_PERIODS
    short = next(item for item in roster if item["key"] == "correct_07")
    assert short["expected_periods"] == 86 and short["registered_periods"] == 215
    periods, identities = cohort.load_periods(short)
    assert len(periods) == 86 and periods[-1]["step"] == 85 and len(identities) == 2
    assert roster[-1]["key"] == "wine_04" and roster[-1]["expected_periods"] == 144
    assert not cohort.zero_difference({}) and not cohort.zero_difference({"tcp_error": 0.001})


def test_aggregate_keeps_each_trajectorys_first_period_out_and_rejects_missing_cases():
    _config, roster, _consumed = cohort.load_manifest()
    prediction = dict(
        predicted_pose_caps_satisfied=True, position_prediction_error_m=0.0, rotation_prediction_error_rad=0.0
    )
    first = dict(history_available=False, ridge_fit_performed=False, actual_pose_caps_satisfied=True)
    scored = dict(
        history_available=True,
        ridge_fit_performed=True,
        actual_pose_caps_satisfied=True,
        **{method: prediction for method in cohort.scoring.METHODS},
    )
    cases = {
        spec["key"]: dict(
            complete=True, passed=True, identity=spec, rows=[first] + [scored] * (spec["expected_periods"] - 1)
        )
        for spec in roster
    }
    result = cohort.aggregate(roster, cases)
    assert result["correct"]["recorded_periods"] == 4778
    assert result["correct"]["scored_periods"] == result["correct"]["ridge_fits"] == 4760
    assert result["wine"]["scored_periods"] == 143
    cases.pop("correct_07")
    with pytest.raises(ValueError, match="All registered cases"):
        cohort.aggregate(roster, cases)

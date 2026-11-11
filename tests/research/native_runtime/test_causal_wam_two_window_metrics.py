import pytest

from scripts.verification.reference_ports import audit_causal_wam_two_window_feedback as audit


def test_completed_two_window_metrics_preserve_replay_and_evidence_limits():
    result = audit.load_audit()
    assert result["completed_steps"] == 16
    assert result["replayed_prefix_inner_samples"] == 2000
    assert result["replayed_continuation_inner_samples"] == 400
    assert result["independent_deferral_max_difference"] == 0
    assert result["terminal_consecutive_success_periods"] == 9
    assert result["qualified_stable_success"] is False
    assert result["new_model_requests"] == result["new_physics_steps"] == 0
    first, second = result["windows"]
    assert [first["start_step"], second["start_step"]] == [80, 88]
    assert first["pose_caps_against_reference"] == second["pose_caps_against_reference"] == 8
    assert first["actual_vs_reference"]["position_rmse_m"] == pytest.approx(0.00240518, abs=1e-8)
    assert second["actual_vs_reference"]["position_rmse_m"] == pytest.approx(0.00270888, abs=1e-8)
    assert second["opening_vs_reference_rmse"] == pytest.approx(0.13314572, abs=1e-8)

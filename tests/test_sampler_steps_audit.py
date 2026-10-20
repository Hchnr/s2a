"""Independent integration and pre-registered SI promotion decision tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "sampler_steps_audit_test", root / "scripts/audit_wam_sampler_steps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def columns(rotation):
    return np.r_[rotation[:, 0], rotation[:, 1]]


def test_independent_local_position_uses_preceding_rotation_and_unpadded_gt(audit):
    identity = np.eye(3)
    quarter = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    states = np.zeros((41, 10))
    states[:, 3:9] = columns(quarter)
    states[0, 3:9] = columns(identity)
    states[1, :3], states[2, :3] = [1, 0, 0], [1, 1, 0]
    states[0, 9], states[1, 9], states[2, 9] = 0.8, 0.4, 0.2
    actions = np.zeros((40, 128))
    actions[:, 3:9] = columns(identity)
    actions[0, 3:9] = columns(quarter)
    actions[:2, 0] = 1
    actions[:2, 9] = [0.4, 0.2]
    count, errors, generated = audit.integrate_and_score(states, dict(anchor_frame=0, valid_future_steps=2), actions)
    assert count == 2 and len(generated["position"]) == 40
    np.testing.assert_allclose(generated["position"][1], [1, 1, 0], atol=1e-15)
    np.testing.assert_allclose(errors["wam"]["position"], 0, atol=1e-15)
    np.testing.assert_allclose(errors["wam"]["rotation"], 0, atol=1e-15)
    assert errors["wam"]["opening"] == [0, 0]
    np.testing.assert_allclose(errors["hold"]["position"], [1, np.sqrt(2)])
    summary = audit.pool([dict(errors=errors)])
    assert [summary[str(h)]["wam"]["valid_endpoints"] for h in (8, 16, 40)] == [2, 2, 2]


def test_constant_velocity_uses_only_prior_four_states(audit):
    states = np.zeros((9, 10))
    for i in range(9):
        states[i, 0] = 0.01 * i
        states[i, 3:9] = columns(Rotation.from_rotvec([0, 0, 0.1 * i]).as_matrix())
        states[i, 9] = 0.5
    actions = np.zeros((40, 128))
    actions[:, 3:9] = columns(np.eye(3))
    count, errors, _ = audit.integrate_and_score(states, dict(anchor_frame=4, valid_future_steps=4), actions)
    assert count == 4
    np.testing.assert_allclose(errors["constant_velocity"]["position"], 0, atol=1e-15)
    np.testing.assert_allclose(errors["constant_velocity"]["rotation"], 0, atol=1e-15)
    # Changing a future state affects scoring, never the extrapolated prediction.
    states[6, 0] += 1
    _, changed, _ = audit.integrate_and_score(states, dict(anchor_frame=4, valid_future_steps=4), actions)
    assert changed["constant_velocity"]["position"][1] > 0.99
    assert changed["constant_velocity"]["position"][0] == errors["constant_velocity"]["position"][0]


def metrics(position=100.0, rotation=0.5, opening=0.2):
    return {
        str(h): dict(wam=dict(position_rmse_mm=position, rotation_rmse_rad=rotation, opening_mae=opening))
        for h in (8, 16, 40)
    }


def test_promotion_requires_all_six_and_one_full10pct_improvement(audit):
    baseline = metrics()
    insufficient = metrics(position=91)
    orientation_regresses = metrics(position=80, rotation=0.500000001)
    result = audit.promotion({"35": baseline, "70": insufficient, "140": orientation_regresses})
    assert result["selected_steps"] is None
    assert result["candidates"]["70"]["at_least_one_position_improves10pct"] is False
    assert result["candidates"]["140"]["all_six_nonregression"] is False


def test_position_nonregression_tolerance_is_si_metres_not_display_mm(audit):
    baseline, candidate = metrics(), metrics()
    candidate["8"]["wam"]["position_rmse_mm"] = 80
    candidate["16"]["wam"]["position_rmse_mm"] += 5e-8  # 5e-11m: inside SI1e-10m.
    result = audit.promotion({"35": baseline, "70": candidate, "140": metrics(position=95)})
    assert result["selected_steps"] == 70
    candidate["16"]["wam"]["position_rmse_mm"] = 100 + 2e-7  # 2e-10m: outside.
    result = audit.promotion({"35": baseline, "70": candidate, "140": metrics(position=95)})
    assert result["selected_steps"] is None


def test_140_preference_needs_further10pct_and_all_six(audit):
    result = audit.promotion({"35": metrics(), "70": metrics(position=80), "140": metrics(position=75)})
    assert result["selected_steps"] == 70
    result = audit.promotion({"35": metrics(), "70": metrics(position=80), "140": metrics(position=70)})
    assert result["selected_steps"] == 140
    assert result["new_physics_authorized"] is False


def test_zero_reference_error_is_not_a10pct_improvement(audit):
    result = audit.promotion({str(n): metrics(position=0, rotation=0, opening=0) for n in (35, 70, 140)})
    assert result["selected_steps"] is None


def test_partial_audit_never_creates_final_report(audit, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "sampler_steps_report_test", root / "scripts/report_wam_sampler_steps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pending = tmp_path / "partial.json"
    pending.write_text('{"complete":false,"status":"partial","errors":[]}')
    output = root / "outputs/never_create_sampler_partial_report"
    target = root / "reports/never_create_sampler_partial_report.md"
    monkeypatch.setattr(
        "sys.argv", ["report", "--audit", str(pending), "--output-dir", str(output), "--report", str(target)]
    )
    with pytest.raises(ValueError, match="complete144"):
        module.main()
    assert not output.exists() and not target.exists()

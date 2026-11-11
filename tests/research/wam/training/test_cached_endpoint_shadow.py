"""Full-timeline shadow must keep source, tail, and event semantics explicit."""

import numpy as np
import pytest

from scripts.wam.tracking.references import shadow_cached_endpoint_calibration as shadow


def config():
    return dict(
        pi_margin_rad=0.0001,
        position_scale_m=0.1,
        rotation_scale_rad=0.5,
        position_cap_m=0.05,
        rotation_cap_rad=0.25,
        opening_cap=0.25,
    )


def candidate():
    return {"pose_coefficients": np.zeros((7, 6)), "opening_coefficients": np.array([-0.1, 0, 0, 0, 0, 0, 0])}


def test_short_oracle_tail_counts_only_available_endpoints_and_closure():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    chunk = dict(anchor_tcp=np.eye(4), effective_targets_tcp=poses, openings=[0.4] * 3, step=0)
    rows = [dict(step=i, opening_before=0.9, opening=0.8, after_source=np.eye(4)) for i in range(3)]
    result = shadow.shadow_case({"family": "correct_reference"}, [chunk], rows, np.eye(4), candidate(), config(), 1e-10)
    summary = shadow.summarize(**result)
    assert summary["steps"] == 3 and summary["queries"] == 1
    assert summary["consumed_closure_queries"] == dict(raw=0, corrected=1, added=1, removed=0)
    assert result["queries"][0]["available"] == 3 and result["queries"][0]["consumed"] == 3
    assert summary["raw"]["opening_mae"] == pytest.approx(0.4)
    assert summary["corrected"]["opening_mae"] == pytest.approx(0.5)


def test_development_reference_must_be_unmodified_raw():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    chunk = dict(
        anchor_tcp=np.eye(4),
        effective_targets_tcp=poses,
        raw_targets_tcp=poses.copy(),
        openings=[0.4] * 40,
        model_rotation_alpha=1,
        retiming=False,
    )
    shadow.bounded_reference({"family": "development"}, chunk, 0.9, candidate(), config(), 1e-10)
    chunk["raw_targets_tcp"][0, 0, 3] = 0.01
    with pytest.raises(ValueError, match="raw WAM reference"):
        shadow.bounded_reference({"family": "development"}, chunk, 0.9, candidate(), config(), 1e-10)


def test_independent_reconstruction_mismatch_stops_the_shadow(monkeypatch):
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    chunk = dict(anchor_tcp=np.eye(4), effective_targets_tcp=poses, openings=[0.4] * 40)
    original = shadow.independent_corrected

    def wrong(*args):
        position, rotation, opening = original(*args)
        position[0, 0] += 0.001
        return position, rotation, opening

    monkeypatch.setattr(shadow, "independent_corrected", wrong)
    with pytest.raises(ValueError, match="independent reference reconstruction"):
        shadow.bounded_reference({"family": "correct_reference"}, chunk, 0.9, candidate(), config(), 1e-10)

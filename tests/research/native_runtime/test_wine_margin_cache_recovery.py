"""The wine cache recovery preserves its incident and exact endpoint identity."""

import copy
import json

import numpy as np
import pytest

from scripts.simulation.runtime import recover_wine_margin_cache as recovery


def test_recovery_configuration_is_single_case_and_bounded():
    config = json.loads(recovery.CONFIG.read_text())
    recovery.validate_config(config)
    for key, value in (("max_total_control_steps_per_phase", 145), ("automatic_retry", True)):
        with pytest.raises(ValueError):
            recovery.validate_config(dict(config, **{key: value}))
    changed = copy.deepcopy(config)
    changed["capture_diagnostic"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        recovery.validate_config(changed)


def test_boundary_requires_original_capture_and_exact_cache_reuse():
    binding = dict(sim_time=5.0, sim_state_sha256="state")
    capture = dict(
        frame_index=104,
        current_endpoint=dict(endpoint_step=104, **binding),
        render_before=dict(binding),
        render_after=dict(binding),
        future_rgb_frames=0,
        camera_keys=["left", "right"],
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
        canvas_rgb_sha256="rgb",
        canvas_png=dict(sha256="png"),
        current_source_pose=np.eye(4),
        current_opening=0.5,
    )
    events = [dict(step=104, time=5.0, state_sha256="state", kind=kind) for kind in ("observed", "reused")]
    branch = dict(runtime=dict(endpoint_cache_events=events), reference_captures=[copy.deepcopy(capture)])
    data = dict(captures=[capture])
    assert recovery.cached_boundary(branch, data)["original_capture_matched"] is True
    branch["runtime"]["endpoint_cache_events"][-1]["state_sha256"] = "different"
    with pytest.raises(ValueError, match="strictly reused"):
        recovery.cached_boundary(branch, data)
    branch["runtime"]["endpoint_cache_events"][-1]["state_sha256"] = "state"
    branch["reference_captures"][0]["canvas_rgb_sha256"] = "different"
    with pytest.raises(ValueError, match="current image or endpoint"):
        recovery.cached_boundary(branch, data)


def test_diagnostic_mismatch_must_remain_recorded(monkeypatch):
    paths = [
        "current_endpoint.sim_state_sha256",
        "render_before.sim_state_sha256",
        "render_after.sim_state_sha256",
    ]
    row = dict(
        status="closed",
        counts=dict(original_guard_passed=False, differing_paths=paths, control_steps=104, native_steps=2600),
        branch_ref="branch",
    )
    diagnostic = dict(
        complete=True,
        passed=True,
        owner_closed=True,
        sources_unchanged=True,
        phases=dict(producer=copy.deepcopy(row), checker=copy.deepcopy(row)),
    )
    monkeypatch.setattr(recovery, "_prepare_original", lambda _spec: dict(identity={}))
    monkeypatch.setattr(recovery.prefix, "read", lambda _ref: diagnostic)
    monkeypatch.setattr(recovery.prefix, "checked", lambda _ref: None)
    assert "capture_diagnostic_ref" in recovery.prepare(recovery.CASE)["identity"]
    diagnostic["phases"]["producer"]["counts"]["original_guard_passed"] = True
    with pytest.raises(ValueError, match="mismatch and cost"):
        recovery.prepare(recovery.CASE)

"""Check the independent audit's bridge gate without physics or runtime governor."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_wam_recovery.py"
    spec = importlib.util.spec_from_file_location("recovery_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits():
    return dict(
        max_translation_speed_m_s=0.2,
        max_rotation_speed_rad_s=1.0,
        max_opening_speed_s=2.0,
        position_soft_m=0.01,
        position_stop_m=0.04,
        rotation_soft_rad=0.1,
        rotation_stop_rad=0.4,
    )


def previous_target(replay):
    pose = np.eye(4)
    pose[0, 3] = 0.1
    pose[:3, :3] = Rotation.from_rotvec([0, 0, 0.3]).as_matrix()
    replay.chunk(np.eye(4), pose[None], [0.2], 0.8, 0.05)
    for _ in range(12):
        result, _ = replay.step(replay.admitted.copy(), 0.05)
    assert result["complete"]
    return pose


def test_bridge_recovery_reduces_both_errors_and_preserves_reference_opening():
    replay = auditor().RecoveryReplay(limits())
    previous_target(replay)
    summary = replay.chunk(np.eye(4), np.eye(4)[None], [0.4], 0.9, 0.05)
    assert summary["bridge_opening_preserved"] and summary["opening_reference_at_anchor"] == 0.2
    result, gate = replay.step(np.eye(4), 0.05)
    assert gate["allowed"] and result["bridge_recovery_advanced"]
    assert result["nominal_tracking_scale"] == 0 and result["model_time_s"] == 0
    assert all(b <= p for p, b in zip(gate["preceding"], gate["candidate"], strict=True))
    assert result["opening"] == 0.2


def test_position_improvement_cannot_pay_for_rotation_regression():
    replay = auditor().RecoveryReplay(limits())
    previous = previous_target(replay)
    replay.chunk(np.eye(4), np.eye(4)[None], [0.4], 0.9, 0.05)
    measured = previous.copy()
    measured[0, 3] = 0
    result, gate = replay.step(measured, 0.05)
    assert not gate["allowed"] and not result["bridge_recovery_advanced"]
    assert result["retimed_clock_s"] == 0
    assert gate["candidate"][0] < gate["preceding"][0] and gate["candidate"][1] > gate["preceding"][1]


def test_nonbridge_tracking_pause_remains_paused():
    replay = auditor().RecoveryReplay(limits())
    target = np.eye(4)
    target[0, 3] = 0.1
    replay.chunk(np.eye(4), target[None], [0.8], 0.8, 0.05)
    far = np.eye(4)
    far[1, 3] = 0.2
    result, gate = replay.step(far, 0.05)
    assert gate is None and not result["bridge_recovery_advanced"]
    assert result["retimed_clock_s"] == result["model_time_s"] == 0

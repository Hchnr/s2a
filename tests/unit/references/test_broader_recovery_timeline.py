"""Check the preregistered TCP geometry, timing, and evidence boundary."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.references.broader_recovery_timeline import broader_recovery_timeline
from s2a.references.training_recovery_timeline import training_recovery_timeline


@pytest.fixture
def native_reference():
    source = np.repeat(np.eye(4)[None], 47, axis=0)
    source[:, :3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix()
    source[:, :3, 3] = np.column_stack((np.linspace(0.1, 0.2, 47), np.linspace(-0.2, 0.1, 47), np.full(47, 0.4)))
    opening = np.linspace(0.9, 0.2, len(source))
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler("xy", [0.15, -0.25]).as_matrix()
    tool[:3, 3] = [0.03, -0.04, 0.12]
    return source, opening, tool


@pytest.mark.parametrize("variant", ["zero", "base_x", "base_y", "base_yaw"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_targets_keep_base_frame_tcp_pivot_and_original_native_timing(native_reference, variant, sign):
    source, opening, tool = native_reference
    before = source.copy(), opening.copy(), tool.copy()
    timeline, meta = broader_recovery_timeline(source, opening, tool, variant=variant, sign=sign)
    initial = source[0] @ tool
    assert timeline.periods == 56 + len(source) - 1
    assert meta["physical_labels_from_reference"] is False
    assert meta["reference_kind"] == "privileged_training_recovery_oracle"
    for cursor, fraction in [(0, 1 / 16), (15, 1), (35, 0.5), (55, 0)]:
        target, target_opening, _ = timeline.window(cursor)
        expected = initial.copy()
        if variant in ("base_x", "base_y"):
            expected[0 if variant == "base_x" else 1, 3] += fraction * sign * 0.06
        elif variant == "base_yaw":
            expected[:3, :3] = Rotation.from_rotvec([0, 0, fraction * sign * 0.5]).as_matrix() @ initial[:3, :3]
        np.testing.assert_allclose(target[0], expected, atol=1e-12)
        assert target_opening[0] == opening[0]
    for index in (1, 13, len(source) - 1):
        target, target_opening, _ = timeline.window(55 + index)
        np.testing.assert_allclose(target[0], source[index] @ tool, atol=1e-12)
        assert target_opening[0] == opening[index]
    tail, _, tail_meta = timeline.window(timeline.periods - 1)
    np.testing.assert_allclose(tail, np.repeat((source[-1] @ tool)[None], 40, axis=0), atol=1e-12)
    assert tail_meta["reference_endpoints"] == 1 and tail_meta["terminal_hold_endpoints"] == 39
    for after, original in zip((source, opening, tool), before, strict=True):
        np.testing.assert_array_equal(after, original)


@pytest.mark.parametrize(("variant", "sign", "offset"), [("zero", 1, 0.0), ("base_x", 1, 0.06), ("base_x", -1, -0.06)])
def test_prior_x_reference_agrees_for_every_endpoint(native_reference, variant, sign, offset):
    source, opening, tool = native_reference
    new, _ = broader_recovery_timeline(source, opening, tool, variant=variant, sign=sign)
    prior, _ = training_recovery_timeline(source, opening, tool, offset_m=offset)
    assert new.periods == prior.periods
    for cursor in range(new.periods):
        new_pose, new_opening, new_meta = new.window(cursor)
        old_pose, old_opening, old_meta = prior.window(cursor)
        np.testing.assert_allclose(new_pose, old_pose, atol=1e-12)
        np.testing.assert_array_equal(new_opening, old_opening)
        assert new_meta == old_meta


def test_rejects_invalid_sources_openings_tool_and_registration(native_reference):
    source, opening, tool = native_reference
    with pytest.raises(ValueError, match="full source endpoints"):
        broader_recovery_timeline(source[:40], opening[:40], tool, variant="zero", sign=1)
    with pytest.raises(ValueError, match="Openings"):
        broader_recovery_timeline(source, np.full(len(opening), np.nan), tool, variant="zero", sign=1)
    with pytest.raises(ValueError, match="SE\\(3\\)"):
        broader_recovery_timeline(source, opening, np.ones((4, 4)), variant="zero", sign=1)
    with pytest.raises(ValueError, match="variant"):
        broader_recovery_timeline(source, opening, tool, variant="base_z", sign=1)
    for sign in (0, True, 1.0):
        with pytest.raises(ValueError, match="sign"):
            broader_recovery_timeline(source, opening, tool, variant="zero", sign=sign)

import numpy as np
import pytest

from s2a.prediction_audit import audit_action_response, denormalize_action


def test_denormalization_masks_preserve_unscaled_rotation_and_gripper():
    spec = dict(normalization_mask=[True] * 3 + [False] * 125, scale=[100] * 3 + [1] * 125, bias=[0.1] * 3 + [0] * 125)
    normalized = np.zeros((2, 128))
    normalized[:, :10] = [0.2, 0.3, 0.4, 1, 0, 0, 0, 1, 0, 1.2]
    physical = denormalize_action(normalized, spec)
    np.testing.assert_allclose(physical[0, :3], [0.001, 0.002, 0.003])
    np.testing.assert_array_equal(physical[:, 3:], normalized[:, 3:])
    content = dict(
        action=physical.tolist(),
        action128=dict(
            normalized_action=normalized.tolist(),
            action_dim_mask=[True] * 10 + [False] * 118,
            action_units="raw_physical_or_source_units",
            action_domain="test",
            action_fps=20,
            action_timestamps_seconds=[0, 0.05],
        ),
    )
    result = audit_action_response(content, spec, domain="test", fps=20)
    assert result["denormalization_max_abs"] == 0 and result["opening_outside_fraction"] == 1
    content["action128"]["action_timestamps_seconds"] = [0, 0.1]
    with pytest.raises(ValueError, match="timestamps"):
        audit_action_response(content, spec, domain="test", fps=20)

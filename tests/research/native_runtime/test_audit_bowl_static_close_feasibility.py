"""Verify physical joint ranges and actual opening after a candidate closing."""

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_static_close_feasibility import opening_after_close


def test_opening_uses_native_joint_convention_and_rejects_overclosing():
    value = opening_after_close([0.03, -0.02], 0.01)
    np.testing.assert_allclose(value["qpos"], [0.02, -0.01])
    assert value["opening"] == pytest.approx(0.029 / 0.079)
    with pytest.raises(ValueError, match="within both native ranges"):
        opening_after_close([0.03, -0.02], 0.021)

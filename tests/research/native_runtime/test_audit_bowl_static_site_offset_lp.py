"""Validate the LP's signed intersection margin on simple boxes."""

import numpy as np
import pytest

from scripts.simulation.runtime.audit_bowl_static_site_offset_lp import box_slack


def box(center):
    return dict(center_world_m=center, rotation_world=np.eye(3), half_size_m=[1, 1, 1])


def test_lp_distinguishes_separated_touching_and_overlapping_boxes():
    assert box_slack(box([0, 0, 0]), box([3, 0, 0])) == pytest.approx(-0.5)
    assert box_slack(box([0, 0, 0]), box([2, 0, 0])) == pytest.approx(0)
    assert box_slack(box([0, 0, 0]), box([1.5, 0, 0])) == pytest.approx(0.25)
    assert box_slack(box([0, 0, 0]), box([0, 0, 0])) == pytest.approx(1)

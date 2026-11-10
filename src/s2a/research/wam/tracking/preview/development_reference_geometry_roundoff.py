"""A separately registered tolerance for static object rotation checks."""

import numpy as np

from s2a.research.wam.tracking.preview.development_reference_geometry import StaticReferenceProbe


class RoundoffBoundedProbe(StaticReferenceProbe):
    """Record forward-normalization drift without altering physical thresholds."""

    def __init__(self, model, target, budget=1201):
        super().__init__(model, target, budget)
        self.max_object_rotation_change = 0.0
        self.max_qpos_change_during_placement = 0.0

    def check_object_unchanged(self, position, rotation, qpos):
        np.testing.assert_array_equal(self.data.xpos[self.object_body], position)
        rotation_change = float(np.max(np.abs(self.data.xmat[self.object_body] - rotation)))
        qpos_change = float(np.max(np.abs(self.data.qpos - qpos)))
        self.max_object_rotation_change = max(self.max_object_rotation_change, rotation_change)
        self.max_qpos_change_during_placement = max(self.max_qpos_change_during_placement, qpos_change)
        np.testing.assert_allclose(self.data.xmat[self.object_body], rotation, atol=1e-12, rtol=0)

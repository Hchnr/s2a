"""Three task-neutral reference gains; distinct from a robot response inverse."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


def relative_features(anchor, poses):
    """Anchor-axis displacement and right-relative SO(3) logarithm, in m/rad."""
    anchor = validate_transform(finite(anchor, (4, 4)))
    poses = validate_transform(poses)
    if poses.ndim != 3 or not len(poses):
        raise ValueError("requires nonempty world endpoints")
    displacement = (poses[:, :3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
    rotation = Rotation.from_matrix(anchor[:3, :3].T @ poses[:, :3, :3]).as_rotvec()
    if np.any(np.linalg.norm(rotation, axis=1) >= np.pi - 1e-6):
        raise ValueError("anchor-relative rotation is ambiguous near pi")
    return displacement, rotation


@dataclass(frozen=True)
class ReferenceGains:
    position_gain: float = 1.0
    rotation_gain: float = 1.0
    opening_bias: float = 0.0

    def __post_init__(self):
        for value, low, high in (
            (self.position_gain, 0, 2),
            (self.rotation_gain, 0, 1),
            (self.opening_bias, -0.25, 0.25),
        ):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError("gains must be finite scalars")
            if not low <= value <= high:
                raise ValueError("reference gain outside frozen bounds")

    def apply(self, anchor, poses, openings):
        """Only measured anchor and already-generated references enter here.

        Orientation damping does not alter world positions. This differs from
        damping each local rotation before integrating local translations.
        Identity preserves every pose value exactly; openings always clip to
        their published range. No task, object, future state or phase input.
        """
        displacement, rotation = relative_features(anchor, poses)
        values = finite(openings, (len(displacement),))
        result = np.asarray(poses, dtype=float).copy()
        if self.position_gain != 1:
            result[:, :3, 3] = np.asarray(anchor)[:3, 3] + (
                self.position_gain * displacement @ np.asarray(anchor)[:3, :3].T
            )
        if self.rotation_gain != 1:
            result[:, :3, :3] = (
                np.asarray(anchor)[:3, :3] @ Rotation.from_rotvec(self.rotation_gain * rotation).as_matrix()
            )
        return result, np.clip(values + self.opening_bias, 0, 1)

    @classmethod
    def fit(cls, predicted, truth, predicted_opening, truth_opening, weights, *, ridge, scales):
        """Box-constrained weighted quadratic fit with an identity/zero prior.

        Input pose features are (displacement, rotvec), each Nx3 in one query's
        anchor axes. Caller owns episode splitting and must supply fit-only
        features, labels and weights. Weights sum to one; this fixes lambda's
        meaning independently of endpoint count. Opening fits pre-clip error.
        """
        w = finite(weights)
        if w.ndim != 1 or not len(w) or np.any(w <= 0) or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("requires positive endpoint weights summing to one")
        if isinstance(ridge, bool) or not np.isscalar(ridge) or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be a positive finite scalar")
        scales = finite(scales, (3,))
        if np.any(scales <= 0) or len(predicted) != 2 or len(truth) != 2:
            raise ValueError("requires position, rotation and opening scales")
        gains = []
        for channel, upper in ((0, 2), (1, 1)):
            x = finite(predicted[channel], (len(w), 3)) / scales[channel]
            y = finite(truth[channel], (len(w), 3)) / scales[channel]
            numerator = np.sum(w[:, None] * x * y) + ridge
            denominator = np.sum(w[:, None] * x * x) + ridge
            gains.append(float(np.clip(numerator / denominator, 0, upper)))
        x, y = finite(predicted_opening, (len(w),)), finite(truth_opening, (len(w),))
        bias = np.sum(w * (y - x)) / (1 + ridge * scales[2] ** 2)
        return cls(*gains, float(np.clip(bias, -0.25, 0.25)))

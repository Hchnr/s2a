"""Bounded, task-independent calibration of generated world reference endpoints.

This changes the model reference. It is neither a controller inverse nor a
planner. Deployment receives only the measured anchor and generated endpoints.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


@dataclass(frozen=True)
class CalibrationConfig:
    position_scale_m: float = 0.1
    rotation_scale_rad: float = 0.5
    maximum_horizon_seconds: float = 2.0
    ridge: float = 0.1
    maximum_translation_correction_speed_m_s: float = 0.05
    maximum_rotation_vector_correction_speed_rad_s: float = 0.25
    maximum_opening_correction_speed_s: float = 0.25
    relative_rotation_pi_margin_rad: float = 1e-6
    control_fps: int = 20

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.control_fps) is not int or self.relative_rotation_pi_margin_rad >= np.pi:
            raise ValueError("requires integer fps and a valid principal-rotation margin")


def reference_features(anchor, anchor_opening, poses, openings, times_seconds, *, config=CalibrationConfig()):
    """u=R0.T(p-p0), v=Log(R0.T R), with a fixed positive endpoint clock.

    No world task geometry, future measured state or labels enter this function.
    The seven pose features and two opening features are returned as copies.
    """
    anchor = validate_transform(anchor)
    poses = validate_transform(poses)
    if anchor.shape != (4, 4) or poses.ndim != 3 or not len(poses):
        raise ValueError("requires one measured anchor and nonempty world endpoints")
    count = len(poses)
    openings = finite(openings, (count,))
    current_opening = finite(anchor_opening, ())
    if not 0 <= current_opening <= 1:
        raise ValueError("measured anchor opening must lie in [0,1]")
    times = finite(times_seconds, (count,))
    expected_times = np.arange(1, count + 1, dtype=float) / config.control_fps
    if not np.array_equal(times, expected_times) or np.any(times > config.maximum_horizon_seconds):
        raise ValueError("requires consecutive endpoint times (j+1)/fps within the fixed horizon")
    position = (poses[:, :3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
    rotation = Rotation.from_matrix(anchor[:3, :3].T @ poses[:, :3, :3]).as_rotvec()
    if np.any(np.linalg.norm(rotation, axis=1) >= np.pi - config.relative_rotation_pi_margin_rad):
        raise ValueError("anchor-relative rotation is ambiguous near pi")
    normalized = np.column_stack((position / config.position_scale_m, rotation / config.rotation_scale_rad))
    return dict(
        pose_features=np.column_stack((normalized, times / config.maximum_horizon_seconds)),
        opening_features=np.column_stack((openings - current_opening, times / config.maximum_horizon_seconds)),
        normalized_local_pose=normalized.copy(),
        local_position_m=position.copy(),
        anchor_relative_rotation_vector_rad=rotation.copy(),
        raw_openings=openings.copy(),
        endpoint_times_seconds=times.copy(),
    )


def bounded_vectors(values, bounds):
    norms = np.linalg.norm(values, axis=1)
    factors = np.ones(len(values))
    active = norms > bounds
    factors[active] = bounds[active] / norms[active]
    return values * factors[:, None], active


@dataclass(frozen=True)
class GeneratedReferenceCalibration:
    pose_weights: tuple = field(default_factory=lambda: tuple((0.0,) * 6 for _ in range(7)))
    opening_weights: tuple = (0.0, 0.0)
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

    def __post_init__(self):
        if not isinstance(self.config, CalibrationConfig):
            raise ValueError("requires a CalibrationConfig")
        pose = finite(self.pose_weights, (7, 6))
        opening = finite(self.opening_weights, (2,))
        object.__setattr__(self, "pose_weights", tuple(tuple(map(float, row)) for row in pose))
        object.__setattr__(self, "opening_weights", tuple(map(float, opening)))

    def apply(self, anchor, anchor_opening, poses, openings, times_seconds):
        """Return corrected world endpoints, clipped openings and full diagnostics.

        R_out=R0 Exp(v+delta_v). Neither left/world multiplication nor serial
        reintegration of changed frame-to-frame deltas implements this mapping.
        """
        features = reference_features(anchor, anchor_opening, poses, openings, times_seconds, config=self.config)
        anchor, poses = np.asarray(anchor), np.asarray(poses)
        times = features["endpoint_times_seconds"]
        weights = np.asarray(self.pose_weights)
        raw = features["pose_features"] @ weights
        if not np.isfinite(raw).all():
            raise ValueError("pose correction is nonfinite")
        translation, position_cap = bounded_vectors(
            raw[:, :3] * self.config.position_scale_m,
            times * self.config.maximum_translation_correction_speed_m_s,
        )
        rotation, rotation_cap = bounded_vectors(
            raw[:, 3:] * self.config.rotation_scale_rad,
            times * self.config.maximum_rotation_vector_correction_speed_rad_s,
        )
        corrected_vector = features["anchor_relative_rotation_vector_rad"] + rotation
        if np.any(np.linalg.norm(corrected_vector, axis=1) >= np.pi - self.config.relative_rotation_pi_margin_rad):
            raise ValueError("corrected anchor-relative rotation is ambiguous near pi")
        # Preserve input dtype and all pose bytes when W is exactly zero. Each
        # unchanged component is also copied rather than needlessly recomposed.
        corrected = poses.copy() if not weights.any() else np.asarray(poses, dtype=float).copy()
        move = np.any(translation != 0, axis=1)
        turn = np.any(rotation != 0, axis=1)
        if np.any(move):
            corrected[move, :3, 3] = (
                anchor[:3, 3] + (features["local_position_m"][move] + translation[move]) @ anchor[:3, :3].T
            )
        if np.any(turn):
            corrected[turn, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec(corrected_vector[turn]).as_matrix()
        raw_opening = features["opening_features"] @ np.asarray(self.opening_weights)
        if not np.isfinite(raw_opening).all():
            raise ValueError("opening correction is nonfinite")
        limit = times * self.config.maximum_opening_correction_speed_s
        opening_delta = np.clip(raw_opening, -limit, limit)
        output_opening = np.clip(features["raw_openings"] + opening_delta, 0, 1)
        diagnostics = dict(
            **features,
            unconstrained_normalized_pose_residual=raw.copy(),
            translation_correction_m=translation.copy(),
            rotation_vector_correction_rad=rotation.copy(),
            translation_cap_active=position_cap,
            rotation_cap_active=rotation_cap,
            unconstrained_opening_residual=raw_opening.copy(),
            opening_correction=opening_delta.copy(),
            opening_cap_active=np.abs(raw_opening) > limit,
            pose_identity_weights=bool(not weights.any()),
        )
        return corrected, output_opening, diagnostics

    @classmethod
    def fit(cls, x, residual, opening_x, opening_residual, weights, *, config=CalibrationConfig()):
        """Fit-only normalized residual ridge with a zero-residual prior.

        Caller supplies only its training partition. Targets never enter apply.
        Positive weights sum to one, fixing ridge strength independently of N.
        """
        x = finite(x)
        if x.ndim != 2 or x.shape[1] != 7 or not len(x):
            raise ValueError("requires nonempty N×7 fit features")
        count = len(x)
        residual = finite(residual, (count, 6))
        opening_x = finite(opening_x, (count, 2))
        opening_residual = finite(opening_residual, (count,))
        weights = finite(weights, (count,))
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("fit weights must be positive and sum to one")
        pose_normal = x.T @ (weights[:, None] * x) + config.ridge * np.eye(7)
        opening_normal = opening_x.T @ (weights[:, None] * opening_x) + config.ridge * np.eye(2)
        pose = np.linalg.solve(pose_normal, x.T @ (weights[:, None] * residual))
        opening = np.linalg.solve(opening_normal, opening_x.T @ (weights * opening_residual))
        return cls(pose, opening, config)

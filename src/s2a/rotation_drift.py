"""Bounded world-left angular drift of generated references, not a robot inverse."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


def episode_query_weights(episodes, endpoint_counts):
    """Fit-only weights: each episode, its queries, then their endpoints equal."""
    if not episodes or len(episodes) != len(endpoint_counts):
        raise ValueError("requires nonempty episode/query endpoint counts")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in endpoint_counts):
        raise ValueError("endpoint counts must be positive integers")
    queries = Counter(episodes)
    return np.concatenate(
        [
            np.full(n, 1 / (len(queries) * queries[episode] * n))
            for episode, n in zip(episodes, endpoint_counts, strict=True)
        ]
    )


def world_rotation_residual(predicted, truth):
    """Principal Log(R_pred @ R_truth.T), in world axes; reject the pi branch."""
    predicted, truth = finite(predicted), finite(truth)
    if predicted.ndim != 3 or predicted.shape[1:] != (3, 3) or truth.shape != predicted.shape or not len(predicted):
        raise ValueError("requires equally sized nonempty Nx3x3 rotations")
    for rotations in (predicted, truth):
        if not (
            np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-7, rtol=0)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError("fit requires SO(3) rotations")
    residual = Rotation.from_matrix(predicted @ truth.transpose(0, 2, 1)).as_rotvec()
    if np.any(np.linalg.norm(residual, axis=1) >= np.pi - 1e-6):
        raise ValueError("world rotation residual is ambiguous near pi")
    return residual


def positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(value)


@dataclass(frozen=True)
class RotationDrift:
    bias_rad_per_second: tuple[float, float, float] = (0.0, 0.0, 0.0)
    maximum_bias_norm_rad_per_second: float = 0.25
    maximum_horizon_seconds: float = 2.0

    def __post_init__(self):
        bias = finite(self.bias_rad_per_second, (3,))
        bound = positive_scalar(self.maximum_bias_norm_rad_per_second, "bias norm bound")
        positive_scalar(self.maximum_horizon_seconds, "maximum horizon")
        if np.linalg.norm(bias) > bound + 1e-14:
            raise ValueError("angular bias exceeds its norm bound")
        object.__setattr__(self, "bias_rad_per_second", tuple(map(float, bias)))

    def times(self, values, count, *, ordered=True):
        times = finite(values, (count,))
        if (
            not count
            or np.any(times < 0)
            or np.any(times > self.maximum_horizon_seconds)
            or (ordered and np.any(np.diff(times) < 0))
        ):
            raise ValueError("requires ordered nonnegative reference times within the horizon")
        return times

    def apply(self, poses, openings, times_seconds):
        """Only generated world poses/openings and their endpoint clock enter.

        R_out(tau)=Exp(-tau*b)@R_in(tau). No GT, episode, task, contact or
        future measured state is accepted. Position/opening values stay exact;
        opening is never clipped here. Zero bias and time0 preserve rotations.
        """
        poses = validate_transform(poses)
        if poses.ndim != 3:
            raise ValueError("requires nonempty world pose endpoints")
        times = self.times(times_seconds, len(poses))
        values = np.asarray(openings)
        finite(values, (len(poses),))
        result = poses.copy()
        bias = np.asarray(self.bias_rad_per_second)
        active = times > 0
        if np.any(bias != 0) and np.any(active):
            correction = Rotation.from_rotvec(-times[active, None] * bias).as_matrix()
            result[active, :3, :3] = correction @ poses[active, :3, :3]
        return result, values.copy()

    @classmethod
    def fit(
        cls,
        predicted,
        truth,
        times_seconds,
        weights,
        *,
        ridge_seconds_squared=0.1,
        maximum_bias_norm_rad_per_second=0.25,
        maximum_horizon_seconds=2.0,
    ):
        """Fit-only weighted ridge, then Euclidean norm projection.

        Minimize sum(w*||Log(R_pred R_gt.T)-tau*b||²)+lambda*||b||².
        Caller owns whole-episode splitting. Positive weights must sum to one,
        fixing lambda's seconds² meaning independently of the endpoint count.
        """
        residual = world_rotation_residual(predicted, truth)
        prototype = cls((0, 0, 0), maximum_bias_norm_rad_per_second, maximum_horizon_seconds)
        # Flattened fit queries each restart tau; ordering is required only
        # within an apply call, not across concatenated independent queries.
        times = prototype.times(times_seconds, len(residual), ordered=False)
        ridge = positive_scalar(ridge_seconds_squared, "ridge")
        w = finite(weights, (len(residual),))
        if np.any(w <= 0) or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("requires positive endpoint weights summing to one")
        numerator = np.sum((w * times)[:, None] * residual, axis=0)
        denominator = float(np.sum(w * times**2) + ridge)
        bias = numerator / denominator
        norm = float(np.linalg.norm(bias))
        if norm > maximum_bias_norm_rad_per_second:
            bias *= maximum_bias_norm_rad_per_second / norm
        return cls(tuple(map(float, bias)), maximum_bias_norm_rad_per_second, maximum_horizon_seconds)

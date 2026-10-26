"""Train-only trajectory-held-out support calibration and causal command blending."""

import numpy as np
from scipy.spatial import cKDTree


def validate_support(points, groups):
    points = np.asarray(points, dtype=float)
    groups = np.asarray(groups)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all() or groups.shape != (len(points),):
        raise ValueError("Finite training points with one trajectory identity per row required")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent training trajectories required")
    return points, groups


def calibrate_support(points, groups):
    """95th percentile nearest other-trajectory distance; no evaluation labels."""
    points, groups = validate_support(points, groups)
    distances = np.empty(len(points))
    for group in np.unique(groups):
        selected = groups == group
        distances[selected] = cKDTree(points[~selected]).query(points[selected], workers=1)[0]
    radius = float(np.quantile(distances, 0.95, method="linear"))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Positive nondegenerate cross-trajectory calibration radius required")
    return dict(radius=radius, cross_trajectory_distances=distances)


class TrainingSupport:
    def __init__(self, points, radius):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Finite training condition matrix required")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Positive finite support radius required")
        self.tree = cKDTree(points.copy())
        self.dimension = points.shape[1]
        self.radius = float(radius)

    def query(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (self.dimension,) or not np.isfinite(point).all():
            raise ValueError("One finite current condition required")
        distance, index = self.tree.query(point, workers=1)
        weight = float(np.clip(2 - distance / self.radius, 0, 1))
        return dict(distance=float(distance), nearest_index=int(index), learned_weight=weight)


def blend_translation(learned, physical, previous, *, weight, lower, upper, dt=0.05, max_rate=5):
    """Blend current candidate commands, then project onto actual bounds and rate."""
    learned, physical, previous, lower, upper = (
        np.asarray(v, dtype=float) for v in (learned, physical, previous, lower, upper)
    )
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (learned, physical, previous, lower, upper)):
        raise ValueError("Finite three-component command vectors required")
    if (
        not np.isfinite(weight)
        or not 0 <= weight <= 1
        or not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(max_rate)
        or max_rate <= 0
    ):
        raise ValueError("Valid weight and positive finite rate/period required")
    if np.any(lower >= upper) or np.any(previous < lower) or np.any(previous > upper):
        raise ValueError("Ordered actual bounds and feasible previous command required")
    lo, hi = np.maximum(lower, previous - dt * max_rate), np.minimum(upper, previous + dt * max_rate)
    return np.clip(weight * learned + (1 - weight) * physical, lo, hi)

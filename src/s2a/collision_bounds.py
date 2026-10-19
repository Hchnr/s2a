"""Conservative separation of a generated SE(3) path and a bounded object."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


def separation_bound(poses, object_center, object_radius, geometries):
    """Lower bound for every gripper opening along linear/SLERP pose segments.

    Each geometry is enclosed by a sphere; its local center is fixed or lies
    on a line segment over all legal finger coordinates. No IK, actual tracking
    error or desired opening is assumed. A positive value certifies separation
    of this specified ideal gripper path, not of arbitrary other robot links.
    """
    poses = validate_transform(poses)
    center = finite(object_center, (3,))
    if poses.ndim != 3 or len(poses) < 2 or not np.isfinite(object_radius) or object_radius <= 0:
        raise ValueError("requires a path and a positive object bound")
    dp = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    rotation = poses[1:, :3, :3] @ poses[:-1, :3, :3].swapaxes(-1, -2)
    angle = Rotation.from_matrix(rotation).magnitude()
    endpoint_bounds, continuous_bounds = [], []
    for row in geometries.values():
        points = finite(row["corner_centers"])
        radius = float(row["geom_rbound_m"])
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("invalid local collision envelope")
        distance = np.sum((points[:, None] - points[None]) ** 2, axis=-1)
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        start, end = points[i], points[j]
        direction = end - start
        squared = float(np.dot(direction, direction))
        ratio = np.clip((points - start) @ direction / max(squared, 1e-30), 0, 1)
        if np.max(np.linalg.norm(points - start - ratio[:, None] * direction, axis=1)) > 1e-10:
            raise ValueError("geometry centers require more than one sliding coordinate")
        world_start = poses[:, :3, :3] @ start + poses[:, :3, 3]
        world_direction = poses[:, :3, :3] @ direction
        ratio = np.clip(np.sum((center - world_start) * world_direction, axis=1) / max(squared, 1e-30), 0, 1)
        nearest = world_start + ratio[:, None] * world_direction
        gaps = np.linalg.norm(nearest - center, axis=1) - radius - object_radius
        # Every path point is within half a segment of its nearest endpoint.
        # Translating contributes dp/2; a local center of length L rotated by
        # at most angle/2 contributes 2 L sin(angle/4). Opening is already fully
        # covered by the endpoint center segment.
        local_radius = max(np.linalg.norm(start), np.linalg.norm(end))
        swept = np.minimum(gaps[:-1], gaps[1:]) - dp / 2 - 2 * local_radius * np.sin(angle / 4)
        endpoint_bounds.append(float(np.min(gaps)))
        continuous_bounds.append(float(np.min(swept)))
    if not endpoint_bounds:
        raise ValueError("empty collision envelope")
    return dict(endpoint_lower_bound_m=min(endpoint_bounds), linear_slerp_lower_bound_m=min(continuous_bounds))

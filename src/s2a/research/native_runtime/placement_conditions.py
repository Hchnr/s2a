"""Offline reconstruction of the frozen LIBERO placement predicate components."""

import numpy as np


def placement_conditions(task_id, target, support, contact, *, site_position=None, site_rotation=None, site_size=None):
    """Match the recorded object/site ``On`` predicates for three fixed tasks.

    Inputs are world-frame saved geometry and an already reconstructed contact
    bit. They are diagnostic data and must not be supplied to an online policy.
    """
    target, support = np.asarray(target, dtype=float), np.asarray(support, dtype=float)
    if (
        task_id not in (6, 8, 9)
        or target.shape != (3,)
        or support.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(support).all()
        or type(contact) is not bool
    ):
        raise ValueError("A registered task, finite world positions and explicit contact are required")
    if task_id in (6, 8):
        xy = float(np.linalg.norm(target[:2] - support[:2]))
        z = float(target[2] - support[2])
        spatial = bool(z >= 0 and xy < 0.03)
        detail = dict(xy_distance_m=xy, xy_clearance_m=0.03 - xy, height_difference_m=z)
    else:
        site_position = np.asarray(site_position, dtype=float)
        site_rotation = np.asarray(site_rotation, dtype=float)
        site_size = np.asarray(site_size, dtype=float)
        if (
            site_position.shape != (3,)
            or site_rotation.shape != (3, 3)
            or site_size.shape != (3,)
            or not all(np.isfinite(x).all() for x in (site_position, site_rotation, site_size))
            or np.any(site_size <= 0)
        ):
            raise ValueError("Wine placement requires the recorded finite site pose and size")
        # Deliberately reproduce the pinned LIBERO SiteObject.under formula.
        local = site_rotation @ (target - site_position)
        low, high = float(site_size[2] - 0.005), float(site_size[2] + 0.1)
        spatial = bool(np.all(np.abs(local[:2]) < site_size[:2]) and low < local[2] < high)
        detail = dict(
            site_local_m=local.tolist(),
            xy_clearance_m=(site_size[:2] - np.abs(local[:2])).tolist(),
            lower_height_clearance_m=float(local[2] - low),
            upper_height_clearance_m=float(high - local[2]),
        )
    return dict(spatial_satisfied=spatial, support_contact=contact, reconstructed_success=spatial and contact, **detail)

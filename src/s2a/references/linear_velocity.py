"""083 finite differences of already generated source-position endpoints only."""

import numpy as np

from s2a.geometry import validate_transform


def reference_linear_velocity(poses, current_index, anchor_source, dt):
    targets = validate_transform(np.asarray(poses, dtype=float))
    anchor = validate_transform(np.asarray(anchor_source, dtype=float))
    if targets.ndim != 3 or not len(targets) or anchor.shape != (4, 4):
        raise ValueError("Generated source endpoints and measured query anchor required")
    if type(current_index) is not int or not 0 <= current_index < len(targets) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Valid available endpoint index and positive interval required")
    i = current_index
    if i + 1 < len(targets):
        previous = targets[i - 1, :3, 3] if i else anchor[:3, 3]
        velocity = (targets[i + 1, :3, 3] - previous) / (2 * dt)
        mode = "central_generated" if i else "central_query_anchor"
        before = i - 1
        after = i + 1
    else:
        previous = targets[i - 1, :3, 3] if i else anchor[:3, 3]
        velocity = (targets[i, :3, 3] - previous) / dt
        mode = "backward_tail" if i else "single_anchor"
        before = i - 1
        after = i
    return dict(
        velocity=velocity,
        mode=mode,
        current_index=i,
        before_index=before,
        after_index=after,
        has_previous_generated_reference=i > 0,
        has_next_generated_reference=i + 1 < len(targets),
    )

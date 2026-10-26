"""222 normalize saved JSON numeric arrays before the unchanged independent audit."""

import numpy as np
from rate_preview_independent import verify_plan as original_verify_plan


def verify_plan(plan, *args, **kwargs):
    normalized = dict(plan)
    for key in ("command", "planned_commands", "predicted_positions", "predicted_velocities"):
        normalized[key] = np.asarray(plan[key], dtype=float)
    return original_verify_plan(normalized, *args, **kwargs)

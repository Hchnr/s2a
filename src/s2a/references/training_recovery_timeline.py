"""Create a training-only offset/return oracle from registered native endpoints."""

import numpy as np

from s2a.geometry import validate_transform
from s2a.references.oracle_timeline import HORIZON, STRIDE, OracleTimeline
from s2a.research.wam.training.native.training_state_recovery import recovery_reference


def training_recovery_timeline(source_poses, openings, source_from_tcp, *, offset_m):
    """Expose the fixed 16/40 recovery schedule and subsequent native targets.

    These are desired references, not physical labels. The caller must obtain
    current feedback and any training labels from subsequent real execution.
    """
    tool = validate_transform(np.asarray(source_from_tcp, dtype=float))
    if tool.shape != (4, 4):
        raise ValueError("One fixed source-from-TCP transform is required")
    desired_source, desired_opening = recovery_reference(source_poses, openings, base_x_offset_m=offset_m)
    tcp = validate_transform(desired_source @ tool)
    windows = [
        dict(
            step=step, effective_targets_tcp=tcp[step : step + HORIZON], openings=desired_opening[step : step + HORIZON]
        )
        for step in range(0, len(tcp), STRIDE)
    ]
    timeline = OracleTimeline(windows, periods=len(tcp))
    return timeline, dict(
        offset_m=float(offset_m),
        perturb_steps=16,
        return_steps=40,
        periods=len(tcp),
        source_endpoints=len(np.asarray(source_poses)),
        reference_kind="privileged_training_recovery_oracle",
        physical_labels_from_reference=False,
    )

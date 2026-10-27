"""116 all198 intervention equations with source identities preserved."""

import numpy as np
from arm_condition_span_training import contrast_rows as span_rows
from arm_joint_wrench_expanded_training import contrast_rows as original_rows


def contrast_rows(config, e, parameters):
    x, y, base, records = original_rows(config, e, parameters)
    sx, sy, sb, metadata = span_rows(config, e, parameters)
    records = list(records) + [
        dict(row, dataset="115", equation_group=3000 + 3 * row["anchor_group"] + row["axis"]) for row in metadata
    ]
    assert len(records) == 198 and len({r["equation_group"] for r in records}) == 198
    return np.vstack((x, sx)), np.vstack((y, sy)), np.vstack((base, sb)), records

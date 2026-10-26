"""107 combine all original72 and new63 contrasts with distinct source identities."""

import numpy as np
from arm_wrench_coverage_training import contrast_rows as coverage_rows
from arm_wrench_training import contrast_rows as prior_rows


def contrast_rows(config, e, parameters):
    x, y, base, records = prior_rows(config, e, parameters)
    extra_x, extra_y, extra_base, metadata = coverage_rows(config, e, parameters)
    records = list(records)
    for row in metadata:
        records.append(dict(row, dataset="106", equation_group=2000 + 3 * row["anchor_group"] + row["axis"]))
    assert len(records) == 135 and len({r["equation_group"] for r in records}) == 135
    return np.vstack((x, extra_x)), np.vstack((y, extra_y)), np.vstack((base, extra_base)), records

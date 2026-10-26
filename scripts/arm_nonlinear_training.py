"""100 nonlinear producer merge of old9 and additional63 physically paired equations."""

import numpy as np
from arm_nonlinear_additional_training import contrast_rows as additional_rows
from arm_nonlinear_original_training import contrast_rows as original_rows


def contrast_rows(config, e, parameters):
    batches = []
    records = []
    for dataset, function, offset in (("086", original_rows, 100), ("097", additional_rows, 1000)):
        x, y, base, metadata = function(config, e, parameters)
        batches.append((x, y, base))
        for row in metadata:
            records.append(dict(row, dataset=dataset, equation_group=offset + row["anchor_group"] * 3 + row["axis"]))
    assert len(records) == 72 and len({r["equation_group"] for r in records}) == 72
    return *(np.vstack([batch[k] for batch in batches]) for k in range(3)), records

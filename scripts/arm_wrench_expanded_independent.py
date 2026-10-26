"""107 independent composition of all135 contrast equations."""

import numpy as np
from arm_wrench_coverage_independent import equations as coverage_equations
from arm_wrench_independent_equations import equations as prior_equations


def equations(config, e, parameters):
    x, y, baseline, identities = prior_equations(config, e, parameters)
    extra_x, extra_y, extra_baseline, extra_ids = coverage_equations(config, e, parameters)
    identities = list(identities) + [("106", *identity) for identity in extra_ids]
    assert len(identities) == 135 and len(set(identities)) == 135
    return np.vstack((x, extra_x)), np.vstack((y, extra_y)), np.vstack((baseline, extra_baseline)), identities

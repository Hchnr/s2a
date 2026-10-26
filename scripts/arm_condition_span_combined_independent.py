"""116 independent original135 plus additional63 equation composition."""

import numpy as np
from arm_condition_span_independent import equations as span_equations
from arm_joint_wrench_expanded_independent import equations as original_equations


def equations(config, e, parameters):
    x, y, base, ids = original_equations(config, e, parameters)
    sx, sy, sb, more = span_equations(config, e, parameters)
    ids = list(ids) + [("115", *identity) for identity in more]
    assert len(ids) == 198 and len(set(ids)) == 198
    return np.vstack((x, sx)), np.vstack((y, sy)), np.vstack((base, sb)), ids

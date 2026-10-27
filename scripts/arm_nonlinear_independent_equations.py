"""100 nonlinear independent old9/additional63 equation composition and identities."""

import numpy as np
from arm_nonlinear_additional_independent import equations as additional_equations
from arm_nonlinear_original_independent import equations as original_equations


def equations(config, e, parameters):
    batches = []
    identities = []
    for dataset, function in (("086", original_equations), ("097", additional_equations)):
        x, y, nominal, ids = function(config, e, parameters)
        batches.append((x, y, nominal))
        identities.extend((dataset, *identity) for identity in ids)
    assert len(identities) == 72 and len(set(identities)) == 72
    return *(np.vstack([batch[k] for batch in batches]) for k in range(3)), identities

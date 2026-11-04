"""Build a causal local model from newly evaluated complete response probes."""

import numpy as np

from s2a.geometry import finite
from s2a.research.trajectory_search.constraint_residual_repair import _squared
from s2a.research.trajectory_search.interior_trajectory_search import (
    _jacobian,
    _outputs,
    _require,
    causal_periods,
)
from s2a.research.trajectory_search.state_vector_gn_search import normalized_error_vector


def build_response_stencil(base, probes, signed_deltas):
    """Use all 72 signed nominal 1e-4 probes; reject future-state dependence.

    Callers must evaluate and independently verify these records with the new
    predictor. This function never loads historical values, evaluates a robot,
    repairs a stencil, or certifies a physical controller.
    """
    x = finite(base["commands"], (12, 6))
    deltas = finite(signed_deltas, (72,))
    _require(len(probes) == 72 and np.all(np.abs(deltas) == 1e-4), "Complete signed nominal 1e-4 stencil required")
    safety, squared = _outputs(base)
    error = normalized_error_vector(base)
    safety_columns, squared_columns, error_columns = [], [], []
    for axis, (probe, delta) in enumerate(zip(probes, deltas, strict=True)):
        command = finite(probe["commands"], (12, 6))
        displacement = command.ravel() - x.ravel()
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(x.ravel()[axis]), abs(command.ravel()[axis]))
        _require(
            np.all(displacement[np.arange(72) != axis] == 0)
            and displacement[axis] != 0
            and abs(displacement[axis] - delta) <= tolerance,
            "Each probe must change only its registered axis by its signed nominal displacement",
        )
        # Original compact command records omit redundant squared errors;
        # fresh records include them, and _squared checks them when present.
        probe_safety = finite(probe["safety_slack"], (4369,))
        probe_squared = _squared(probe).ravel()
        safety_columns.append((probe_safety - safety) / delta)
        squared_columns.append((probe_squared - squared) / delta)
        error_columns.append((normalized_error_vector(probe) - error) / delta)
    safety_periods, squared_periods = causal_periods()
    return dict(
        Js=_jacobian(np.asarray(safety_columns).T, safety_periods),
        Jq=_jacobian(np.asarray(squared_columns).T, squared_periods),
        Je=_jacobian(np.asarray(error_columns).T, np.repeat(np.arange(12), 6)),
    )

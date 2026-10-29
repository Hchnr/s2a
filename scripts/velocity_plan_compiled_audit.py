"""Independent exhaustive scalar search compiled without fast-math; original gates."""

import ctypes
import math

import numpy as np
from velocity_plan_independent_audit import scalar_step

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference

BUILD_REF = reference(REPO / "outputs/velocity_scalar_audit_native_20261028/build.json")
_build = read(BUILD_REF)
checked(_build["source_ref"])
_library = ctypes.CDLL(str(checked(_build["binary_ref"])))
_function = _library.velocity_enumerate
_double = ctypes.POINTER(ctypes.c_double)
_count = ctypes.POINTER(ctypes.c_longlong)
_function.argtypes = [
    _double,
    _double,
    _double,
    *([ctypes.c_double] * 4),
    _double,
    ctypes.c_int,
    _count,
    _double,
    _double,
]
_function.restype = ctypes.c_int


def enumerate_all(parameters, envelope, opening, motor, velocity, reference_values):
    mean = np.ascontiguousarray(parameters["mean"], dtype=np.float64)
    scale = np.ascontiguousarray(parameters["scale"], dtype=np.float64)
    coefficients = np.ascontiguousarray(parameters["coefficients"], dtype=np.float64)
    values = np.ascontiguousarray(reference_values, dtype=np.float64)
    horizon = min(8, len(values))
    assert mean.shape == scale.shape == (4,) and coefficients.shape == (15, 2) and 1 <= horizon <= 8
    assert all(np.isfinite(a).all() for a in (mean, scale, coefficients, values)) and np.all(scale > 0)
    counts = np.zeros(3, dtype=np.int64)
    best = ctypes.c_double(math.inf)
    sequence = np.zeros(horizon, dtype=np.float64)
    result = _function(
        mean.ctypes.data_as(_double),
        scale.ctypes.data_as(_double),
        coefficients.ctypes.data_as(_double),
        envelope,
        opening,
        velocity,
        float(motor[0]),
        values.ctypes.data_as(_double),
        horizon,
        counts.ctypes.data_as(_count),
        ctypes.byref(best),
        sequence.ctypes.data_as(_double),
    )
    assert result == 0 and int(counts.sum()) == 3**horizon
    return dict(
        valid_sequences=int(counts[0]),
        rejected_nonfinite_sequences=int(counts[1]),
        rejected_velocity_sequences=int(counts[2]),
        minimum_cost=best.value,
        best_sequence=tuple(sequence),
    )


def audit_plan(parameters, envelope, opening, motor, velocity, reference, plan):
    horizon = min(8, len(reference))
    assert plan["horizon"] == horizon and plan["reference_available_count"] == len(reference)
    assert plan["reference_endpoints"] == list(reference[:horizon]) and abs(velocity) <= envelope
    enumeration = enumerate_all(parameters, envelope, opening, motor, velocity, reference)
    for key in ("valid_sequences", "rejected_nonfinite_sequences", "rejected_velocity_sequences"):
        assert enumeration[key] == plan[key]
    sequence = tuple(plan["sequence"])
    assert sequence and plan["command"] == sequence[0] and len(sequence) == horizon
    q, v, m = opening, velocity, float(motor[0])
    cost = 0.0
    for i, u in enumerate(sequence):
        assert u in (0.0, -1.0, 1.0)
        q, v, m = scalar_step(parameters, q, v, m, u)
        assert abs(v) <= envelope
        for value, key in [
            (q, "predicted_openings"),
            (v, "predicted_period_velocities"),
            (m, "predicted_motor_targets"),
        ]:
            assert abs(value - plan[key][i]) <= 1e-10
        cost += (q - reference[i]) ** 2 + 0.0001 * u * u
    assert abs(cost - plan["cost"]) <= 1e-10 and cost <= enumeration["minimum_cost"] + 1e-10
    return dict(
        valid_sequences=enumeration["valid_sequences"],
        minimum_cost=enumeration["minimum_cost"],
        chosen_cost=cost,
        optimality_gap=cost - enumeration["minimum_cost"],
        scalar_best_sequence=enumeration["best_sequence"],
        numerical_near_tie=sequence != enumeration["best_sequence"],
    )

"""Independent scalar depth-first full080 optimum audit; no producer imports."""

import math


def scalar_step(parameters, q, v, m, u):
    before = m
    for _ in range(25):
        m = max(-1.0, min(1.0, m - 0.01 * u))
    z = [
        (value - mean) / scale
        for value, mean, scale in zip([q, before, m, v], parameters["mean"], parameters["scale"], strict=True)
    ]
    x = z + [z[i] * z[j] for i in range(4) for j in range(i, 4)] + [1.0]
    residual = [math.fsum(x[i] * parameters["coefficients"][i][column] for i in range(15)) for column in (0, 1)]
    rawq = q + v + residual[0]
    nextv = v + residual[1]
    if not math.isfinite(rawq) or not math.isfinite(nextv):
        raise ValueError("Nonfinite hypothetical state")
    return max(0.0, min(1.0, rawq)), nextv, m


def audit_plan(parameters, envelope, opening, motor, velocity, reference, plan):
    horizon = min(8, len(reference))
    assert plan["horizon"] == horizon and plan["reference_available_count"] == len(reference)
    assert plan["reference_endpoints"] == list(reference[:horizon])
    assert abs(velocity) <= envelope
    valid = nonfinite = unsupported = 0
    best_cost = math.inf
    best_sequence = None

    def visit(depth, q, v, m, cost, sequence):
        nonlocal valid, nonfinite, unsupported, best_cost, best_sequence
        if depth == horizon:
            valid += 1
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence
            return
        for u in (0.0, -1.0, 1.0):
            try:
                nq, nv, nm = scalar_step(parameters, q, v, m, u)
            except (ValueError, OverflowError):
                nonfinite += 3 ** (horizon - depth - 1)
                continue
            if abs(nv) > envelope:
                unsupported += 3 ** (horizon - depth - 1)
                continue
            visit(depth + 1, nq, nv, nm, cost + (nq - reference[depth]) ** 2 + 0.0001 * u * u, sequence + (u,))

    visit(0, opening, velocity, float(motor[0]), 0.0, ())
    assert (
        valid == plan["valid_sequences"]
        and nonfinite == plan["rejected_nonfinite_sequences"]
        and unsupported == plan["rejected_velocity_sequences"]
    )
    assert valid + nonfinite + unsupported == 3**horizon
    sequence = tuple(plan["sequence"])
    assert sequence and plan["command"] == sequence[0]
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
    assert len(sequence) == horizon and abs(cost - plan["cost"]) <= 1e-10 and cost <= best_cost + 1e-10
    return dict(
        valid_sequences=valid,
        minimum_cost=best_cost,
        chosen_cost=cost,
        optimality_gap=cost - best_cost,
        scalar_best_sequence=best_sequence,
        numerical_near_tie=sequence != best_sequence,
    )

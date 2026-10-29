"""Independent scalar073 replay without producer feature, adapter or quantizer imports."""

import math


def replay(parameters, opening, current_motor, previous_opening, reference):
    assert parameters["target"] == "next_motor_minus_current_motor"
    assert parameters["baseline"] == "current_motor"
    assert len(reference) and all(math.isfinite(v) and 0 <= v <= 1 for v in reference)
    assert math.isfinite(opening) and 0 <= opening <= 1 and -1 <= current_motor <= 1
    delta = 0.0 if previous_opening is None else opening - previous_opening
    offsets = (0, 1, 3, 7)
    raw = [opening, current_motor, delta] + [float(reference[min(i, len(reference) - 1)]) - opening for i in offsets]
    z = [(v - m) / s for v, m, s in zip(raw, parameters["mean"], parameters["scale"], strict=True)]
    available = [i < len(reference) for i in offsets]
    features = z + [z[i] * z[j] for i in range(7) for j in range(i, 7)] + [float(v) for v in available] + [1.0]
    predicted = max(
        -1.0, min(1.0, current_motor + sum(a * b for a, b in zip(features, parameters["coefficients"], strict=True)))
    )
    targets = [max(-1.0, min(1.0, current_motor + delta)) for delta in (0.0, 0.25, -0.25)]
    chosen = min(range(3), key=lambda i: (targets[i] - predicted) ** 2)
    return dict(
        command=(0.0, -1.0, 1.0)[chosen],
        predicted_motor_target=predicted,
        selected_motor_target=targets[chosen],
        previous_opening_delta=delta,
        available=available,
    )
